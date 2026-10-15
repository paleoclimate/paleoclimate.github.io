#!/usr/bin/env python3
"""Encrypt the published HTML with StatiCrypt.

The viewer, the add-points page, each generated map and the comparison pages
are replaced with a password shell. The editable add-points form stays in
``templates/add-point.html``. Map and viewer HTML are encrypted in place after
``render_map.py`` writes them.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from site_crypto import REPO_ROOT, TEAM_PASSWORD, is_encrypted, load_salt

TEMPLATE = REPO_ROOT / 'staticrypt' / 'password_template.html'
ADD_POINT_SOURCE = REPO_ROOT / 'templates' / 'add-point.html'
STATICRYPT_CLI = REPO_ROOT / 'node_modules' / 'staticrypt' / 'cli' / 'index.js'
GATE_TAGS = (
    '<script src="../viewer_gate.js"></script>',
    '<script src="viewer_gate.js"></script>',
)


def _ensure_staticrypt() -> None:
    if STATICRYPT_CLI.is_file():
        return
    subprocess.run(
        ['npm', 'install', '--no-fund', '--no-audit'],
        cwd=REPO_ROOT,
        check=True,
    )
    if not STATICRYPT_CLI.is_file():
        raise SystemExit('staticrypt is not installed after npm install')


def published_html() -> list[Path]:
    files = [REPO_ROOT / 'index.html']
    maps = REPO_ROOT / 'GENERATED_IDW_MAPS'
    comparison = REPO_ROOT / 'COMPARISON'
    if maps.is_dir():
        files.extend(sorted(maps.glob('map_*.html')))
    if comparison.is_dir():
        files.extend(sorted(comparison.glob('*.html')))
    return [path for path in files if path.is_file()]


def _strip_gate(text: str) -> str:
    for tag in GATE_TAGS:
        text = text.replace(tag, '')
    return text


def encrypt_html(source: Path, destination: Path, *, force: bool) -> None:
    text = source.read_text(encoding='utf-8')
    if is_encrypted(text):
        if force:
            raise SystemExit(f'{source} is already ciphertext; encrypt the plaintext source')
        return
    text = _strip_gate(text)
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        incoming = workspace / 'plain.html'
        outgoing = workspace / 'out'
        incoming.write_text(text, encoding='utf-8')
        outgoing.mkdir()
        environment = os.environ.copy()
        environment['STATICRYPT_PASSWORD'] = TEAM_PASSWORD
        subprocess.run(
            [
                'node',
                str(STATICRYPT_CLI),
                str(incoming),
                '--salt',
                load_salt(),
                '--remember',
                'false',
                '--template',
                str(TEMPLATE),
                '--template-title',
                'PCVS',
                '--template-instructions',
                'This atlas is not public. Enter the team password.',
                '--template-button',
                'Enter',
                '--template-placeholder',
                'Password',
                '--template-error',
                'Wrong password.',
                '--template-toggle-show',
                'Show password',
                '--template-toggle-hide',
                'Hide password',
                '--config',
                'false',
                '--directory',
                str(outgoing),
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
        produced = outgoing / 'plain.html'
        if not produced.is_file():
            found = list(outgoing.rglob('*.html'))
            if len(found) != 1:
                raise SystemExit(f'StatiCrypt wrote {found} for {source}')
            produced = found[0]
        payload = produced.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Windows refuses to truncate a file another process has mapped (EINVAL).
    # Write aside, then replace, and keep trying until the lock clears.
    temporary = destination.with_name(destination.name + '.writing')
    temporary.write_bytes(payload)
    last_error = None
    for attempt in range(12):
        try:
            os.replace(temporary, destination)
            last_error = None
            break
        except OSError as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    if last_error is not None:
        temporary.unlink(missing_ok=True)
        raise last_error
    shown = destination.resolve()
    try:
        shown = shown.relative_to(REPO_ROOT)
    except ValueError:
        pass
    print(f'Encrypted {shown}')


def encrypt_published() -> None:
    if not TEMPLATE.is_file():
        raise SystemExit(f'Missing {TEMPLATE}')
    if not ADD_POINT_SOURCE.is_file():
        raise SystemExit(f'Missing {ADD_POINT_SOURCE}')
    _ensure_staticrypt()
    encrypt_html(ADD_POINT_SOURCE, REPO_ROOT / 'add-point.html', force=True)
    for path in published_html():
        encrypt_html(path, path, force=False)


if __name__ == '__main__':
    try:
        encrypt_published()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode or 1)
