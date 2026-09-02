#!/usr/bin/env python3
"""Regenerate maps (optional) and run the UI/UX regression suite.

Typical use after changing the renderer or the viewer:

    python verify.py --generate --power 4.0 --gradient-sharp 18.0 --kdtree

Run the tests against maps already on disk:

    python verify.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _run(command: list[str]) -> int:
    print('\n>', ' '.join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT)
    return completed.returncode


def _ensure_chromium() -> None:
    from playwright.sync_api import sync_playwright

    playwright = sync_playwright().start()
    try:
        browser = playwright.chromium.launch()
        browser.close()
    except Exception:
        print('Chromium missing for Playwright; installing…', flush=True)
        code = _run([sys.executable, '-m', 'playwright', 'install', 'chromium'])
        if code != 0:
            raise SystemExit('Failed to install Playwright Chromium.')
    finally:
        playwright.stop()


def _generate(args: argparse.Namespace) -> None:
    geojson = REPO_ROOT / (args.geojson_dir or 'GEOJSON')
    if not geojson.is_dir():
        raise SystemExit(
            f'Cannot generate maps: {geojson} is missing. '
            'Place point + coastline GeoJSON pairs there first.'
        )

    command = [
        sys.executable,
        str(REPO_ROOT / 'render_map.py'),
        '--power', str(args.power),
        '--gradient-sharp', str(args.gradient_sharp),
        '--geojson-dir', args.geojson_dir,
    ]
    if args.kdtree:
        command.append('--kdtree')
    elif args.brute:
        command.append('--brute')
    if args.pdf:
        command.append('--pdf')
    if args.map:
        command.append('--map')
        command.extend(str(item) for item in args.map)

    code = _run(command)
    if code != 0:
        raise SystemExit(f'Map generation failed with exit code {code}.')


def _test(args: argparse.Namespace) -> None:
    _ensure_chromium()
    command = [sys.executable, '-m', 'pytest', str(REPO_ROOT / 'tests')]
    if args.headed:
        command.append('--headed')
    if args.slow:
        command.append('--runslow')
    if args.pytest_args:
        command.extend(args.pytest_args)

    code = _run(command)
    if code != 0:
        raise SystemExit(f'UI/UX tests failed with exit code {code}.')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate paleoclimate maps and/or run the UI/UX regression suite.',
    )
    parser.add_argument(
        '--generate',
        action='store_true',
        help='Run render_map.py before the tests (same defaults as the published maps)',
    )
    parser.add_argument('--power', type=float, default=4.0,
                        help='IDW/KNN power used with --generate (default: 4.0)')
    parser.add_argument('--gradient-sharp', type=float, default=18.0,
                        help='Color-ramp sharpness used with --generate (default: 18.0)')
    parser.add_argument('--geojson-dir', default='GEOJSON')
    parser.add_argument('--kdtree', action='store_true', default=True,
                        help='Use the k-d tree backend (default)')
    parser.add_argument('--brute', action='store_true',
                        help='Use brute-force neighbor search instead of --kdtree')
    parser.add_argument(
        '--map',
        nargs='+',
        metavar='DATASET',
        help='With --generate, render only these datasets (e.g. 110 115)',
    )
    parser.add_argument(
        '--pdf',
        action='store_true',
        help='With --generate, also write pre-rendered PDFs',
    )
    parser.add_argument('--skip-tests', action='store_true',
                        help='Generate only; do not run the UI/UX suite')
    parser.add_argument('--headed', action='store_true',
                        help='Show the browser while tests run')
    parser.add_argument('--slow', action='store_true',
                        help='Include live PDF export tests')
    parser.add_argument(
        'pytest_args',
        nargs=argparse.REMAINDER,
        help='Extra pytest arguments, after --',
    )
    args = parser.parse_args()
    if args.brute:
        args.kdtree = False

    extra = list(args.pytest_args or [])
    if extra and extra[0] == '--':
        extra = extra[1:]
    args.pytest_args = extra

    if args.generate:
        _generate(args)
    elif args.skip_tests:
        raise SystemExit('Nothing to do: pass --generate or omit --skip-tests.')

    if not args.skip_tests:
        _test(args)

    print('\nVerification finished.')


if __name__ == '__main__':
    main()
