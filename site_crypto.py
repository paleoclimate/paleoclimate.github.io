"""Password and AES helpers shared by StatiCrypt publish and the test suite.

The published HTML is ciphertext. These functions reproduce StatiCrypt 3's
PBKDF2 chain and AES-CBC so a test can read a page without a browser. The
team password is the one the suite already typed into the old gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

REPO_ROOT = Path(__file__).resolve().parent
SALT_PATH = REPO_ROOT / '.staticrypt.json'
TEAM_PASSWORD = 'Paleoclima-2026-entrega'
ENCRYPTED_MARKER = 'staticryptEncryptedMsgUniqueVariableName'

_HASHES: dict[str, str] = {}


def load_salt() -> str:
    payload = json.loads(SALT_PATH.read_text(encoding='utf-8'))
    salt = str(payload['salt']).lower()
    if len(salt) != 32 or any(char not in '0123456789abcdef' for char in salt):
        raise ValueError(f'{SALT_PATH} salt must be 32 hex characters')
    return salt


def _pbkdf2(password: str, salt: str, iterations: int, algorithm: str) -> str:
    derived = hashlib.pbkdf2_hmac(
        algorithm,
        password.encode('utf-8'),
        salt.encode('utf-8'),
        iterations,
        dklen=32,
    )
    return derived.hex()


def hash_password(password: str, salt: str) -> str:
    """StatiCrypt 3's three-round hash. The result is the AES-256 key, as hex."""
    cached = _HASHES.get(salt)
    if cached is not None and password == TEAM_PASSWORD:
        return cached
    hashed = _pbkdf2(password, salt, 1000, 'sha1')
    hashed = _pbkdf2(hashed, salt, 14000, 'sha256')
    hashed = _pbkdf2(hashed, salt, 585000, 'sha256')
    if password == TEAM_PASSWORD:
        _HASHES[salt] = hashed
    return hashed


def passphrase_hash() -> str:
    return hash_password(TEAM_PASSWORD, load_salt())


def is_encrypted(text: str) -> bool:
    return ENCRYPTED_MARKER in text


def decrypt_page(text: str) -> str:
    message = re.search(
        rf'"{ENCRYPTED_MARKER}":\s*"([0-9a-f]+)"',
        text,
    )
    salt = re.search(r'"staticryptSaltUniqueVariableName":\s*"([0-9a-f]+)"', text)
    if message is None or salt is None:
        raise ValueError('Encrypted page is missing the StatiCrypt payload')
    key = bytes.fromhex(hash_password(TEAM_PASSWORD, salt.group(1)))
    # HMAC-SHA256 (64 hex chars) then the AES payload (IV + ciphertext).
    raw = bytes.fromhex(message.group(1)[64:])
    iv, ciphertext = raw[:16], raw[16:]
    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    plain = unpadder.update(padded) + unpadder.finalize()
    return plain.decode('utf-8')


def read_html(path: Path) -> str:
    text = Path(path).read_text(encoding='utf-8')
    if not is_encrypted(text):
        return text
    return decrypt_page(text)
