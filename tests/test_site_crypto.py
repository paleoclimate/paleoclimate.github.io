"""The team password stays out of the repository."""

from __future__ import annotations

import pytest

import site_crypto


def test_team_password_comes_from_outside_the_repo(monkeypatch, tmp_path):
    monkeypatch.setattr(site_crypto, 'PASSWORD_PATH', tmp_path / 'missing')
    monkeypatch.delenv('STATICRYPT_PASSWORD', raising=False)
    with pytest.raises(RuntimeError, match='STATICRYPT_PASSWORD'):
        site_crypto.team_password()

    secret_file = tmp_path / 'secret'
    secret_file.write_text('from-file\n', encoding='utf-8')
    monkeypatch.setattr(site_crypto, 'PASSWORD_PATH', secret_file)
    assert site_crypto.team_password() == 'from-file'

    monkeypatch.setenv('STATICRYPT_PASSWORD', 'from-env')
    assert site_crypto.team_password() == 'from-env'


def test_published_pdf_bytes_are_not_a_pdf(monkeypatch, tmp_path):
    monkeypatch.setenv('STATICRYPT_PASSWORD', 'from-env')
    monkeypatch.setattr(site_crypto, 'PASSWORD_PATH', tmp_path / 'missing')
    site_crypto._HASHES.clear()
    plain = b'%PDF-1.4\n' + (b'0' * 64)
    sealed = site_crypto.encrypt_pdf_bytes(plain)
    assert sealed.startswith(site_crypto.PDF_MAGIC)
    assert not sealed.startswith(b'%PDF-')
    assert site_crypto.decrypt_pdf_bytes(sealed) == plain
    path = tmp_path / 'map_raster.pdf'
    path.write_bytes(plain)
    assert site_crypto.encrypt_pdf(path) is True
    assert path.read_bytes().startswith(site_crypto.PDF_MAGIC)
    assert site_crypto.encrypt_pdf(path) is False
