"""Pytest fixtures: static HTTP server and a shared Chromium browser."""

from __future__ import annotations

import socket
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.helpers import REPO_ROOT


class _SilentHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        '.html': 'text/html; charset=utf-8',
        '.js': 'text/javascript; charset=utf-8',
        '.json': 'application/json; charset=utf-8',
        '.geojson': 'application/geo+json; charset=utf-8',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def log_message(self, format, *args):
        return


def pytest_addoption(parser):
    parser.addoption(
        '--headed',
        action='store_true',
        default=False,
        help='Show the browser window while UI tests run',
    )
    parser.addoption(
        '--runslow',
        action='store_true',
        default=False,
        help='Run long-running checks (live PDF export)',
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'slow: long-running UI checks such as live PDF rasterization',
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption('--runslow'):
        return
    skip_slow = pytest.mark.skip(reason='pass --runslow to include live PDF export')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope='session')
def base_url():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()

    server = ThreadingHTTPServer(('127.0.0.1', port), _SilentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{port}'
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture(scope='session')
def browser(pytestconfig):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        pytest.skip(f'playwright is not installed: {exc}')

    headed = bool(pytestconfig.getoption('--headed'))
    playwright = sync_playwright().start()
    try:
        try:
            instance = playwright.chromium.launch(headless=not headed)
        except Exception as exc:
            pytest.skip(
                'Chromium is not available. Run: python -m playwright install chromium '
                f'({exc})'
            )
        yield instance
        instance.close()
    finally:
        playwright.stop()


@pytest.fixture
def context(browser):
    ctx = browser.new_context(
        viewport={'width': 1440, 'height': 900},
        locale='en-US',
        accept_downloads=True,
    )
    yield ctx
    ctx.close()


@pytest.fixture
def page(context):
    page = context.new_page()
    page.set_default_timeout(20_000)
    errors = []

    def on_page_error(exc):
        errors.append(f'pageerror: {exc}')

    page.on('pageerror', on_page_error)
    page._pcvs_errors = errors
    yield page
    leftover = list(page._pcvs_errors)
    page.close()
    if leftover:
        pytest.fail('Uncaught JavaScript errors:\n' + '\n'.join(leftover))
