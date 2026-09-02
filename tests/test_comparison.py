"""UI/UX of the Floegel comparison reports and the GCP picker."""

from __future__ import annotations

import json
import re

from tests.helpers import COMPARISON_AGES, COMPARISON_DIR, repo_url


def test_comparison_index_lists_both_ages(page, base_url):
    page.goto(repo_url(base_url, 'COMPARISON/index_comparison.html'),
              wait_until='domcontentloaded')
    assert page.locator('h1').count() >= 1
    links = page.locator('a')
    hrefs = [links.nth(i).get_attribute('href') for i in range(links.count())]
    for age in COMPARISON_AGES:
        assert any(href and f'comparison_report_{age}.html' in href for href in hrefs)
        assert any(href and f'metrics_{age}.csv' in href for href in hrefs)


def test_comparison_index_links_open_reports(page, base_url):
    page.goto(repo_url(base_url, 'COMPARISON/index_comparison.html'),
              wait_until='domcontentloaded')
    page.locator('a', has_text=re.compile(r'105')).first.click()
    page.wait_for_url(re.compile(r'comparison_report_105'))
    assert page.locator('h1').inner_text()
    assert '105' in page.locator('h1').inner_text()


def test_comparison_reports_show_metrics_tables_and_images(page, base_url):
    for age in COMPARISON_AGES:
        page.goto(
            repo_url(base_url, f'COMPARISON/comparison_report_{age}.html'),
            wait_until='domcontentloaded',
            timeout=60_000,
        )
        heading = page.locator('h1').inner_text()
        assert str(age) in heading
        body = page.locator('body').inner_text()
        for token in ('Similaridade', 'Kappa', 'IoU', 'Matriz'):
            assert token in body, f'{age} Ma report is missing {token}'

        cards = page.locator('.card .big')
        assert cards.count() >= 3
        accuracy = cards.nth(0).inner_text().strip()
        assert '%' in accuracy
        assert 0 <= float(accuracy.replace('%', '').replace(',', '.')) <= 100

        tables = page.locator('table')
        assert tables.count() >= 2

        images = page.evaluate(
            """() => Array.from(document.images).map(img => ({
                 complete: img.complete,
                 width: img.naturalWidth,
                 height: img.naturalHeight,
               }))"""
        )
        assert images, f'{age} Ma report has no images'
        for image in images:
            assert image['complete']
            assert image['width'] > 10
            assert image['height'] > 10


def test_metrics_csv_is_served_and_matches_report_age(page, base_url):
    for age in COMPARISON_AGES:
        response = page.request.get(repo_url(base_url, f'COMPARISON/metrics_{age}.csv'))
        assert response.ok
        text = response.text()
        assert str(age) in text
        assert 'IoU' in text or 'Kappa' in text
        disk = (COMPARISON_DIR / f'metrics_{age}.csv').read_text(encoding='utf-8')
        assert 'Acuracia' in disk or 'Kappa' in disk


def test_gcp_files_have_enough_control_points():
    for age in COMPARISON_AGES:
        payload = json.loads((COMPARISON_DIR / f'gcp_{age}.json').read_text(encoding='utf-8'))
        points = payload.get('points') or payload.get('pairs') or []
        assert len(points) >= 4, f'gcp_{age}.json needs at least 4 pairs'
        assert int(payload.get('age', age)) == int(age)


def test_gcp_picker_loads_both_canvases(page, base_url):
    page.goto(repo_url(base_url, 'COMPARISON/gcp_picker.html'),
              wait_until='domcontentloaded', timeout=60_000)
    page.wait_for_function(
        """() => {
          const left = document.getElementById('cvFloegel');
          const right = document.getElementById('cvRef');
          return left && right && left.width > 20 && right.width > 20;
        }""",
        timeout=30_000,
    )
    options = page.locator('#ageSel option')
    values = {options.nth(i).get_attribute('value') for i in range(options.count())}
    assert {'105', '115'} <= values
    assert page.locator('#undoBtn').is_visible()
    assert page.locator('#clearBtn').is_visible()
    assert page.locator('#exportBtn').is_visible()
    assert 'FLOEGEL' in page.locator('#status').inner_text()


def test_gcp_picker_pair_undo_clear_and_validation(page, base_url):
    page.goto(repo_url(base_url, 'COMPARISON/gcp_picker.html'),
              wait_until='domcontentloaded', timeout=60_000)
    page.wait_for_function(
        """() => document.getElementById('cvFloegel')
                 && document.getElementById('cvFloegel').width > 20""",
        timeout=30_000,
    )

    dialogs = []

    def on_dialog(dialog):
        dialogs.append(dialog.message)
        dialog.accept()

    page.on('dialog', on_dialog)

    page.locator('#cvRef').click(position={'x': 40, 'y': 40})
    page.wait_for_timeout(200)
    assert dialogs, 'Clicking the render first should warn the user'
    assert 'Floegel' in dialogs[-1]

    page.locator('#cvFloegel').click(position={'x': 50, 'y': 50})
    assert 'RENDER' in page.locator('#status').inner_text()
    page.locator('#cvRef').click(position={'x': 60, 'y': 60})
    assert 'Pares: 1' in page.locator('#status').inner_text()
    assert page.locator('#list li').count() == 1

    page.locator('#undoBtn').click()
    assert 'Pares: 0' in page.locator('#status').inner_text()

    page.locator('#cvFloegel').click(position={'x': 40, 'y': 40})
    page.locator('#cvRef').click(position={'x': 45, 'y': 45})
    page.locator('#clearBtn').click()
    assert 'Pares: 0' in page.locator('#status').inner_text()

    page.locator('#exportBtn').click()
    page.wait_for_timeout(200)
    assert any('4' in message or 'pares' in message.lower() for message in dialogs)


def test_gcp_picker_export_after_four_pairs(page, base_url):
    page.goto(repo_url(base_url, 'COMPARISON/gcp_picker.html'),
              wait_until='domcontentloaded', timeout=60_000)
    page.wait_for_function(
        """() => document.getElementById('cvFloegel')
                 && document.getElementById('cvFloegel').width > 20""",
        timeout=30_000,
    )
    page.on('dialog', lambda dialog: dialog.accept())

    for index in range(4):
        offset = 30 + index * 12
        page.locator('#cvFloegel').click(position={'x': offset, 'y': offset})
        page.locator('#cvRef').click(position={'x': offset + 8, 'y': offset + 4})
    assert 'Pares: 4' in page.locator('#status').inner_text()

    with page.expect_download() as download_info:
        page.locator('#exportBtn').click()
    download = download_info.value
    assert download.suggested_filename.startswith('gcp_')
    assert download.suggested_filename.endswith('.json')


def test_gcp_picker_age_switch_keeps_state_per_age(page, base_url):
    page.goto(repo_url(base_url, 'COMPARISON/gcp_picker.html'),
              wait_until='domcontentloaded', timeout=60_000)
    page.wait_for_function(
        """() => document.getElementById('cvFloegel')
                 && document.getElementById('cvFloegel').width > 20""",
        timeout=30_000,
    )
    first = page.locator('#ageSel').input_value()
    other = '105' if first == '115' else '115'

    page.locator('#cvFloegel').click(position={'x': 55, 'y': 40})
    page.locator('#cvRef').click(position={'x': 70, 'y': 50})
    assert 'Pares: 1' in page.locator('#status').inner_text()

    page.locator('#ageSel').select_option(other)
    page.wait_for_timeout(200)
    assert other in page.locator('#status').inner_text()
    assert 'Pares: 0' in page.locator('#status').inner_text()

    page.locator('#ageSel').select_option(first)
    page.wait_for_timeout(200)
    assert 'Pares: 1' in page.locator('#status').inner_text()
