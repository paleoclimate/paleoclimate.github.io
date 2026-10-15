"""UI/UX of the PCVS viewer shell (index.html)."""

from __future__ import annotations

import json
import re

import pytest

from tests.helpers import (
    COMPARISON_AGES,
    STORAGE_KEY,
    catalog_ages,
    current_viewer_age,
    auth_init_script,
    goto_viewer,
    load_maps_catalog,
    map_frame,
    representative_ages,
    select_age_via_combo,
    wait_viewer_ready,
)


@pytest.fixture
def maps():
    return load_maps_catalog()


def test_viewer_chrome_and_first_load(page, base_url, maps):
    goto_viewer(page, base_url)
    assert 'Paleoclimate' in page.title()
    assert page.locator('.brand-mark').inner_text().strip() == 'PCVS'
    assert page.locator('#ageBtn').is_visible()
    assert page.locator('#ageRange').count() == 0
    assert page.locator('#prevBtn').count() == 0
    assert page.locator('#nextBtn').count() == 0
    assert page.locator('#pdfBtn').is_visible()
    assert page.locator('#pdfBtn').is_enabled()
    assert page.locator('#rasterOnly').count() == 0
    assert page.locator('#mapFrame').is_visible()
    assert not page.locator('#loader').evaluate("el => el.classList.contains('on')")

    youngest = int(maps[0]['age'])
    assert current_viewer_age(page) == youngest
    assert page.locator('#ctxAge').inner_text().strip() == f'{youngest} Ma'
    assert 'KNN' in page.locator('#ctxSub').inner_text()
    assert 'IDW' in page.locator('#ctxSub').inner_text()
    assert page.locator('#ageBtn').get_attribute('aria-expanded') == 'false'
    assert page.locator('#mapFrame').get_attribute('title')


def test_combo_lists_every_age_and_selects(page, base_url, maps):
    goto_viewer(page, base_url)
    page.locator('#ageBtn').click()
    assert page.locator('#ageCombo').evaluate("el => el.classList.contains('open')")
    assert page.locator('#ageBtn').get_attribute('aria-expanded') == 'true'

    options = page.locator('.combo-opt')
    assert options.count() == len(maps)
    labels = [options.nth(i).inner_text().strip() for i in range(options.count())]
    assert labels == [entry['label'] for entry in maps]

    target = int(maps[-1]['age'])
    select_age_via_combo(page, target)
    assert not page.locator('#ageCombo').evaluate("el => el.classList.contains('open')")
    assert current_viewer_age(page) == target
    assert page.evaluate('location.hash') == f'#age={target}'
    assert page.locator('#ctxAge').inner_text().strip() == f'{target} Ma'
    frame = map_frame(page)
    assert f'map_{target}_ma' in frame.url


def test_combo_closes_on_escape_and_outside_click(page, base_url):
    goto_viewer(page, base_url)
    page.locator('#ageBtn').click()
    assert page.locator('#ageCombo').evaluate("el => el.classList.contains('open')")
    page.keyboard.press('Escape')
    assert not page.locator('#ageCombo').evaluate("el => el.classList.contains('open')")

    page.locator('#ageBtn').click()
    page.locator('.brand-mark').click()
    assert not page.locator('#ageCombo').evaluate("el => el.classList.contains('open')")


def test_combo_arrow_keys_change_age(page, base_url, maps):
    goto_viewer(page, base_url, age=int(maps[0]['age']))
    page.locator('#ageBtn').focus()
    page.keyboard.press('ArrowDown')
    page.keyboard.press('ArrowDown')
    wait_viewer_ready(page)
    assert current_viewer_age(page) == int(maps[1]['age'])
    page.keyboard.press('ArrowUp')
    wait_viewer_ready(page)
    assert current_viewer_age(page) == int(maps[0]['age'])


def test_age_changes_only_through_the_menu(page, base_url, maps):
    goto_viewer(page, base_url)
    assert page.locator('#ageRange').count() == 0
    assert page.locator('#prevBtn').count() == 0
    assert page.locator('#nextBtn').count() == 0
    assert page.get_by_text('Ancient', exact=True).count() == 0
    assert page.get_by_text('Recent', exact=True).count() == 0

    select_age_via_combo(page, int(maps[1]['age']))
    assert current_viewer_age(page) == int(maps[1]['age'])


def test_arrow_keys_do_not_change_age(page, base_url, maps):
    goto_viewer(page, base_url, age=int(maps[1]['age']))
    page.evaluate('document.activeElement && document.activeElement.blur()')
    page.keyboard.press('ArrowRight')
    page.keyboard.press('ArrowLeft')
    page.wait_for_timeout(200)
    assert current_viewer_age(page) == int(maps[1]['age'])
    assert page.evaluate('location.hash') == f"#age={maps[1]['age']}"


def test_export_shortcut_ignored_while_focus_is_on_an_input(page, base_url, maps):
    goto_viewer(page, base_url, age=int(maps[1]['age']))
    page.evaluate(
        """() => {
          const input = document.createElement('input');
          input.id = 'guardProbe';
          document.body.appendChild(input);
        }"""
    )
    page.locator('#guardProbe').focus()
    page.keyboard.press('p')
    page.wait_for_timeout(300)
    assert not page.locator('#toast').evaluate("el => el.classList.contains('on')")
    assert current_viewer_age(page) == int(maps[1]['age'])


def test_hash_selects_age_and_invalid_hash_falls_back(page, browser, base_url, maps):
    target = int(maps[len(maps) // 2]['age'])
    goto_viewer(page, base_url, age=target)
    assert current_viewer_age(page) == target

    # Same tab: a bad hash falls back to the age stored from the last visit.
    page.goto(f'{base_url}/index.html#age=99999', wait_until='domcontentloaded')
    wait_viewer_ready(page)
    assert current_viewer_age(page) == target

    # Fresh profile, no stored age: a bad hash opens the youngest map.
    isolated = browser.new_context(viewport={'width': 1440, 'height': 900})
    isolated.add_init_script(auth_init_script())
    blank = isolated.new_page()
    try:
        blank.goto(f'{base_url}/index.html#age=99999', wait_until='domcontentloaded')
        wait_viewer_ready(blank)
        assert current_viewer_age(blank) == int(maps[0]['age'])
    finally:
        isolated.close()


def test_hashchange_updates_map_without_reload(page, base_url, maps):
    goto_viewer(page, base_url, age=int(maps[0]['age']))
    target = int(maps[-1]['age'])
    page.evaluate(f'location.hash = "age={target}"')
    wait_viewer_ready(page)
    assert current_viewer_age(page) == target
    assert f'map_{target}_ma' in map_frame(page).url


def test_local_storage_restores_last_age(page, context, base_url, maps):
    target = int(maps[-2]['age']) if len(maps) > 1 else int(maps[0]['age'])
    goto_viewer(page, base_url, age=target)
    stored = page.evaluate(f'localStorage.getItem("{STORAGE_KEY}")')
    assert stored == str(target)

    page2 = context.new_page()
    try:
        page2.goto(f'{base_url}/index.html', wait_until='domcontentloaded')
        wait_viewer_ready(page2)
        assert current_viewer_age(page2) == target
    finally:
        page2.close()


def test_comparison_button_only_for_supported_ages(page, base_url, maps):
    ages = catalog_ages(maps)
    sample = sorted(set(ages[:1] + ages[-1:] + [age for age in COMPARISON_AGES if age in ages]))
    for age in sample:
        goto_viewer(page, base_url, age=age)
        button = page.locator('#comparisonBtn')
        box = button.bounding_box()
        assert box and box['width'] > 20, 'Comparison slot must keep its width on every age'
        if age in COMPARISON_AGES:
            assert not button.evaluate("el => el.classList.contains('is-absent')")
            assert button.get_attribute('aria-hidden') == 'false'
            assert button.get_attribute('tabindex') == '0'
            with page.expect_popup() as popup_info:
                button.click()
            popup = popup_info.value
            popup.wait_for_load_state('domcontentloaded')
            assert f'comparison_report_{age}.html' in popup.url
            assert popup.locator('h1').count() >= 1
            popup.close()
        else:
            assert button.evaluate("el => el.classList.contains('is-absent')")
            assert button.get_attribute('aria-hidden') == 'true'
            assert button.get_attribute('tabindex') == '-1'


def test_loader_appears_while_switching_maps(page, base_url, maps):
    goto_viewer(page, base_url, age=int(maps[0]['age']))
    page.locator('#ageBtn').click()
    page.locator('#ageMenu .combo-opt').nth(1).click()
    page.wait_for_function(
        "() => document.getElementById('loader').classList.contains('on') "
        "|| !document.getElementById('pdfBtn').disabled "
        "|| document.getElementById('ageBtnLabel').textContent.includes('"
        + maps[1]['label'] + "')"
    )
    wait_viewer_ready(page, timeout=120_000)
    assert current_viewer_age(page) == int(maps[1]['age'])
    assert page.locator('#pdfBtn').is_enabled()


def test_pdf_button_shows_progress_toast(page, base_url):
    goto_viewer(page, base_url)
    page.locator('#pdfBtn').click()
    toast = page.locator('#toast')
    toast.wait_for(state='visible')
    text = toast.inner_text().lower()
    assert 'raster' in text


def test_p_shortcut_triggers_export_but_modifiers_do_not(page, base_url):
    goto_viewer(page, base_url)
    page.evaluate('document.activeElement && document.activeElement.blur()')
    page.keyboard.press('p')
    page.locator('#toast').wait_for(state='visible')
    assert page.locator('#toast').inner_text().strip()

    page.goto(f'{base_url}/index.html', wait_until='domcontentloaded')
    wait_viewer_ready(page)
    page.evaluate('document.activeElement && document.activeElement.blur()')
    page.keyboard.press('Control+p')
    page.wait_for_timeout(300)
    assert not page.locator('#toast').evaluate("el => el.classList.contains('on')")


@pytest.mark.parametrize('age', representative_ages())
def test_representative_ages_load_in_iframe(page, base_url, age):
    goto_viewer(page, base_url, age=age)
    frame = map_frame(page)
    assert f'map_{age}_ma' in frame.url
    assert frame.locator('.leaflet-container').is_visible()
    assert frame.evaluate('() => !!window.PCVS')
    state = frame.evaluate(
        """() => {
          const box = document.querySelector('.leaflet-container').getBoundingClientRect();
          return {width: box.width, height: box.height};
        }"""
    )
    assert state['width'] > 200
    assert state['height'] > 200


def test_every_catalog_age_can_be_opened(page, base_url, maps):
    goto_viewer(page, base_url)
    for entry in maps:
        age = int(entry['age'])
        page.evaluate(f'location.hash = "age={age}"')
        wait_viewer_ready(page)
        assert current_viewer_age(page) == age
        frame = map_frame(page)
        assert entry['path'] in frame.url.replace('\\', '/')
        assert frame.locator('.leaflet-container').is_visible()


def test_iframe_is_same_origin_and_exposes_pcvs(page, base_url):
    goto_viewer(page, base_url)
    api = page.evaluate(
        """() => {
          const child = document.getElementById('mapFrame').contentWindow;
          return {
            hasPCVS: !!(child && child.PCVS),
            keys: child && child.PCVS ? Object.keys(child.PCVS).sort() : [],
          };
        }"""
    )
    assert api['hasPCVS']
    for name in ('exportPdf', 'exportSize', 'beginExport', 'endExport', 'basename'):
        assert name in api['keys']


def test_viewer_asks_for_the_password_before_loading_a_map(browser, base_url):
    context = browser.new_context()
    page = context.new_page()
    try:
        page.goto(f'{base_url}/index.html', wait_until='domcontentloaded')
        password = page.locator('#staticrypt-password')
        password.wait_for()
        assert page.locator('#mapFrame').count() == 0
        box = password.bounding_box()
        assert box and box['width'] < 260
        password.fill('not-the-password')
        page.locator('.staticrypt-decrypt-button').click()
        page.wait_for_function(
            "() => (document.getElementById('staticrypt-error').textContent || '')"
            ".includes('Wrong password')",
            timeout=60_000,
        )
        page.locator('.staticrypt-toggle-password-visibility').click()
        assert password.evaluate('el => el.type') == 'text'
        password.fill('Paleoclima-2026-entrega')
        page.locator('.staticrypt-decrypt-button').click()
        page.wait_for_function(
            "() => (document.getElementById('mapFrame') && "
            "document.getElementById('mapFrame').getAttribute('src') || '')"
            ".includes('map_')",
            timeout=120_000,
        )
    finally:
        page.close()
        context.close()


def _fill_required(page, latitude, longitude, time, climate):
    page.locator('[name="Latitude"]').fill(latitude)
    page.locator('[name="Longitude"]').fill(longitude)
    page.locator('[name="TIME"]').select_option(time)
    page.locator('[name="Climate_Cl"]').select_option(climate)
    page.locator('[name="Country"]').fill('Brazil')
    page.locator('[name="Basin_Sub_"]').fill('Test Basin')
    page.locator('[name="Formation"]').fill('Test Formation')
    page.locator('[name="PlateID"]').fill('201')
    page.locator('[name="REF(Authors, Year)"]').fill('Author, 2020')


def test_add_point_form_uses_the_table_without_weight(page, base_url):
    page.goto(f'{base_url}/add-point.html', wait_until='domcontentloaded')
    page.evaluate("localStorage.removeItem('pcvs:added-points')")
    page.goto(f'{base_url}/add-point.html', wait_until='domcontentloaded')
    assert page.locator('h1').inner_text() == 'Add points'
    assert page.get_by_role('heading', name='Required').is_visible()
    assert page.get_by_text('All fields in this section are required.').is_visible()
    assert page.get_by_role('heading', name='Optional Fields').is_visible()
    assert page.locator('.lead').inner_text() == (
        'Add as many points as you wish, then export the standardized GeoJSON. '
        'No data is updated on the current maps.'
    )
    labels = page.locator('form label').all_inner_texts()
    assert labels
    assert all('Weight' not in label for label in labels)
    for hidden in ('FILE1', 'RECONFILE1', 'ID', 'ANCHOR', 'Show_on_Ma'):
        assert page.locator(f'[name="{hidden}"]').count() == 0
    assert page.locator('input[type="number"]').count() == 0
    assert page.locator('[name="Latitude"]').evaluate('el => el.type') == 'text'
    assert '.' in page.locator('[name="Latitude"]').get_attribute('placeholder')
    assert ',' not in page.locator('[name="Latitude"]').get_attribute('placeholder')
    assert '.' in page.locator('[name="Longitude"]').get_attribute('placeholder')
    assert page.locator('[name="REF(Authors, Year)"]').count() == 1
    assert page.locator('[name="Paleoenvironment"]').count() == 1
    assert page.locator('[name="Dating evidence"]').count() == 1
    assert page.locator('[name="Lithology, structures, paleowind"]').count() == 1
    assert page.locator('#essential [name="Group"]').count() == 0
    assert page.locator('#optional [name="Group"]').count() == 1
    assert page.locator('#optional [name="Member"]').count() == 1
    assert page.locator('#optional [name="Well name/ outcrop"]').count() == 1
    for name in ('Group', 'Member', 'Well name/ outcrop'):
        assert page.locator(f'[name="{name}"]').evaluate('el => el.required') is False
    for name in ('Latitude', 'Country', 'Basin_Sub_', 'Formation', 'PlateID', 'REF(Authors, Year)'):
        assert page.locator(f'[name="{name}"]').evaluate('el => el.required') is True
    ages = page.locator('[name="TIME"] option').all_inner_texts()
    assert '65 Ma' in ages and '145 Ma' in ages
    climate = page.locator('[name="Climate_Cl"]')
    assert climate.locator('option:checked').get_attribute('disabled') is not None
    page.locator('[name="Latitude"]').fill('10,5')
    assert page.locator('[name="Latitude"]').evaluate('el => el.checkValidity()') is False
    lat = page.locator('[name="Latitude"]')
    lon = page.locator('[name="Longitude"]')
    lat.fill('-10.5')
    lon.fill('-40.25')
    assert lat.bounding_box()['width'] < 200
    assert page.locator('#pointCount').inner_text() == '0 points added'
    assert page.locator('#downloadBtn').is_disabled()
    page.locator('#addBtn').click()
    assert page.locator('#pointCount').inner_text() == '0 points added'

    _fill_required(page, '-10.5', '-40.25', '100', 'H')
    page.locator('#addBtn').click()
    assert page.locator('#pointCount').inner_text() == '1 point added'
    assert page.locator('[name="Latitude"]').input_value() == ''
    assert page.locator('[name="TIME"]').input_value() == ''
    assert page.locator('[name="Climate_Cl"]').input_value() == ''
    queued = json.loads(page.evaluate("localStorage.getItem('pcvs:added-points')"))
    assert queued[0]['properties']['TIME'] == 100
    assert queued[0]['properties']['Show_on_Ma'] == 100

    page.goto(f'{base_url}/add-point.html', wait_until='domcontentloaded')
    assert page.locator('#pointCount').inner_text() == '1 point added'

    _fill_required(page, '5.5', '10.0', '110', 'D')
    page.locator('#addBtn').click()
    assert page.locator('#pointCount').inner_text() == '2 points added'
    page.locator('#pointList button.remove').first.click()
    assert page.locator('#pointCount').inner_text() == '1 point added'

    _fill_required(page, '-10.5', '-40.25', '100', 'H')
    page.locator('#addBtn').click()
    assert page.locator('#pointCount').inner_text() == '2 points added'

    with page.expect_download() as download_info:
        page.locator('#downloadBtn').click()
    download = download_info.value
    assert download.suggested_filename.startswith('points-')
    assert download.suggested_filename.endswith('.geojson')
    text = open(download.path(), encoding='utf-8').read()
    payload = json.loads(text)
    assert payload['type'] == 'FeatureCollection'
    assert len(payload['features']) == 2
    by_time = {feature['properties']['TIME']: feature for feature in payload['features']}
    assert by_time[100]['geometry']['coordinates'] == [-40.25, -10.5]
    assert by_time[100]['properties']['Climate_Cl'] == 'H'
    assert by_time[100]['properties']['Country'] == 'Brazil'
    assert by_time[100]['properties']['PlateID'] == 201
    assert by_time[110]['geometry']['coordinates'] == [10, 5.5]
    assert by_time[110]['properties']['Climate_Cl'] == 'D'
    for feature in payload['features']:
        props = feature['properties']
        assert 'Weight' not in props
        assert props['ANCHOR'] == 0
        assert props['ID'] is None
        assert props['Show_on_Ma'] == props['TIME']
        assert props['FILE1'] == download.suggested_filename
        assert props['Group'] is None
        assert props['Member'] is None
        assert props['Well name/ outcrop'] is None
        assert props['RECONFILE1'] == (
            'Zahirovic_etal_2022_OptimisedMantleRef_and_NNRMantleRef.rot'
        )
    assert page.locator('#pointCount').inner_text() == '0 points added'
    assert page.evaluate("localStorage.getItem('pcvs:added-points')") is None
