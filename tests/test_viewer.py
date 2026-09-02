"""UI/UX of the PCVS viewer shell (index.html)."""

from __future__ import annotations

import re

import pytest

from tests.helpers import (
    COMPARISON_AGES,
    STORAGE_KEY,
    catalog_ages,
    current_viewer_age,
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
    assert page.locator('#ageRange').is_visible()
    assert page.locator('#prevBtn').is_visible()
    assert page.locator('#nextBtn').is_visible()
    assert page.locator('#pdfBtn').is_visible()
    assert page.locator('#pdfBtn').is_enabled()
    assert page.locator('#rasterOnly').is_visible()
    assert page.locator('#mapFrame').is_visible()
    assert not page.locator('#loader').evaluate("el => el.classList.contains('on')")

    youngest = int(maps[0]['age'])
    assert current_viewer_age(page) == youngest
    assert page.locator('#ctxAge').inner_text().strip() == f'{youngest} Ma'
    assert 'KNN' in page.locator('#ctxSub').inner_text()
    assert page.locator('#prevBtn').is_disabled()
    assert page.locator('#nextBtn').is_enabled()
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
    assert page.locator('#nextBtn').is_disabled()
    assert page.locator('#prevBtn').is_enabled()
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


def test_timeline_and_stepper_navigate_ages(page, base_url, maps):
    goto_viewer(page, base_url)
    last_index = len(maps) - 1
    slider = page.locator('#ageRange')
    assert slider.get_attribute('min') == '0'
    assert slider.get_attribute('max') == str(last_index)

    slider.evaluate(
        """(el, value) => {
          el.value = String(value);
          el.dispatchEvent(new Event('input', {bubbles: true}));
        }""",
        last_index,
    )
    assert page.locator('#ageBtnLabel').inner_text().strip() == maps[-1]['label']
    slider.evaluate(
        """(el) => el.dispatchEvent(new Event('change', {bubbles: true}))"""
    )
    wait_viewer_ready(page)
    assert current_viewer_age(page) == int(maps[-1]['age'])

    page.locator('#prevBtn').click()
    wait_viewer_ready(page)
    assert current_viewer_age(page) == int(maps[-2]['age'])

    page.locator('#nextBtn').click()
    wait_viewer_ready(page)
    assert current_viewer_age(page) == int(maps[-1]['age'])


def test_keyboard_arrows_and_hash_stay_in_sync(page, base_url, maps):
    goto_viewer(page, base_url, age=int(maps[1]['age']))
    page.evaluate('document.activeElement && document.activeElement.blur()')
    page.keyboard.press('ArrowRight')
    wait_viewer_ready(page)
    assert current_viewer_age(page) == int(maps[2]['age'])
    assert page.evaluate('location.hash') == f"#age={maps[2]['age']}"

    page.keyboard.press('ArrowLeft')
    wait_viewer_ready(page)
    assert current_viewer_age(page) == int(maps[1]['age'])


def test_keyboard_ignored_while_focus_is_on_an_input(page, base_url, maps):
    goto_viewer(page, base_url, age=int(maps[1]['age']))
    page.locator('#rasterOnly').focus()
    page.keyboard.press('ArrowRight')
    page.wait_for_timeout(200)
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
    page.locator('#nextBtn').click()
    page.wait_for_function(
        "() => document.getElementById('loader').classList.contains('on') "
        "|| !document.getElementById('pdfBtn').disabled "
        "|| document.getElementById('ageBtnLabel').textContent.includes('"
        + maps[1]['label'] + "')"
    )
    wait_viewer_ready(page)
    assert current_viewer_age(page) == int(maps[1]['age'])
    assert page.locator('#pdfBtn').is_enabled()


def test_pdf_button_shows_progress_toast(page, base_url):
    goto_viewer(page, base_url)
    assert not page.locator('#rasterOnly').is_checked()
    page.locator('#pdfBtn').click()
    toast = page.locator('#toast')
    toast.wait_for(state='visible')
    text = toast.inner_text()
    assert 'Rendering' in text or 'PDF' in text or 'exported' in text.lower()


def test_raster_only_toggle_changes_export_scope_copy(page, base_url):
    goto_viewer(page, base_url)
    page.locator('#rasterOnly').check()
    page.locator('#pdfBtn').click()
    page.locator('#toast').wait_for(state='visible')
    assert 'raster' in page.locator('#toast').inner_text().lower()


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
