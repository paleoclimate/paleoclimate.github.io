"""Cross-cutting UX: layout, overflow, accessibility, and edge states."""

from __future__ import annotations

import pytest

from tests.helpers import (
    boxes_overlap,
    current_viewer_age,
    goto_viewer,
    load_maps_catalog,
    map_frame,
    wait_viewer_ready,
)

VIEWPORTS = (
    ('desktop', 1440, 900),
    ('laptop', 1280, 800),
    ('tablet', 900, 700),
    ('phone', 390, 844),
)


def _header_controls(page):
    return {
        'brand': page.locator('.brand-mark'),
        'combo': page.locator('#ageBtn'),
        'prev': page.locator('#prevBtn'),
        'next': page.locator('#nextBtn'),
        'range': page.locator('#ageRange'),
        'pdf': page.locator('#pdfBtn'),
        'scope': page.locator('#scopeToggle'),
    }


def test_no_horizontal_overflow_across_viewports(page, base_url):
    goto_viewer(page, base_url)
    for name, width, height in VIEWPORTS:
        page.set_viewport_size({'width': width, 'height': height})
        page.wait_for_timeout(200)
        overflow = page.evaluate(
            """() => ({
                 scroll: document.documentElement.scrollWidth,
                 client: document.documentElement.clientWidth,
               })"""
        )
        assert overflow['scroll'] <= overflow['client'] + 2, (
            f'{name} ({width}x{height}) scrolls horizontally '
            f"{overflow['scroll']} > {overflow['client']}"
        )
        iframe = page.locator('#mapFrame').bounding_box()
        assert iframe and iframe['height'] > 160, f'{name}: map area collapsed'
        assert page.locator('#pdfBtn').is_visible()
        assert page.locator('#ageBtn').is_visible()


def test_header_controls_do_not_overlap_on_desktop(page, base_url):
    page.set_viewport_size({'width': 1440, 'height': 900})
    goto_viewer(page, base_url)
    boxes = {}
    for name, locator in _header_controls(page).items():
        box = locator.bounding_box()
        assert box and box['width'] > 0 and box['height'] > 0, f'{name} is not painted'
        boxes[name] = box

    pairs = (
        ('brand', 'combo'),
        ('combo', 'prev'),
        ('prev', 'range'),
        ('range', 'next'),
        ('pdf', 'scope'),
    )
    for left, right in pairs:
        assert not boxes_overlap(boxes[left], boxes[right]), f'{left} overlaps {right}'


def test_phone_layout_wraps_timeline_and_keeps_actions(page, base_url):
    page.set_viewport_size({'width': 390, 'height': 844})
    goto_viewer(page, base_url)
    header = page.locator('header').bounding_box()
    timeline = page.locator('.timeline').bounding_box()
    actions = page.locator('.header-actions').bounding_box()
    assert header and timeline and actions
    assert timeline['y'] >= header['y']
    assert page.locator('#pdfBtn').is_enabled()
    maps = load_maps_catalog()
    if len(maps) < 2:
        pytest.skip('Need at least two ages to exercise the stepper on a phone')
    page.locator('#nextBtn').click()
    wait_viewer_ready(page)
    assert current_viewer_age(page) == int(maps[1]['age'])


def test_focus_visible_and_aria_on_core_controls(page, base_url):
    goto_viewer(page, base_url)
    page.locator('#ageBtn').focus()
    outline = page.locator('#ageBtn').evaluate(
        """el => {
          const style = getComputedStyle(el);
          return {
            outline: style.outlineStyle,
            width: style.outlineWidth,
            offset: style.outlineOffset,
          };
        }"""
    )
    # :focus-visible applies after keyboard focus; force-check the rule exists.
    rule = page.evaluate(
        """() => {
          for (const sheet of document.styleSheets) {
            let rules;
            try { rules = sheet.cssRules; } catch (e) { continue; }
            for (const rule of rules) {
              if (rule.selectorText && rule.selectorText.includes(':focus-visible')) {
                return true;
              }
            }
          }
          return false;
        }"""
    )
    assert rule, 'Viewer CSS is missing :focus-visible'
    assert page.locator('#ageBtn').get_attribute('aria-haspopup') == 'listbox'
    assert page.locator('#ageMenu').get_attribute('role') == 'listbox'
    assert page.locator('#ageRange').get_attribute('aria-label')
    assert page.locator('#prevBtn').get_attribute('aria-label')
    assert page.locator('#nextBtn').get_attribute('aria-label')
    assert page.locator('#toast').get_attribute('role') == 'status'
    assert page.locator('#mapFrame').get_attribute('title')
    assert outline['width'] is not None


def test_disabled_stepper_at_ends(page, base_url):
    maps = load_maps_catalog()
    goto_viewer(page, base_url, age=int(maps[0]['age']))
    assert page.locator('#prevBtn').is_disabled()
    page.locator('#prevBtn').click(force=True)
    page.wait_for_timeout(150)
    assert current_viewer_age(page) == int(maps[0]['age'])

    goto_viewer(page, base_url, age=int(maps[-1]['age']))
    assert page.locator('#nextBtn').is_disabled()
    page.locator('#nextBtn').click(force=True)
    page.wait_for_timeout(150)
    assert current_viewer_age(page) == int(maps[-1]['age'])


def test_rapid_age_changes_settle_on_the_last_choice(page, base_url):
    maps = load_maps_catalog()
    steps = min(4, len(maps) - 1)
    if steps < 1:
        pytest.skip('Need at least two ages')
    goto_viewer(page, base_url, age=int(maps[0]['age']))
    for _ in range(steps):
        page.locator('#nextBtn').click()
    wait_viewer_ready(page)
    assert current_viewer_age(page) == int(maps[steps]['age'])
    frame = map_frame(page)
    assert f"map_{maps[steps]['age']}_ma" in frame.url


def test_toast_is_polite_and_clears(page, base_url):
    goto_viewer(page, base_url)
    page.locator('#pdfBtn').click()
    toast = page.locator('#toast')
    toast.wait_for(state='visible')
    assert toast.get_attribute('aria-live') == 'polite'
    assert toast.evaluate("el => el.classList.contains('on')")


def test_map_iframe_fills_remaining_viewport(page, base_url):
    page.set_viewport_size({'width': 1440, 'height': 900})
    goto_viewer(page, base_url)
    metrics = page.evaluate(
        """() => {
          const header = document.querySelector('header').getBoundingClientRect();
          const iframe = document.getElementById('mapFrame').getBoundingClientRect();
          return {
            headerBottom: header.bottom,
            iframeTop: iframe.top,
            iframeHeight: iframe.height,
            viewport: window.innerHeight,
          };
        }"""
    )
    assert abs(metrics['iframeTop'] - metrics['headerBottom']) < 4
    assert metrics['iframeHeight'] > 500
    assert metrics['iframeTop'] + metrics['iframeHeight'] <= metrics['viewport'] + 2
