import cv2

from digibook.enhance import flatten_lighting
from digibook.highlights import detect_highlights
from synth import ENGLISH, LATIN_FONT, render_page


def test_highlight_colours_and_positions():
    page = render_page(ENGLISH, LATIN_FONT, highlight_lines=(4,))
    flat = flatten_lighting(page)
    found, _ = detect_highlights(flat)
    assert len(found) == 1
    assert found[0].color == "yellow"


def test_no_false_highlights_on_plain_page():
    page = render_page(ENGLISH, LATIN_FONT)
    found, _ = detect_highlights(flatten_lighting(page))
    assert found == []


def test_green_and_pink():
    for rgb, name in (((120, 230, 120), "green"), ((255, 140, 190), "pink")):
        page = render_page(ENGLISH, LATIN_FONT, highlight_lines=(6,), highlight_color=rgb)
        found, _ = detect_highlights(flatten_lighting(page))
        assert [h.color for h in found] == [name]
