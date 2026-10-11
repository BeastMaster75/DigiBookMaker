import numpy as np

from digibook.enhance import flatten_lighting
from digibook.figures import detect_figures, figure_mask
from digibook.highlights import detect_highlights
from synth import ENGLISH, LATIN_FONT, PHOTO_BOX, render_page


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter)


def test_photo_found_text_and_diagram_ignored():
    page = render_page(ENGLISH[:6], LATIN_FONT, highlight_lines=(3,), photo_box=PHOTO_BOX,
                       diagram_at=(130, 1380))
    figs = detect_figures(page)
    assert len(figs) == 1
    assert _iou(figs[0], PHOTO_BOX) > 0.85


def test_plain_text_and_highlights_are_not_figures():
    page = render_page(ENGLISH, LATIN_FONT, highlight_lines=(2, 3, 4, 5))
    assert detect_figures(page) == []


def test_photo_not_washed_out_and_not_highlighted():
    page = render_page(ENGLISH[:6], LATIN_FONT, photo_box=PHOTO_BOX)
    # Uneven light, like a phone photo.
    shade = np.linspace(0.75, 1.0, page.shape[1])[None, :, None]
    lit = np.clip(page * shade, 0, 255).astype(np.uint8)
    mask = figure_mask(lit.shape, detect_figures(lit))
    flat = flatten_lighting(lit, mask)
    x, y, w, h = PHOTO_BOX
    region = (slice(y + 30, y + h - 30), slice(x + 30, x + w - 30))
    diff = np.abs(flat[region].astype(int) - page[region].astype(int)).mean()
    assert diff < 25, f"photo tones changed too much ({diff:.1f})"
    found, _ = detect_highlights(flat, mask)
    assert found == []  # the yellow sun is not a highlight

    # Without picture awareness the same photo gets blown out; make sure the test can tell.
    naive = flatten_lighting(lit)
    assert np.abs(naive[region].astype(int) - page[region].astype(int)).mean() > diff
