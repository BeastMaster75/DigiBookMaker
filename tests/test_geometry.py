import cv2
import numpy as np

from digibook import geometry
from synth import ENGLISH, LATIN_FONT, make_spread, photograph, render_page


def test_detect_page_finds_corners():
    page = render_page(ENGLISH, LATIN_FONT)
    for seed in range(4):
        photo, truth = photograph(page, seed=seed, return_corners=True)
        found = geometry.detect_page(photo)
        assert found is not None
        err = np.abs(found - geometry.order_corners(truth)).max()
        assert err < 25, f"seed {seed}: corner error {err:.1f}px"


def test_tight_scan_is_left_alone():
    page = render_page(ENGLISH, LATIN_FONT)
    assert geometry.detect_page(page) is None


def test_warp_gives_page_aspect():
    page = render_page(ENGLISH, LATIN_FONT)
    photo, truth = photograph(page, seed=1, return_corners=True)
    warped = geometry.warp_page(photo, geometry.detect_page(photo))
    h, w = warped.shape[:2]
    assert abs(w / h - page.shape[1] / page.shape[0]) < 0.08


def test_skew_estimate():
    page = render_page(ENGLISH, LATIN_FONT)
    tilted = geometry.rotate(page, 2.5)
    assert abs(geometry.estimate_skew(tilted) + 2.5) < 0.4


def test_split_spread_at_gutter():
    page = render_page(ENGLISH, LATIN_FONT)
    spread = make_spread(page, page)
    assert geometry.is_spread(spread)
    left, right = geometry.split_spread(spread)
    assert abs(left.shape[1] - page.shape[1]) < spread.shape[1] * 0.02
    assert not geometry.is_spread(left) and not geometry.is_spread(right)


def test_target_size():
    assert geometry.target_size([(1000, 700), (1010, 690), (990, 720)], "auto", 300) == (1680, 2400)
    assert geometry.target_size([(1, 1)], "a5", 300) == (1748, 2480)
    assert geometry.target_size([(1, 1)], "100x200", 254) == (1000, 2000)
