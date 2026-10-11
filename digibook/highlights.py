"""Find highlighter marks on a lighting-flattened page."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

# OpenCV hue runs 0..180.
HUE_NAMES = [
    (8, "pink"),
    (18, "orange"),
    (38, "yellow"),
    (85, "green"),
    (130, "blue"),
    (150, "purple"),
    (181, "pink"),
]


@dataclass
class Highlight:
    x: int
    y: int
    w: int
    h: int
    color: str
    rgb: tuple[int, int, int]
    text: str = ""
    words: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("words")
        return d


def hue_name(hue: float) -> str:
    for limit, name in HUE_NAMES:
        if hue < limit:
            return name
    return "pink"


def highlight_mask(flat: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    return ((s > 40) & (v > 150)).astype(np.uint8) * 255


def detect_highlights(flat: np.ndarray, exclude: np.ndarray | None = None
                      ) -> tuple[list[Highlight], np.ndarray]:
    """Return highlight boxes and the raw highlighter pixel mask.

    `exclude` masks pictures, whose colours are not highlighter.
    """
    h, w = flat.shape[:2]
    raw = highlight_mask(flat)
    if exclude is not None:
        raw[exclude > 0] = 0
    unit = max(1, h // 400)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((2 * unit + 1, 2 * unit + 1), np.uint8))
    # Bridge the gaps that dark letters punch into a highlighter stroke (horizontally only,
    # so separate lines stay separate).
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((3 * unit + 1, 12 * unit + 1), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    hsv = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV)
    found: list[Highlight] = []
    keep = np.zeros_like(raw)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bh < h * 0.008 or bw < w * 0.02:
            continue  # specks
        if bh > h * 0.12 and bw > w * 0.12:
            continue  # big in both directions: a colour photo or illustration, not a highlight
        box_raw = raw[y:y + bh, x:x + bw] > 0
        fill = box_raw.mean()
        if fill < 0.4:
            continue  # coloured printed text: thin strokes, low fill
        region = labels[y:y + bh, x:x + bw] == i
        pix = region & box_raw
        hue = float(np.median(hsv[y:y + bh, x:x + bw, 0][pix]))
        b, g, r = (int(np.median(flat[y:y + bh, x:x + bw, c][pix])) for c in range(3))
        found.append(Highlight(int(x), int(y), int(bw), int(bh), hue_name(hue), (r, g, b)))
        keep[y:y + bh, x:x + bw][region] = 255
    found.sort(key=lambda hl: (hl.y, hl.x))
    return found, keep


def remove_highlighter(flat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Grayscale page for OCR with highlighter ink turned white.

    Inside highlighted areas the brightest colour channel is used: highlighter
    is bright in at least one channel while black ink is dark in all of them.
    """
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    if not mask.any():
        return gray
    grown = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    brightest = flat.max(axis=2)
    out = gray.copy()
    out[grown > 0] = brightest[grown > 0]
    return out
