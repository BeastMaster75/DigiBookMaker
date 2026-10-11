"""Find photos and illustrations on a page so cleanup does not wash them out."""

from __future__ import annotations

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h


def detect_figures(img: np.ndarray, min_area: float = 0.01) -> list[Box]:
    """Return boxes of picture regions (photos, shaded illustrations) on a page.

    Text is sparse ink on paper; pictures are areas where most pixels are
    clearly darker than the surrounding paper, or coloured. Plain line art has
    little ink and is left to the normal text cleanup, which suits it.
    """
    h, w = img.shape[:2]
    s = min(1.0, 800 / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat, v = hsv[:, :, 1], hsv[:, :, 2]

    # Local paper brightness: brightest value nearby, which ignores ink and shadows' gradient.
    k = max(15, max(sh, sw) // 6) | 1
    paper = cv2.blur(cv2.dilate(v, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), (k, k))
    dark = v.astype(np.int16) < paper.astype(np.int16) - 40
    nonpaper = (dark | (sat > 40)).astype(np.float32)

    win = max(9, sh // 25) | 1
    dense = (cv2.blur(nonpaper, (win, win)) > 0.5).astype(np.uint8) * 255
    dense = cv2.morphologyEx(dense, cv2.MORPH_CLOSE, np.ones((win, win), np.uint8))
    dense = cv2.morphologyEx(dense, cv2.MORPH_OPEN, np.ones((win // 2 | 1, win // 2 | 1), np.uint8))

    boxes: list[Box] = []
    contours, _ = cv2.findContours(dense, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area * sh * sw or bw < 0.08 * sw or bh < 0.05 * sh:
            continue
        hl_like = ((sat[y:y + bh, x:x + bw] > 40) & (v[y:y + bh, x:x + bw] > 150)).mean()
        dark_frac = dark[y:y + bh, x:x + bw].mean()
        if hl_like > 0.45 and dark_frac < 0.3:
            continue  # a highlighted passage, not a picture
        pad = max(2, win // 3)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(sw, x + bw + pad), min(sh, y + bh + pad)
        boxes.append((int(x0 / s), int(y0 / s), int((x1 - x0) / s), int((y1 - y0) / s)))
    return merge_boxes(boxes)


def merge_boxes(boxes: list[Box]) -> list[Box]:
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ax, ay, aw, ah = boxes[i]
                bx, by, bw, bh = boxes[j]
                if ax <= bx + bw and bx <= ax + aw and ay <= by + bh and by <= ay + ah:
                    x0, y0 = min(ax, bx), min(ay, by)
                    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                    boxes[i] = (x0, y0, x1 - x0, y1 - y0)
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def figure_mask(shape: tuple[int, ...], boxes: list[Box]) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    for x, y, w, h in boxes:
        mask[y:y + h, x:x + w] = 255
    return mask
