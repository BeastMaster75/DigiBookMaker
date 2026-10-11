"""Lighting cleanup: remove shadows and uneven light, whiten the paper."""

from __future__ import annotations

import cv2
import numpy as np

MODES = ("color", "gray", "bw", "original")


def estimate_background(channel: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray:
    """Paper brightness at every pixel, with the ink (and any pictures) removed."""
    h, w = channel.shape
    size = (max(1, w // 4), max(1, h // 4))
    small = cv2.resize(channel, size, interpolation=cv2.INTER_AREA)
    if exclude is not None and exclude.any():
        # Pictures are not paper: fill them in from the paper around them.
        hole = cv2.resize(exclude, size, interpolation=cv2.INTER_NEAREST)
        hole = cv2.dilate(hole, np.ones((5, 5), np.uint8))
        tiny = (max(1, size[0] // 4), max(1, size[1] // 4))
        filled = cv2.inpaint(cv2.resize(small, tiny, interpolation=cv2.INTER_AREA),
                             cv2.resize(hole, tiny, interpolation=cv2.INTER_NEAREST), 3,
                             cv2.INPAINT_TELEA)
        filled = cv2.resize(filled, size, interpolation=cv2.INTER_LINEAR)
        small = np.where(hole > 0, filled, small)
    k = max(5, min(small.shape) // 80) | 1
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg, min(255, max(21, (min(small.shape) // 15) | 1)))
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def flatten_lighting(img: np.ndarray, figures: np.ndarray | None = None) -> np.ndarray:
    """Even out light and whiten paper. `figures` masks pictures, which only get the
    same lighting/white-balance correction as the paper, never contrast or whitening."""
    # Correct brightness only (same gain on every channel) so highlighter colours
    # survive. The brightest channel is high on both paper and highlighter ink.
    bg = estimate_background(img.max(axis=2), figures).astype(np.float32)
    gain = 255.0 / np.maximum(bg, 1.0)
    flat = np.clip(img.astype(np.float32) * gain[..., None], 0, 255)
    text_area = np.ones(img.shape[:2], bool) if figures is None else figures == 0
    # White balance: make plain paper neutral white (removes yellow lamp light / old paper tint).
    hsv = cv2.cvtColor(flat.astype(np.uint8), cv2.COLOR_BGR2HSV)
    paper = (hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 180) & text_area
    if paper.sum() > 1000:
        ref = np.median(flat[paper], axis=0)
        flat = np.clip(flat * (250.0 / np.maximum(ref, 1.0)), 0, 255)
    flat = flat.astype(np.uint8)
    # Stretch so the darkest ink is truly dark.
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    lo = float(np.percentile(gray[text_area], 0.5)) if text_area.any() else 0.0
    lo = min(lo, 160.0)
    # Paper sits at ~250 after white balance; clip its grain/noise to pure white.
    lut = np.clip((np.arange(256) - lo) * 255.0 / max(1.0, 236.0 - lo), 0, 255).astype(np.uint8)
    out = cv2.LUT(flat, lut)
    if figures is not None and figures.any():
        # Keep picture tones, but let blank paper inside a picture's box go white too.
        paperish = (hsv[:, :, 1] < 25) & (flat.min(axis=2) > 238)
        keep = (figures > 0) & ~paperish
        out[keep] = flat[keep]
    return out


def render(flat: np.ndarray, original: np.ndarray, mode: str,
           figures: np.ndarray | None = None) -> np.ndarray:
    """Produce the page image that is shown in the PDF."""
    if mode == "original":
        return original
    if mode == "color":
        return flat
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    if mode == "gray":
        return gray
    if mode == "bw":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if figures is not None and figures.any():
            bw[figures > 0] = gray[figures > 0]  # keep pictures in grayscale, not posterised
        return bw
    raise ValueError(f"Unknown mode {mode!r}; choose from {', '.join(MODES)}")
