"""Page finding, perspective flattening, spread splitting, deskewing, sizing."""

from __future__ import annotations

import cv2
import numpy as np

DETECT_MAX_SIDE = 1000


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _quad_candidates(mask: np.ndarray, min_area: float):
    """Yield (quad, contour_area, fill) for large blobs in a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if area < min_area:
            break
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in (0.02, 0.03, 0.04, 0.06):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
                break
        if quad is None:
            quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        quad_area = cv2.contourArea(quad)
        if quad_area <= 0:
            continue
        yield quad, quad_area, min(area / quad_area, quad_area / area)


def _looks_like_paper(small: np.ndarray, quad: np.ndarray) -> bool:
    """A page's margins are bright and colourless, and not darker than what lies
    outside it. This rejects a photo or figure printed on a flat-scanned page."""
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = np.zeros(small.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.int32(quad), 255)
    band = max(3, int(min(small.shape[:2]) * 0.02))
    kernel = np.ones((2 * band + 1, 2 * band + 1), np.uint8)
    inside = (mask > 0) & (cv2.erode(mask, kernel) == 0)
    outside = (cv2.dilate(mask, kernel) > 0) & (mask == 0)
    if inside.sum() < 50:
        return False
    v_in = float(np.median(hsv[:, :, 2][inside]))
    s_in = float(np.median(hsv[:, :, 1][inside]))
    v_out = float(np.median(hsv[:, :, 2][outside])) if outside.sum() > 50 else 0.0
    return s_in < 70 and v_in >= v_out - 10


def detect_page(img: np.ndarray) -> np.ndarray | None:
    """Find the page outline in a photo.

    Returns 4 ordered corners in image coordinates, or None when the image
    already is a tight scan (or no convincing page could be found).
    """
    h, w = img.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    img_area = small.shape[0] * small.shape[1]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    masks = []
    # 1) Paper is usually brighter than whatever it lies on.
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)))
    # 2) Strong edges around the sheet.
    edges = cv2.Canny(gray, 40, 120)
    masks.append(cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2))
    # 3) Low saturation (white paper) against a coloured background.
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
    _, low_sat = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(cv2.morphologyEx(low_sat, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)))

    best, best_score = None, 0.0
    for mask in masks:
        for quad, quad_area, fill in _quad_candidates(mask, 0.2 * img_area):
            frac = quad_area / img_area
            if frac > 0.97 or fill < 0.8:
                continue
            score = frac * fill
            if score > best_score and _looks_like_paper(small, quad):
                best, best_score = quad, score
    if best is None:
        return None
    return order_corners(best / scale)


def warp_page(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    m = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, m, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def is_spread(img: np.ndarray, threshold: float = 1.15) -> bool:
    h, w = img.shape[:2]
    return w / h > threshold


def split_spread(img: np.ndarray) -> list[np.ndarray]:
    """Split a two-page spread at the gutter (the darkest column near the middle)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lo, hi = int(w * 0.4), int(w * 0.6)
    profile = gray[:, lo:hi].mean(axis=0)
    k = max(3, (hi - lo) // 20) | 1
    profile = np.convolve(profile, np.ones(k) / k, mode="same")
    cut = lo + int(np.argmin(profile[k:-k])) + k if len(profile) > 2 * k else w // 2
    if profile.max() - profile.min() < 8:  # no visible gutter shadow
        cut = w // 2
    return [img[:, :cut], img[:, cut:]]


def _skew_score(binary: np.ndarray, angle: float) -> float:
    h, w = binary.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST)
    # Text lines (horizontal) or columns (vertical Japanese) give peaky profiles.
    return float(max(np.var(rot.sum(axis=1)), np.var(rot.sum(axis=0))))


def estimate_skew(img: np.ndarray, max_angle: float = 5.0) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = min(1.0, 800 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    binary = cv2.adaptiveThreshold(small, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 15)
    # Ignore a thin border, where page edges and shadows live.
    b = max(2, int(min(binary.shape) * 0.04))
    binary = np.ascontiguousarray(binary[b:-b, b:-b])
    if binary.sum() < 50:
        return 0.0
    coarse = np.arange(-max_angle, max_angle + 0.01, 0.5)
    best = max(coarse, key=lambda a: _skew_score(binary, a))
    fine = np.arange(best - 0.5, best + 0.51, 0.1)
    return float(max(fine, key=lambda a: _skew_score(binary, a)))


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.05:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def rotate90(img: np.ndarray, degrees_cw: int) -> np.ndarray:
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    code = codes.get(degrees_cw % 360)
    return img if code is None else cv2.rotate(img, code)


PAGE_SIZES_MM = {
    "a4": (210, 297),
    "a5": (148, 210),
    "a6": (105, 148),
    "b5": (176, 250),
    "b6": (125, 176),
    "letter": (215.9, 279.4),
    "bunko": (105, 148),
    "shinsho": (103, 182),
}


def target_size(page_shapes: list[tuple[int, int]], page_size: str, dpi: int) -> tuple[int, int]:
    """Pick one pixel size (w, h) that every page of the book gets resized to."""
    page_size = (page_size or "auto").lower()
    if page_size != "auto":
        if page_size in PAGE_SIZES_MM:
            w_mm, h_mm = PAGE_SIZES_MM[page_size]
        else:  # e.g. "148x210" in millimetres
            try:
                w_mm, h_mm = (float(v) for v in page_size.split("x"))
            except ValueError as exc:
                raise ValueError(f"Unknown page size {page_size!r}") from exc
        return int(round(w_mm / 25.4 * dpi)), int(round(h_mm / 25.4 * dpi))
    ratios = [w / h for h, w in page_shapes]
    ratio = float(np.median(ratios))
    # Treat the median page as ~8 inches tall: typical book height, ~2400 px at 300 dpi.
    height = int(round(8 * dpi))
    return int(round(height * ratio)), height
