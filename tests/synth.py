"""Generate fake phone photos of book pages for tests and demos."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = ["/usr/share/fonts", "/System/Library/Fonts", "/Library/Fonts"]


def find_font(*names: str) -> str | None:
    for d in FONT_DIRS:
        for n in names:
            hits = list(Path(d).rglob(n)) if Path(d).exists() else []
            if hits:
                return str(hits[0])
    return None


LATIN_FONT = find_font("DejaVuSerif.ttf", "DejaVuSans.ttf")
CJK_FONT = find_font("NotoSerifCJK-Regular.ttc", "NotoSansCJK-Regular.ttc", "NotoSerifCJK-Bold.ttc")
ARABIC_FONT = find_font("DejaVuSans.ttf", "NotoNaskhArabic-Regular.ttf")

ENGLISH = [
    "Chapter One", "",
    "The library was quiet in the early morning,",
    "and the old books smelled of dust and paper.",
    "Every page held a small piece of history,",
    "waiting patiently for someone to read it.",
    "She opened the heavy volume on the table",
    "and began to copy the important passages.",
    "Knowledge is only useful when it is shared.",
    "The light from the window moved slowly",
    "across the desk as the hours went by.",
]
JAPANESE = [
    "第一章", "",
    "図書館は朝早くとても静かだった。",
    "古い本は紙とほこりのにおいがした。",
    "どのページにも小さな歴史があった。",
    "彼女は重い本を机の上で開いた。",
    "大切な文章を丁寧に書き写した。",
    "知識は共有されて初めて役に立つ。",
]
ARABIC = [
    "الفصل الأول", "",
    "كانت المكتبة هادئة في الصباح الباكر",
    "وكانت الكتب القديمة تفوح برائحة الورق",
    "كل صفحة تحمل جزءا صغيرا من التاريخ",
    "المعرفة مفيدة فقط عندما نشاركها",
    "فتحت الكتاب الثقيل على الطاولة",
]


def fake_photo(w: int, h: int, seed: int = 0) -> Image.Image:
    """A colourful 'photograph': sky gradient, a yellow sun, dark hills, texture."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h)[:, None, None]
    sky = (np.array([90, 150, 230]) * (1 - y) + np.array([200, 220, 245]) * y)
    arr = np.broadcast_to(sky, (h, w, 3)).copy()
    yy, xx = np.mgrid[0:h, 0:w]
    sun = (xx - w * 0.75) ** 2 + (yy - h * 0.3) ** 2 < (min(w, h) * 0.12) ** 2
    arr[sun] = (255, 220, 40)
    hills = yy > h * (0.6 + 0.12 * np.sin(xx / w * 6 + seed))
    arr[hills] = (40, 90, 35)
    arr += rng.normal(0, 10, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def draw_diagram(draw: ImageDraw.ImageDraw, x: int, y: int, font) -> None:
    """Simple line-art diagram with labels."""
    for i, label in enumerate(("Input", "Process", "Output")):
        bx = x + i * 330
        draw.rectangle([bx, y, bx + 240, y + 120], outline=(20, 20, 20), width=4)
        draw.text((bx + 30, y + 35), label, font=font, fill=(20, 20, 20))
        if i < 2:
            draw.line([bx + 240, y + 60, bx + 330, y + 60], fill=(20, 20, 20), width=4)
            draw.polygon([(bx + 330, y + 60), (bx + 312, y + 50), (bx + 312, y + 70)], fill=(20, 20, 20))


def render_page(lines: list[str], font_path: str, rtl: bool = False, vertical: bool = False,
                highlight_lines: tuple[int, ...] = (), size=(1240, 1754),
                highlight_color=(255, 235, 60), page_no: int | None = None,
                photo_box: tuple[int, int, int, int] | None = None,
                diagram_at: tuple[int, int] | None = None) -> np.ndarray:
    """Render a clean page (BGR) with optional highlighter strokes, photo and diagram."""
    w, h = size
    img = Image.new("RGB", size, (250, 248, 240))
    if photo_box:
        px, py, pw, ph = photo_box
        img.paste(fake_photo(pw, ph), (px, py))
    draw = ImageDraw.Draw(img, "RGBA")
    if diagram_at:
        draw_diagram(draw, *diagram_at, ImageFont.truetype(font_path, 40))
    font = ImageFont.truetype(font_path, 44)
    margin = 110
    if vertical:
        x = w - margin - 50
        for i, line in enumerate(lines):
            if not line:
                x -= 40
                continue
            y = margin
            if i in highlight_lines:
                draw.rectangle([x - 6, y - 6, x + 52, y + len(line) * 56], fill=(*highlight_color, 150))
            for ch in line:
                draw.text((x, y), ch, font=font, fill=(25, 25, 25))
                y += 56
            x -= 80
    else:
        y = margin
        for i, line in enumerate(lines):
            if not line:
                y += 40
                continue
            kw = {"direction": "rtl", "language": "ar"} if rtl else {}
            box = draw.textbbox((0, 0), line, font=font, **kw)
            tw = box[2] - box[0]
            x = w - margin - tw if rtl else margin
            if i in highlight_lines:
                draw.rectangle([x - 8, y + box[1] - 8, x + tw + 8, y + box[3] + 10],
                               fill=(*highlight_color, 150))
            draw.text((x, y), line, font=font, fill=(25, 25, 25), **kw)
            y += 80
    if page_no is not None:
        f2 = ImageFont.truetype(LATIN_FONT or font_path, 30)
        draw.text((w // 2 - 10, h - 80), str(page_no), font=f2, fill=(60, 60, 60))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def make_spread(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Two pages side by side with a shadowed gutter, like an open book."""
    spread = np.hstack([left, right]).astype(np.float32)
    w = spread.shape[1]
    x = np.abs(np.arange(w) - w / 2) / (w * 0.04)
    shade = 1 - 0.35 * np.exp(-x ** 2)
    return np.clip(spread * shade[None, :, None], 0, 255).astype(np.uint8)


def photograph(page: np.ndarray, seed: int = 0, out_size=(2400, 1800),
               return_corners: bool = False):
    """Place the page on a table at an angle with uneven light, like a phone photo."""
    rng = np.random.default_rng(seed)
    ow, oh = out_size[1], out_size[0]
    ph, pw = page.shape[:2]
    if pw > ph and ow < oh:  # landscape subject: hold the phone sideways
        ow, oh = oh, ow
    # Wooden-ish background.
    bg = np.zeros((oh, ow, 3), np.uint8)
    bg[:] = (40, 70, 110)
    noise = rng.normal(0, 12, (oh, ow, 1)).astype(np.int16)
    stripes = (np.sin(np.linspace(0, 60, ow))[None, :, None] * 10).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise + stripes, 0, 255).astype(np.uint8)

    scale = min(ow * 0.8 / pw, oh * 0.8 / ph)
    cw, ch = pw * scale, ph * scale
    cx, cy = ow / 2, oh / 2
    jitter = lambda: rng.uniform(-0.06, 0.06) * min(ow, oh)
    dst = np.float32([
        [cx - cw / 2 + jitter(), cy - ch / 2 + jitter()],
        [cx + cw / 2 + jitter(), cy - ch / 2 + jitter()],
        [cx + cw / 2 + jitter(), cy + ch / 2 + jitter()],
        [cx - cw / 2 + jitter(), cy + ch / 2 + jitter()],
    ])
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, m, (ow, oh))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), m, (ow, oh))
    out = np.where(mask[..., None] > 0, warped, bg)
    # Uneven light: a soft gradient shadow.
    gx = np.linspace(rng.uniform(0.65, 0.8), 1.05, ow)[None, :]
    gy = np.linspace(1.0, rng.uniform(0.75, 0.9), oh)[:, None]
    light = (gx * gy)[..., None]
    out = np.clip(out.astype(np.float32) * light, 0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    return (out, dst) if return_corners else out


PHOTO_BOX = (110, 620, 1020, 640)


def make_demo_set(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pages = [
        render_page(ENGLISH, LATIN_FONT, highlight_lines=(4, 8), page_no=1),
        render_page(JAPANESE, CJK_FONT, highlight_lines=(4,), page_no=2,
                    highlight_color=(120, 230, 120)),
        render_page(JAPANESE, CJK_FONT, vertical=True, highlight_lines=(7,), page_no=3),
        render_page(ARABIC, ARABIC_FONT, rtl=True, highlight_lines=(4,), page_no=4,
                    highlight_color=(255, 140, 190)),
        render_page(ENGLISH[:6], LATIN_FONT, highlight_lines=(3,), page_no=5,
                    photo_box=PHOTO_BOX, diagram_at=(130, 1380)),
    ]
    paths = []
    for i, p in enumerate(pages):
        path = folder / f"IMG_{i + 1:04d}.jpg"
        cv2.imwrite(str(path), photograph(p, seed=i), [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(path)
    return paths


if __name__ == "__main__":
    import sys

    for p in make_demo_set(sys.argv[1] if len(sys.argv) > 1 else "demo_photos"):
        print(p)
