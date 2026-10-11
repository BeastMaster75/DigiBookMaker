"""Load page images from files, folders, and PDFs in natural (page) order."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageOps

try:  # iPhone photos are usually HEIC
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}
PDF_EXTS = {".pdf"}


def natural_key(path: Path) -> list:
    """Sort 'page2' before 'page10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def collect_files(inputs: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found = [f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS | PDF_EXTS]
            files.extend(sorted(found, key=natural_key))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"No such file or folder: {p}")
    if not files:
        raise ValueError("No images or PDFs found in the given inputs.")
    return files


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img)  # respect phone rotation metadata
    if img.mode != "RGB":
        img = img.convert("RGB")
    rgb = np.asarray(img)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def iter_pages(files: list[Path], pdf_dpi: int = 300) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (label, BGR image) for every page in the given files."""
    for f in files:
        ext = f.suffix.lower()
        if ext in PDF_EXTS:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(f))
            try:
                for i in range(len(pdf)):
                    pil = pdf[i].render(scale=pdf_dpi / 72).to_pil()
                    yield f"{f.name}#{i + 1}", pil_to_bgr(pil)
            finally:
                pdf.close()
        elif ext in IMAGE_EXTS:
            with Image.open(f) as im:
                yield f.name, pil_to_bgr(im)
        else:
            raise ValueError(f"Unsupported file type: {f.name}")
