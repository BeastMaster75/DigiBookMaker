"""The whole job: photos in, a uniform searchable book out."""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import geometry, ocr
from .enhance import MODES, flatten_lighting, render
from .figures import detect_figures, figure_mask
from .highlights import Highlight, detect_highlights, remove_highlighter
from .library import Library
from .pdf import PdfPage, build_pdf
from .sources import collect_files, iter_pages
from .textnorm import CJK_RE, join_words

Progress = Callable[[str, int, int], None]


@dataclass
class Options:
    lang: str = "auto"          # auto | eng | jpn+eng | jpn_vert+jpn | ara+eng | any Tesseract string
    mode: str = "color"         # color | gray | bw | original
    page_size: str = "auto"     # auto | a4 | a5 | b6 | letter | 148x210 (mm)
    split: str = "auto"         # auto | on | off : split two-page spreads
    dpi: int = 300
    jpeg_quality: int = 85
    jobs: int = field(default_factory=lambda: max(1, min(4, os.cpu_count() or 1)))
    title: str | None = None
    ocr: bool = True


@dataclass
class BookResult:
    slug: str
    title: str
    folder: Path
    pdf: Path
    text: Path
    highlights: Path
    pages: int
    highlight_count: int
    figure_count: int
    seconds: float


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", title.strip().lower(), flags=re.UNICODE).strip("-_")
    return slug[:80] or "book"


# ---------------------------------------------------------------- stage 1: find pages

def prepare_source(label: str, img: np.ndarray, opts: Options) -> dict:
    """Find and flatten the page(s) in one photo/scan. Runs in a worker process."""
    corners = geometry.detect_page(img)
    page = geometry.warp_page(img, corners) if corners is not None else img
    if corners is None:
        page = geometry.rotate(page, geometry.estimate_skew(page))

    script = None
    rotated = 0
    if opts.ocr:
        osd = ocr.detect_osd(page, opts.dpi)
        script = osd.script
        # Vertical Japanese looks "rotated" to OSD, so only trust 90/270 for other scripts.
        trust = osd.orientation_conf > 3 and (osd.rotate == 180 or script not in ocr.CJK_SCRIPTS)
        if osd.rotate and trust:
            page = geometry.rotate90(page, osd.rotate)
            rotated = osd.rotate

    split = opts.split == "on" or (opts.split == "auto" and geometry.is_spread(page))
    pages = geometry.split_spread(page) if split else [page]
    if split:  # the halves of a photographed spread are often tilted differently
        pages = [geometry.rotate(p, geometry.estimate_skew(p, 3.0)) for p in pages]

    # Small preview of what was detected, to make bad photos easy to spot.
    scale = min(1.0, 900 / max(img.shape[:2]))
    preview = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if corners is not None:
        cv2.polylines(preview, [np.int32(corners * scale)], True, (0, 0, 255), 3)
    _, prev_jpg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 75])

    return {
        "label": label,
        "pages": pages,
        "script": script,
        "found_page": corners is not None,
        "rotated": rotated,
        "split": split,
        "preview": prev_jpg.tobytes(),
    }


# ---------------------------------------------------------------- stage 2: clean + OCR

def _attach_words(highlights: list[Highlight], words: list[ocr.Word]) -> None:
    for hl in highlights:
        pad = min(hl.w, hl.h) * 0.25
        picked = []
        for idx, w in enumerate(words):
            cx, cy = w.x + w.w / 2, w.y + w.h / 2
            if hl.x - pad <= cx <= hl.x + hl.w + pad and hl.y - pad <= cy <= hl.y + hl.h + pad:
                picked.append(idx)
        hl.words = picked
        hl.text = join_words([words[i].text for i in picked])


def process_page(page: np.ndarray, size: tuple[int, int], script: str | None,
                 opts: Options) -> dict:
    """Resize to the book's page size, clean, find highlights, OCR. Worker process."""
    tw, th = size
    interp = cv2.INTER_AREA if page.shape[0] > th else cv2.INTER_CUBIC
    page = cv2.resize(page, (tw, th), interpolation=interp)
    figures = detect_figures(page)
    fig_mask = figure_mask(page.shape, figures)
    flat = flatten_lighting(page, fig_mask)
    shown = render(flat, page, opts.mode, fig_mask)
    highlights, hl_mask = detect_highlights(flat, fig_mask)

    result = ocr.OcrResult(lang="")
    if opts.ocr:
        ocr_img = remove_highlighter(flat, hl_mask)
        result = ocr.ocr_page(ocr_img, opts.lang, script, opts.dpi)
        # Photos make OCR "see" junk words; keep only confident words inside pictures
        # (real labels in diagrams survive).
        result.words = [w for w in result.words
                        if w.conf >= 60 or not fig_mask[min(th - 1, w.y + w.h // 2),
                                                         min(tw - 1, w.x + w.w // 2)]]
        result.text = ocr.words_to_text(result.words)
        _attach_words(highlights, result.words)

    figure_jpegs = []
    for x, y, w, h in figures:
        _, fj = cv2.imencode(".jpg", flat[y:y + h, x:x + w], [cv2.IMWRITE_JPEG_QUALITY, 92])
        figure_jpegs.append(fj.tobytes())

    ok, jpg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, opts.jpeg_quality])
    if not ok:
        raise RuntimeError("Could not encode page image")
    return {
        "jpeg": jpg.tobytes(),
        "gray": shown.ndim == 2,
        "text": result.text,
        "lang": result.lang,
        "text_pdf": result.pdf,
        "highlights": highlights,
        "figures": figures,
        "figure_jpegs": figure_jpegs,
    }


# ---------------------------------------------------------------- driver

def _pool(jobs: int):
    if jobs <= 1:
        return None
    # "spawn" is safe from the web server's threads (fork is not).
    return ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"))


def _map(pool, fn, *iterables):
    if pool is None:
        return map(fn, *iterables)
    return pool.map(fn, *iterables)


def make_book(inputs: list[str | Path], library_dir: str | Path, opts: Options | None = None,
              progress: Progress | None = None) -> BookResult:
    opts = opts or Options()
    if opts.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if opts.split not in ("auto", "on", "off"):
        raise ValueError("split must be auto, on or off")
    start = time.time()
    report = progress or (lambda stage, done, total: None)
    if opts.ocr:
        ocr.tesseract_bin()  # fail fast with install instructions

    files = collect_files(inputs)
    title = opts.title or (Path(inputs[0]).stem if len(inputs) == 1 else files[0].parent.name) or "book"
    slug = slugify(title)

    library = Library(library_dir)
    folder = library.book_dir(slug)
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "pages").mkdir(parents=True)
    (folder / "debug").mkdir()
    (folder / "figures").mkdir()

    pool = _pool(opts.jobs)
    try:
        report("Finding pages", 0, len(files))
        futures = []
        for label, img in iter_pages(files, opts.dpi):
            if pool is None:
                futures.append(prepare_source(label, img, opts))
            else:
                futures.append(pool.submit(prepare_source, label, img, opts))
        sources = []
        for i, f in enumerate(futures):
            sources.append(f if isinstance(f, dict) else f.result())
            report("Finding pages", i + 1, len(futures))

        pages, scripts, labels = [], [], []
        for i, src in enumerate(sources):
            (folder / "debug" / f"source-{i + 1:03d}.jpg").write_bytes(src["preview"])
            for j, p in enumerate(src["pages"]):
                pages.append(p)
                scripts.append(src["script"])
                labels.append(src["label"] + (f" ({'LR'[j]})" if src["split"] else ""))
        if not pages:
            raise ValueError("No pages found")

        size = geometry.target_size([p.shape[:2] for p in pages], opts.page_size, opts.dpi)
        report("Cleaning + reading text", 0, len(pages))
        results = []
        for i, r in enumerate(_map(pool, process_page, pages, [size] * len(pages), scripts,
                                   [opts] * len(pages))):
            results.append(r)
            report("Cleaning + reading text", i + 1, len(pages))
    finally:
        if pool is not None:
            pool.shutdown()

    report("Building PDF", 0, 1)
    pdf_pages = []
    fig_count = 0
    for i, r in enumerate(results):
        (folder / "pages" / f"page-{i + 1:03d}.jpg").write_bytes(r["jpeg"])
        r["figure_files"] = []
        for j, data in enumerate(r["figure_jpegs"]):
            name = f"figures/page-{i + 1:03d}-figure-{j + 1}.jpg"
            (folder / name).write_bytes(data)
            r["figure_files"].append(name)
            fig_count += 1
        pdf_pages.append(PdfPage(jpeg=r["jpeg"], width=size[0], height=size[1], gray=r["gray"],
                                 dpi=opts.dpi, text_pdf=r["text_pdf"], highlights=r["highlights"]))
    all_text = "\n".join(r["text"] for r in results)
    main_lang = "ja" if CJK_RE.search(all_text) else ("ar" if re.search(r"[؀-ۿ]", all_text) else "en")
    pdf_path = folder / f"{slug}.pdf"
    pdf_path.write_bytes(build_pdf(pdf_pages, title=title, lang=main_lang if opts.ocr else ""))

    text_path = folder / "text.md"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        for i, r in enumerate(results):
            f.write(f"\n## Page {i + 1}\n\n{r['text']}\n")
            for j, name in enumerate(r["figure_files"]):
                f.write(f"\n![Page {i + 1}, picture {j + 1}]({name})\n")

    hl_path = folder / "highlights.md"
    hl_count = 0
    with open(hl_path, "w", encoding="utf-8") as f:
        f.write(f"# Highlights — {title}\n")
        for i, r in enumerate(results):
            if not r["highlights"]:
                continue
            f.write(f"\n## Page {i + 1}\n\n")
            for hl in r["highlights"]:
                hl_count += 1
                f.write(f"- ({hl.color}) {hl.text or '[no text recognised]'}\n")
        if hl_count == 0:
            f.write("\nNo highlights found.\n")

    meta = {
        "title": title,
        "slug": slug,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "options": {k: v for k, v in vars(opts).items()},
        "page_size_px": list(size),
        "sources": [{"label": s["label"], "found_page": s["found_page"], "rotated": s["rotated"],
                     "split": s["split"], "script": s["script"]} for s in sources],
        "pages": [{"page": i + 1, "source": labels[i], "lang": r["lang"],
                   "chars": len(r["text"]),
                   "figures": [{"file": n, "box": list(b)}
                               for n, b in zip(r["figure_files"], r["figures"])],
                   "highlights": [hl.to_dict() for hl in r["highlights"]]}
                  for i, r in enumerate(results)],
    }
    (folder / "book.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    library.add_book(slug, title, [r["text"] for r in results])
    library.close()
    report("Done", 1, 1)
    return BookResult(slug, title, folder, pdf_path, text_path, hl_path, len(results), hl_count,
                      fig_count, time.time() - start)
