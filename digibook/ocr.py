"""Tesseract OCR: script detection, language choice, words + invisible-text PDF."""

from __future__ import annotations

import csv
import functools
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .textnorm import join_words

CJK_SCRIPTS = {"Japanese", "Han", "HanS", "HanT", "Hiragana", "Katakana"}
LANG_ALIASES = {
    "japanese": "jpn+eng",
    "jpn-vert": "jpn_vert+jpn",
    "vertical": "jpn_vert+jpn",
    "arabic": "ara+eng",
    "english": "eng",
    "all": "eng+jpn+ara",
}


class TesseractMissing(RuntimeError):
    pass


@dataclass
class Word:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float
    line: tuple[int, int, int]


@dataclass
class OcrResult:
    lang: str
    words: list[Word] = field(default_factory=list)
    text: str = ""
    pdf: bytes = b""
    score: float = 0.0


@dataclass
class Osd:
    rotate: int = 0
    orientation_conf: float = 0.0
    script: str | None = None
    script_conf: float = 0.0


def tesseract_bin() -> str:
    exe = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not exe:
        raise TesseractMissing(
            "Tesseract is not installed. Install it with language packs, e.g. "
            "`sudo apt install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert "
            "tesseract-ocr-ara` or `brew install tesseract tesseract-lang`, or use the Docker image."
        )
    return exe


@functools.lru_cache(maxsize=1)
def available_langs() -> frozenset[str]:
    out = subprocess.run([tesseract_bin(), "--list-langs"], capture_output=True, text=True)
    langs = {ln.strip() for ln in out.stdout.splitlines()[1:] if ln.strip()}
    return frozenset(langs)


def _filter_lang(lang: str) -> str | None:
    have = available_langs()
    parts = [p for p in lang.split("+") if p in have]
    return "+".join(dict.fromkeys(parts)) or None


def _env() -> dict:
    # Pages are processed in parallel; keep each Tesseract single-threaded.
    return {**os.environ, "OMP_THREAD_LIMIT": "1"}


def _write_png(img: np.ndarray, folder: str) -> str:
    path = os.path.join(folder, "page.png")
    cv2.imwrite(path, img)
    return path


def detect_osd(img: np.ndarray, dpi: int = 300) -> Osd:
    """Orientation and script detection. Never raises; returns defaults on failure."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_png(img, tmp)
        try:
            out = subprocess.run(
                [tesseract_bin(), path, "stdout", "--psm", "0", "--dpi", str(dpi)],
                capture_output=True, text=True, env=_env(), timeout=120,
            ).stdout
        except (subprocess.SubprocessError, OSError):
            return Osd()
    osd = Osd()
    if m := re.search(r"Rotate:\s*(\d+)", out):
        osd.rotate = int(m.group(1))
    if m := re.search(r"Orientation confidence:\s*([\d.]+)", out):
        osd.orientation_conf = float(m.group(1))
    if m := re.search(r"Script:\s*(\S+)", out):
        osd.script = m.group(1)
    if m := re.search(r"Script confidence:\s*([\d.]+)", out):
        osd.script_conf = float(m.group(1))
    return osd


def candidate_langs(lang: str, script: str | None) -> list[str]:
    lang = LANG_ALIASES.get(lang.lower(), lang)
    if lang != "auto":
        cands = [lang]
    elif script in CJK_SCRIPTS:
        cands = ["jpn+eng", "jpn_vert+jpn"]
    elif script == "Arabic":
        cands = ["ara+eng"]
    elif script == "Latin":
        cands = ["eng"]
    else:
        cands = ["eng", "jpn+eng", "jpn_vert+jpn", "ara+eng"]
    filtered = [c for c in (_filter_lang(c) for c in cands) if c]
    if not filtered:
        raise TesseractMissing(f"None of the OCR languages {cands} are installed for Tesseract.")
    return list(dict.fromkeys(filtered))


def _parse_tsv(path: Path) -> list[Word]:
    words: list[Word] = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if row.get("level") != "5":
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            words.append(Word(
                text=text,
                x=int(row["left"]), y=int(row["top"]), w=int(row["width"]), h=int(row["height"]),
                conf=float(row["conf"]),
                line=(int(row["block_num"]), int(row["par_num"]), int(row["line_num"])),
            ))
    return words


def words_to_text(words: list[Word]) -> str:
    lines: list[str] = []
    current: list[str] = []
    prev = None
    for w in words:
        if prev is not None and w.line != prev:
            lines.append(join_words(current))
            if w.line[:2] != prev[:2]:
                lines.append("")  # paragraph break
            current = []
        current.append(w.text)
        prev = w.line
    if current:
        lines.append(join_words(current))
    return "\n".join(lines).strip()


def run_ocr(img: np.ndarray, lang: str, dpi: int = 300, psm: int = 3) -> OcrResult:
    """OCR one page; returns words, text and a text-only (invisible) PDF page."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_png(img, tmp)
        base = os.path.join(tmp, "out")
        cmd = [tesseract_bin(), path, base, "-l", lang, "--psm", str(psm), "--dpi", str(dpi),
               "-c", "textonly_pdf=1", "pdf", "tsv"]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=_env(), timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"Tesseract failed ({lang}): {proc.stderr.strip()[-500:]}")
        words = _parse_tsv(Path(base + ".tsv"))
        pdf = Path(base + ".pdf").read_bytes()
    score = sum(max(w.conf, 0) / 100 * len(w.text) for w in words)
    return OcrResult(lang=lang, words=words, text=words_to_text(words), pdf=pdf, score=score)


def ocr_page(img: np.ndarray, lang: str = "auto", script: str | None = None,
             dpi: int = 300) -> OcrResult:
    """OCR with every candidate language setting and keep the most confident result."""
    best: OcrResult | None = None
    for cand in candidate_langs(lang, script):
        # Automatic layout analysis often shreds tategaki; also try "one vertical block".
        psms = (3, 5) if cand.startswith("jpn_vert") else (3,)
        for psm in psms:
            res = run_ocr(img, cand, dpi=dpi, psm=psm)
            if best is None or res.score > best.score:
                best = res
    assert best is not None
    return best
