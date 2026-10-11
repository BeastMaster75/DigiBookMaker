"""Text cleanup for OCR output and normalisation for search (CJK + Arabic aware)."""

from __future__ import annotations

import re
import unicodedata

# Hiragana, katakana, CJK ideographs, CJK punctuation, full-width forms.
_CJK = r"　-〿぀-ヿㇰ-ㇿ㐀-䶿一-鿿豈-﫿＀-￯"
_CJK_GAP = re.compile(rf"(?<=[{_CJK}])[ \t]+(?=[{_CJK}])")
_ARABIC_MARKS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_ARABIC_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
_SPACES = re.compile(r"[ \t]+")

CJK_RE = re.compile(rf"[{_CJK}]")


def clean_ocr_text(text: str) -> str:
    """Tidy OCR text: Tesseract puts spaces between Japanese characters."""
    text = unicodedata.normalize("NFC", text)
    text = _CJK_GAP.sub("", text)
    return "\n".join(_SPACES.sub(" ", line).strip() for line in text.splitlines())


def join_words(words: list[str]) -> str:
    return clean_ocr_text(" ".join(words))


def normalize_for_search(text: str) -> str:
    """Fold case, width, Arabic diacritics/letter variants; drop CJK spacing."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _ARABIC_MARKS.sub("", text).translate(_ARABIC_FOLD)
    text = re.sub(r"\s+", " ", text)
    return _CJK_GAP.sub("", text).strip()
