"""A full-text index over every book you have made (SQLite FTS5, trigram)."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .textnorm import normalize_for_search

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    slug TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    pages INTEGER NOT NULL,
    created REAL NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS pages USING fts5(
    norm, text UNINDEXED, slug UNINDEXED, page UNINDEXED, tokenize = 'trigram'
);
"""


@dataclass
class Hit:
    slug: str
    title: str
    page: int
    snippet: str


class Library:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "library.db", check_same_thread=False)
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def book_dir(self, slug: str) -> Path:
        return self.root / slug

    def add_book(self, slug: str, title: str, page_texts: list[str]) -> None:
        with self.db:
            self.db.execute("DELETE FROM pages WHERE slug = ?", (slug,))
            self.db.execute(
                "INSERT OR REPLACE INTO books (slug, title, pages, created) VALUES (?, ?, ?, ?)",
                (slug, title, len(page_texts), time.time()),
            )
            self.db.executemany(
                "INSERT INTO pages (norm, text, slug, page) VALUES (?, ?, ?, ?)",
                [(normalize_for_search(t), t, slug, i + 1) for i, t in enumerate(page_texts)],
            )

    def remove_book(self, slug: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM pages WHERE slug = ?", (slug,))
            self.db.execute("DELETE FROM books WHERE slug = ?", (slug,))

    def books(self) -> list[dict]:
        rows = self.db.execute("SELECT slug, title, pages, created FROM books ORDER BY created DESC")
        return [dict(zip(("slug", "title", "pages", "created"), r)) for r in rows]

    def search(self, query: str, limit: int = 50) -> list[Hit]:
        q = normalize_for_search(query)
        if not q:
            return []
        if len(q) >= 3:
            phrase = '"' + q.replace('"', '""') + '"'
            sql = "SELECT norm, text, slug, page FROM pages WHERE pages MATCH ? LIMIT ?"
            rows = self.db.execute(sql, (phrase, limit)).fetchall()
        else:  # trigram index needs 3+ characters; short CJK words fall back to a scan
            sql = "SELECT norm, text, slug, page FROM pages WHERE norm LIKE ? LIMIT ?"
            like = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            rows = self.db.execute(sql.replace("?", "? ESCAPE '\\'", 1), (like, limit)).fetchall()
        titles = {b["slug"]: b["title"] for b in self.books()}
        return [Hit(slug, titles.get(slug, slug), int(page), _snippet(norm, q))
                for norm, _text, slug, page in rows]


def _snippet(norm: str, q: str, width: int = 60) -> str:
    i = norm.find(q)
    if i < 0:
        return norm[: 2 * width]
    start, end = max(0, i - width), min(len(norm), i + len(q) + width)
    return ("…" if start else "") + norm[start:i] + "[" + norm[i:i + len(q)] + "]" + \
        norm[i + len(q):end] + ("…" if end < len(norm) else "")
