"""Small local web page: drop your page photos, get a book back."""

from __future__ import annotations

import queue
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .library import Library
from .pipeline import Options, make_book, slugify
from .sources import IMAGE_EXTS, PDF_EXTS

STATIC = Path(__file__).parent / "static"


class Jobs:
    """Runs one book at a time in a background thread."""

    def __init__(self, library_dir: Path):
        self.library_dir = library_dir
        self.state: dict[str, dict] = {}
        self.queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, upload_dir: Path, opts: Options) -> str:
        job_id = uuid.uuid4().hex[:12]
        self.state[job_id] = {"id": job_id, "status": "queued", "stage": "Waiting", "done": 0,
                              "total": 1, "title": opts.title, "error": None, "result": None}
        self.queue.put((job_id, upload_dir, opts))
        return job_id

    def _worker(self) -> None:
        while True:
            job_id, upload_dir, opts = self.queue.get()
            job = self.state[job_id]
            job["status"] = "running"

            def progress(stage: str, done: int, total: int) -> None:
                job.update(stage=stage, done=done, total=total)

            try:
                res = make_book([upload_dir], self.library_dir, opts, progress=progress)
                job["status"] = "done"
                job["result"] = {"slug": res.slug, "title": res.title, "pages": res.pages,
                                 "highlights": res.highlight_count,
                                 "figures": res.figure_count, "seconds": round(res.seconds),
                                 "pdf": res.pdf.name}
            except Exception as exc:  # report any failure to the page
                traceback.print_exc()
                job["status"] = "error"
                job["error"] = str(exc) or exc.__class__.__name__
            finally:
                shutil.rmtree(upload_dir, ignore_errors=True)


def create_app(library_dir: str | Path = "library") -> FastAPI:
    library_dir = Path(library_dir).resolve()
    library = Library(library_dir)
    jobs = Jobs(library_dir)
    uploads = library_dir / "_uploads"
    app = FastAPI(title="DigiBookMaker")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.post("/api/books")
    async def create_book(
        files: list[UploadFile] = File(...),
        title: str = Form(""),
        lang: str = Form("auto"),
        mode: str = Form("color"),
        page_size: str = Form("auto"),
        split: str = Form("auto"),
    ) -> dict:
        title = title.strip() or time.strftime("Book %Y-%m-%d %H%M%S")
        job_dir = uploads / uuid.uuid4().hex
        job_dir.mkdir(parents=True)
        saved = 0
        for f in files:
            name = Path(f.filename or "").name
            if Path(name).suffix.lower() not in IMAGE_EXTS | PDF_EXTS:
                continue
            with open(job_dir / name, "wb") as out:
                shutil.copyfileobj(f.file, out)
            saved += 1
        if not saved:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(400, "No images or PDFs in the upload.")
        opts = Options(title=title, lang=lang, mode=mode, page_size=page_size, split=split)
        return {"job": jobs.submit(job_dir, opts), "slug": slugify(title)}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        if job_id not in jobs.state:
            raise HTTPException(404, "Unknown job")
        return jobs.state[job_id]

    def _pdf_name(slug: str) -> str | None:
        pdfs = list(library.book_dir(slug).glob("*.pdf"))
        return pdfs[0].name if pdfs else None

    @app.get("/api/books")
    def list_books() -> list[dict]:
        return [{**b, "pdf": _pdf_name(b["slug"])} for b in library.books()]

    @app.delete("/api/books/{slug}")
    def delete_book(slug: str) -> dict:
        folder = _book_folder(slug)
        library.remove_book(slug)
        shutil.rmtree(folder, ignore_errors=True)
        return {"deleted": slug}

    @app.get("/api/search")
    def search(q: str) -> list[dict]:
        return [{**vars(h), "pdf": _pdf_name(h.slug)} for h in library.search(q)]

    def _book_folder(slug: str) -> Path:
        folder = (library_dir / slug).resolve()
        if folder.parent != library_dir or not folder.is_dir() or slug.startswith("_"):
            raise HTTPException(404, "No such book")
        return folder

    @app.get("/files/{slug}/{path:path}")
    def book_file(slug: str, path: str) -> FileResponse:
        folder = _book_folder(slug)
        target = (folder / path).resolve()
        if not target.is_file() or not target.is_relative_to(folder):
            raise HTTPException(404, "No such file")
        return FileResponse(target)

    return app
