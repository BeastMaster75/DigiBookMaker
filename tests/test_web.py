import time

import cv2
import pytest

from conftest import needs_tesseract
from synth import ENGLISH, LATIN_FONT, photograph, render_page

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from digibook.web import create_app  # noqa: E402


@needs_tesseract
def test_upload_and_search(tmp_path):
    client = TestClient(create_app(tmp_path / "lib"))
    assert "DigiBookMaker" in client.get("/").text
    ok, jpg = cv2.imencode(".jpg", photograph(render_page(ENGLISH, LATIN_FONT, highlight_lines=(4,))))
    r = client.post("/api/books", data={"title": "Web Book"},
                    files=[("files", ("IMG_1.jpg", jpg.tobytes(), "image/jpeg"))])
    assert r.status_code == 200, r.text
    job = r.json()["job"]
    for _ in range(120):
        s = client.get(f"/api/jobs/{job}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert s["status"] == "done", s
    assert s["result"]["highlights"] == 1
    books = client.get("/api/books").json()
    assert books[0]["slug"] == "web-book"
    pdf = client.get(f"/files/web-book/{books[0]['pdf']}")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    hits = client.get("/api/search", params={"q": "patiently"}).json()
    assert hits and hits[0]["page"] == 1
    assert client.get("/files/web-book/../library.db").status_code == 404
    assert client.get("/files/_uploads/x").status_code == 404
    assert client.delete("/api/books/web-book").status_code == 200
    assert client.get("/api/books").json() == []


def test_rejects_non_images(tmp_path):
    client = TestClient(create_app(tmp_path / "lib"))
    r = client.post("/api/books", files=[("files", ("notes.txt", b"hi", "text/plain"))])
    assert r.status_code == 400
