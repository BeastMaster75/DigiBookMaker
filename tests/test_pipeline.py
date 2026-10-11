import json

import cv2
import pikepdf
import pypdfium2 as pdfium

from conftest import needs_tesseract
from digibook import geometry
from digibook.library import Library
from digibook.pipeline import Options, make_book
from synth import ENGLISH, LATIN_FONT, make_demo_set, make_spread, photograph, render_page

pytestmark = needs_tesseract


def _pdf_text(path):
    pdf = pdfium.PdfDocument(str(path))
    try:
        return [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
    finally:
        pdf.close()


def test_demo_book_end_to_end(tmp_path):
    photos = make_demo_set(tmp_path / "photos")
    res = make_book([tmp_path / "photos"], tmp_path / "lib", Options(title="Demo"))
    assert res.pages == len(photos) == 5

    # Every page has the same size.
    with pikepdf.open(res.pdf) as pdf:
        boxes = {tuple(round(float(v)) for v in p.mediabox) for p in pdf.pages}
        assert len(boxes) == 1
        assert [len(p.obj.get("/Annots", [])) for p in pdf.pages] == [2, 1, 1, 1, 1]

    text = res.text.read_text(encoding="utf-8")
    assert "Knowledge is only useful when it is shared" in text
    assert "知識は共有されて初めて役に立つ" in text  # horizontal and vertical pages
    assert "كانت المكتبة هادئة" in text
    assert "Process" in text  # labels inside a line diagram
    assert "](figures/page-005-figure-1.jpg)" in text

    hl = res.highlights.read_text(encoding="utf-8")
    assert "(yellow) Every page held a small piece of history" in hl
    assert "(green) どのページにも小さな歴史があった" in hl
    assert "(pink) كل صفحة تحمل جزءا صغيرا من التاريخ" in hl
    assert res.highlight_count == 6  # the yellow sun in the photo is not a highlight

    assert res.figure_count == 1
    assert (res.folder / "figures" / "page-005-figure-1.jpg").is_file()

    pages = _pdf_text(res.pdf)
    assert "Knowledge" in pages[0]  # invisible text layer is there

    lib = Library(tmp_path / "lib")
    assert [h.page for h in lib.search("knowledge")] == [1]
    assert {h.page for h in lib.search("歴史")} == {2, 3}
    assert [h.page for h in lib.search("المكتبه")] == [4]  # ة/ه folded


def test_spread_is_split_and_upside_down_fixed(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    left = render_page(ENGLISH[:6], LATIN_FONT, page_no=10)
    right = render_page(ENGLISH[6:], LATIN_FONT, page_no=11)
    cv2.imwrite(str(src / "a_spread.jpg"), photograph(make_spread(left, right), seed=3))
    upside = cv2.rotate(photograph(render_page(ENGLISH, LATIN_FONT), seed=5), cv2.ROTATE_180)
    cv2.imwrite(str(src / "b_upside_down.jpg"), upside)
    res = make_book([src], tmp_path / "lib", Options(title="Spread"))
    assert res.pages == 3
    meta = json.loads((res.folder / "book.json").read_text())
    assert meta["sources"][0]["split"] is True
    assert meta["sources"][1]["rotated"] == 180
    text = res.text.read_text()
    assert "The library was quiet" in text
    assert "She opened the heavy volume" in text
    assert text.count("Chapter One") == 2


def test_flat_tilted_scan_and_pdf_input(tmp_path):
    scan = geometry.rotate(render_page(ENGLISH, LATIN_FONT), 2.0)
    img_path = tmp_path / "scan.png"
    cv2.imwrite(str(img_path), scan)
    first = make_book([img_path], tmp_path / "lib", Options(title="Scan", page_size="a5"))
    assert "The library was quiet in the early morning" in first.text.read_text()
    with pikepdf.open(first.pdf) as pdf:
        w, h = (float(v) for v in pdf.pages[0].mediabox[2:])
    assert abs(w / 72 * 25.4 - 148) < 1 and abs(h / 72 * 25.4 - 210) < 1

    # A scanned PDF goes back in and comes out searchable.
    again = make_book([first.pdf], tmp_path / "lib", Options(title="Again"))
    assert "The library was quiet" in again.text.read_text()


def test_black_and_white_keeps_photo_gray(tmp_path):
    from synth import PHOTO_BOX

    page = render_page(ENGLISH[:4], LATIN_FONT, photo_box=PHOTO_BOX)
    cv2.imwrite(str(tmp_path / "p.png"), page)
    res = make_book([tmp_path / "p.png"], tmp_path / "lib",
                    Options(title="BW", mode="bw", ocr=False))
    out = cv2.imread(str(res.folder / "pages" / "page-001.jpg"), cv2.IMREAD_GRAYSCALE)
    x, y, w, h = PHOTO_BOX
    sy, sx = out.shape[0] / page.shape[0], out.shape[1] / page.shape[1]
    inner = out[int((y + 40) * sy):int((y + h - 40) * sy), int((x + 40) * sx):int((x + w - 40) * sx)]
    mid_tones = ((inner > 40) & (inner < 215)).mean()
    assert mid_tones > 0.5, "photo was posterised to black and white"
