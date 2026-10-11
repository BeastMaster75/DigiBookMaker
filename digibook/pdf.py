"""Assemble the searchable PDF: page image + invisible OCR text + highlight notes."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

import pikepdf
from pikepdf import Array, Dictionary, Name, String

from .highlights import Highlight


@dataclass
class PdfPage:
    jpeg: bytes
    width: int  # pixels
    height: int
    gray: bool
    dpi: int
    text_pdf: bytes = b""
    highlights: list[Highlight] = field(default_factory=list)


def _blank_page(pdf: pikepdf.Pdf, w_pt: float, h_pt: float) -> pikepdf.Page:
    page = pdf.add_blank_page(page_size=(w_pt, h_pt))
    return page


def build_pdf(pages: list[PdfPage], title: str = "", lang: str = "") -> bytes:
    out = pikepdf.new()
    sources: list[pikepdf.Pdf] = []
    for i, p in enumerate(pages):
        w_pt, h_pt = p.width / p.dpi * 72, p.height / p.dpi * 72
        if p.text_pdf:
            src = pikepdf.open(BytesIO(p.text_pdf))
            sources.append(src)
            out.pages.append(src.pages[0])
            page = out.pages[-1]
        else:
            page = _blank_page(out, w_pt, h_pt)
        x0, y0, x1, y1 = (float(v) for v in page.mediabox)
        pw, ph = x1 - x0, y1 - y0

        image = pikepdf.Stream(out, p.jpeg)
        image.Type = Name.XObject
        image.Subtype = Name.Image
        image.Width = p.width
        image.Height = p.height
        image.ColorSpace = Name.DeviceGray if p.gray else Name.DeviceRGB
        image.BitsPerComponent = 8
        image.Filter = Name.DCTDecode

        if Name.Resources not in page.obj:
            page.obj.Resources = Dictionary()
        res = page.obj.Resources
        if Name.XObject not in res:
            res.XObject = Dictionary()
        res.XObject[Name("/DBPage")] = image
        draw = f"q {pw:.4f} 0 0 {ph:.4f} {x0:.4f} {y0:.4f} cm /DBPage Do Q\n".encode()
        page.contents_add(pikepdf.Stream(out, draw), prepend=True)

        sx, sy = pw / p.width, ph / p.height
        annots = []
        for hl in p.highlights:
            left, right = x0 + hl.x * sx, x0 + (hl.x + hl.w) * sx
            top, bottom = y0 + ph - hl.y * sy, y0 + ph - (hl.y + hl.h) * sy
            annots.append(out.make_indirect(Dictionary(
                Type=Name.Annot,
                Subtype=Name.Highlight,
                Rect=Array([left, bottom, right, top]),
                QuadPoints=Array([left, top, right, top, left, bottom, right, bottom]),
                C=Array([c / 255 for c in hl.rgb]),
                CA=0.35,
                F=4,
                T=String("DigiBookMaker"),
                Subj=String(f"{hl.color} highlight"),
                Contents=String(hl.text),
                NM=String(f"p{i + 1}-hl{len(annots) + 1}"),
            )))
        if annots:
            page.obj.Annots = Array(annots)

    if title:
        out.docinfo[Name.Title] = String(title)
    out.docinfo[Name.Producer] = String("DigiBookMaker")
    if lang:
        out.Root.Lang = String(lang)
    buf = BytesIO()
    out.save(buf, linearize=True)
    for s in sources:
        s.close()
    return buf.getvalue()
