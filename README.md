# DigiBookMaker

Turn phone photos or scans of book pages into **one clean, searchable PDF**.

- **Straightens every page**: finds the page in the photo, removes the perspective tilt, and resizes all pages to the *same size and shape*.
- **Cleans the paper**: removes shadows and uneven light and makes the paper white. Your highlighter colours are kept.
- **Keeps pictures intact**: photos and illustrations keep their real tones (they are never bleached or turned black and white), and each one is also saved as its own image file.
- **Reads the text (OCR)**: English, Japanese (horizontal *and* vertical/tategaki) and Arabic, detected automatically per page.
- **Searchable**: the PDF has an invisible text layer, so Ctrl+F, copy and paste work. There's also a search box that covers every book you've made.
- **Finds your highlights**: yellow, green, pink, orange, blue and purple. You get a list of everything you highlighted, and the highlights show up as notes in PDF readers.
- **Splits two-page photos** of an open book into separate pages, and turns upside-down or sideways photos the right way up.

A 10-page book takes about 10–15 seconds on a normal laptop.

---

## Quick start (easiest: Docker)

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/), then in this folder run:

```bash
docker compose up --build
```

Open **http://localhost:8000**, drop your page photos in, press **Make book**. Finished books are saved in the `library/` folder.

## Quick start (without Docker)

You need Python 3.10+ and Tesseract with the language packs.

**Ubuntu / Debian**
```bash
sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-jpn tesseract-ocr-jpn-vert tesseract-ocr-ara
```
**macOS**
```bash
brew install tesseract tesseract-lang
```
**Windows**: install Tesseract from <https://github.com/UB-Mannheim/tesseract/wiki> and tick Japanese, Japanese (vertical) and Arabic in the installer.

Then:
```bash
pip install .
digibook serve            # web page at http://localhost:8000
```

## Command line

```bash
digibook make ./my_photos/ -t "My Book"          # a folder of photos, in file-name order
digibook make IMG_0012.HEIC IMG_0013.HEIC         # or individual files
digibook make old_scan.pdf -t "Old Scan"          # or a scanned PDF
digibook search "knowledge"                       # search every book
digibook search 歴史
digibook search المكتبة
digibook list
```

Useful options for `make`:

| Option | Values | Default |
|---|---|---|
| `-l, --lang` | `auto`, `eng`, `japanese`, `vertical` (tategaki), `arabic`, `all`, or a Tesseract string like `eng+jpn` | `auto` |
| `-m, --mode` | `color` (cleaned), `gray`, `bw` (black & white text, pictures stay grey), `original` | `color` |
| `-s, --page-size` | `auto` (matches your pages), `a4`, `a5`, `a6`, `b5`, `b6`, `letter`, `bunko`, `shinsho`, or `WxH` in mm like `128x188` | `auto` |
| `--split` | `auto`, `on`, `off` (two-page photos) | `auto` |
| `--no-ocr` | only clean up pages | off |

## What you get

Each book goes into `library/<book-name>/`:

```
my-book.pdf        the book: clean, same-size pages, searchable, highlights as notes
text.md            all the text, page by page, with links to the pictures
highlights.md      everything you highlighted, with its colour
figures/           every photo/illustration found, as separate images
pages/             each cleaned page as an image
debug/             your original photos with the detected page outlined in red
book.json          details: languages used per page, highlight and picture positions
```

## Tips for good photos

- Lay the page on a **darker, plain surface** so its edges stand out, and get all four corners in the frame.
- Use even light and avoid glare on glossy paper. Turn off the flash if it causes a hotspot.
- Fill most of the frame with the page; a slight angle is fine because it gets corrected.
- Name or shoot the photos in page order. Pages are sorted by file name (`IMG_2`, then `IMG_10`).
- If a page comes out wrong, check its picture in `debug/`: the red outline shows what was detected as the page.

## Limits (honest list)

- **Strongly curved pages** near the spine of a thick book are flattened as a plane, so text near the gutter can stay a bit curved. Pressing the book flat (or using a sheet of glass) helps a lot.
- **OCR quality** depends on the photo. Clean print is very accurate; handwriting, decorative fonts and very small furigana are not.
- **Japanese in PDF viewers**: searching inside the PDF works for words, but some viewers insert spaces between Japanese characters in copied text. `text.md` and the built-in search are cleaned of those spaces.
- Pictures are recognised as areas of photo or shading. Plain line drawings are treated like text, which keeps them sharp.

## Development

```bash
pip install -e '.[dev]'
pytest
python tests/synth.py demo_photos     # generate fake phone photos to try it on
```

How it works, in `digibook/`:

| File | Job |
|---|---|
| `geometry.py` | find page outline, perspective warp, split spreads, deskew, uniform page size |
| `figures.py` | find photos/illustrations so they are protected from cleanup |
| `enhance.py` | shadow and lighting removal, white balance, colour/gray/bw output |
| `highlights.py` | highlighter detection, and removing highlighter ink before OCR |
| `ocr.py` | Tesseract: orientation and script detection, language choice, words and text layer |
| `pdf.py` | build the PDF: page image, invisible text, highlight notes |
| `library.py` | SQLite full-text search (trigram, handles Japanese and Arabic) |
| `pipeline.py` | runs it all, in parallel |
| `web.py`, `static/` | the upload page |

## Licence

MIT (see `LICENSE`). Built on open-source software:
[Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (Apache-2.0),
[OpenCV](https://opencv.org) (Apache-2.0),
[pikepdf](https://github.com/pikepdf/pikepdf) (MPL-2.0),
[pypdfium2](https://github.com/pypdfium2-team/pypdfium2) (Apache-2.0 / BSD-3),
[Pillow](https://python-pillow.org) (MIT-CMU),
[pillow-heif](https://github.com/bigcat88/pillow_heif) (BSD-3),
[FastAPI](https://fastapi.tiangolo.com) (MIT).
