"""Command line: `digibook make`, `digibook search`, `digibook list`, `digibook serve`."""

from __future__ import annotations

import argparse
import os
import sys

from .library import Library

DEFAULT_LIBRARY = os.environ.get("DIGIBOOK_LIBRARY", "library")


def _progress(stage: str, done: int, total: int) -> None:
    end = "\n" if done >= total else ""
    print(f"\r{stage}: {done}/{total}   ", end=end, flush=True)


def cmd_make(args: argparse.Namespace) -> int:
    from .pipeline import Options, make_book

    opts = Options(lang=args.lang, mode=args.mode, page_size=args.page_size, split=args.split,
                   dpi=args.dpi, title=args.title, ocr=not args.no_ocr)
    if args.jobs:
        opts.jobs = args.jobs
    res = make_book(args.inputs, args.library, opts, progress=_progress)
    print(f"\nDone in {res.seconds:.0f}s: {res.pages} pages, {res.highlight_count} highlights, "
          f"{res.figure_count} pictures")
    print(f"  PDF:        {res.pdf}")
    print(f"  Text:       {res.text}")
    print(f"  Highlights: {res.highlights}")
    print(f"  Pictures:   {res.folder / 'figures'}")
    print(f"  Page check: {res.folder / 'debug'}  (red outline = detected page)")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    lib = Library(args.library)
    hits = lib.search(" ".join(args.query))
    if not hits:
        print("No matches.")
        return 1
    for h in hits:
        print(f"{h.title} — page {h.page}\n    {h.snippet}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for b in Library(args.library).books():
        print(f"{b['slug']:40s} {b['pages']:4d} pages  {b['title']}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .web import create_app

    print(f"Open http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}")
    uvicorn.run(create_app(args.library), host=args.host, port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="digibook", description=__doc__)
    p.add_argument("--library", default=DEFAULT_LIBRARY, help="library folder (default: ./library)")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("make", help="turn photos/scans into a searchable PDF book")
    m.add_argument("inputs", nargs="+", help="image files, folders of images, or PDFs (in page order)")
    m.add_argument("-t", "--title", help="book title (default: folder name)")
    m.add_argument("-l", "--lang", default="auto",
                   help="auto (default), eng, japanese, vertical (tategaki), arabic, all, "
                        "or a Tesseract string like eng+jpn")
    m.add_argument("-m", "--mode", default="color", choices=["color", "gray", "bw", "original"],
                   help="page look: cleaned colour (default), grayscale, black & white, untouched")
    m.add_argument("-s", "--page-size", default="auto",
                   help="auto (median of your photos), a4, a5, a6, b5, b6, letter, bunko, shinsho, or WxH mm")
    m.add_argument("--split", default="auto", choices=["auto", "on", "off"],
                   help="split two-page spreads into separate pages")
    m.add_argument("--dpi", type=int, default=300)
    m.add_argument("-j", "--jobs", type=int, help="pages processed in parallel")
    m.add_argument("--no-ocr", action="store_true", help="only clean up pages, skip text recognition")
    m.set_defaults(func=cmd_make)

    s = sub.add_parser("search", help="search every book in the library")
    s.add_argument("query", nargs="+")
    s.set_defaults(func=cmd_search)

    ls = sub.add_parser("list", help="list books in the library")
    ls.set_defaults(func=cmd_list)

    sv = sub.add_parser("serve", help="start the upload web page")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
