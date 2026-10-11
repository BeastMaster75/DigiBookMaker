"""Desktop window: pick page photos, press one button, get a book."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .library import Library
from .sources import IMAGE_EXTS, PDF_EXTS, natural_key

try:  # drag and drop (bundled in the Windows app; optional otherwise)
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:  # pragma: no cover
    TkinterDnD = None

LANGS = {
    "Auto-detect": "auto",
    "English": "eng",
    "Japanese (horizontal)": "japanese",
    "Japanese (vertical / tategaki)": "vertical",
    "Arabic": "arabic",
    "Mixed English + Japanese + Arabic": "all",
}
MODES = {
    "Cleaned colour": "color",
    "Grayscale": "gray",
    "Black & white": "bw",
    "Untouched photo": "original",
}
SIZES = {
    "Match my pages": "auto",
    "A5": "a5",
    "A4": "a4",
    "B6": "b6",
    "Bunko (A6)": "bunko",
    "Letter": "letter",
}
SPLITS = {
    "Split automatically": "auto",
    "Always split": "on",
    "Never split": "off",
}
STAGES = {"Finding pages": (0.0, 0.3), "Cleaning + reading text": (0.3, 0.95),
          "Building PDF": (0.95, 1.0), "Done": (1.0, 1.0)}
FILETYPES = [("Photos, scans and PDFs",
              " ".join(f"*{e}" for e in sorted(IMAGE_EXTS | PDF_EXTS))), ("All files", "*.*")]


def default_library() -> Path:
    env = os.environ.get("DIGIBOOK_LIBRARY")
    if env:
        return Path(env)
    docs = Path.home() / "Documents"
    return (docs if docs.is_dir() else Path.home()) / "DigiBookMaker"


def open_path(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class App:
    def __init__(self, root: tk.Tk, library_dir: Path):
        self.root = root
        self.library_dir = library_dir
        self.library = Library(library_dir)
        self.files: list[Path] = []
        self.events: queue.Queue = queue.Queue()
        self.result = None
        self.busy = False

        root.title("DigiBookMaker")
        root.geometry("880x660")
        root.minsize(720, 560)
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 8))
        style.configure("Hint.TLabel", foreground="#666666")

        tabs = ttk.Notebook(root)
        tabs.pack(fill="both", expand=True, padx=10, pady=10)
        self.make_tab = ttk.Frame(tabs, padding=12)
        self.search_tab = ttk.Frame(tabs, padding=12)
        tabs.add(self.make_tab, text="  Make a book  ")
        tabs.add(self.search_tab, text="  Search & library  ")
        tabs.bind("<<NotebookTabChanged>>", lambda e: self.refresh_books())
        self._build_make_tab()
        self._build_search_tab()
        root.after(300, self._check_tesseract)
        root.after(150, self._poll)

    # ------------------------------------------------------------------ make tab

    def _build_make_tab(self) -> None:
        f = self.make_tab
        f.columnconfigure(0, weight=1)
        f.rowconfigure(1, weight=1)

        hint = "Add your page photos (or a scanned PDF). They become pages in this order."
        if TkinterDnD is not None:
            hint += "  You can also drag files here."
        ttk.Label(f, text=hint, style="Hint.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")

        list_frame = ttk.Frame(f)
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 10))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(list_frame, selectmode="extended", activestyle="none",
                                  font=("Segoe UI", 10), height=12)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.bind("<Delete>", lambda e: self.remove_selected())
        if TkinterDnD is not None:
            self.listbox.drop_target_register(DND_FILES)
            self.listbox.dnd_bind("<<Drop>>", self._on_drop)

        btns = ttk.Frame(f)
        btns.grid(row=1, column=1, sticky="n", padx=(10, 0), pady=(6, 0))
        for text, cmd in (("Add photos…", self.add_files), ("Add folder…", self.add_folder),
                          (None, None), ("Move up", lambda: self.move(-1)),
                          ("Move down", lambda: self.move(1)), ("Sort by name", self.sort_files),
                          (None, None), ("Remove", self.remove_selected), ("Clear all", self.clear)):
            if text is None:
                ttk.Separator(btns).pack(fill="x", pady=6)
            else:
                ttk.Button(btns, text=text, command=cmd, width=16).pack(fill="x", pady=2)

        opts = ttk.Frame(f)
        opts.grid(row=2, column=0, columnspan=2, sticky="ew")
        for c in range(4):
            opts.columnconfigure(c, weight=1)
        self.title_var = tk.StringVar()
        self.lang_var = tk.StringVar(value="Auto-detect")
        self.mode_var = tk.StringVar(value="Cleaned colour")
        self.size_var = tk.StringVar(value="Match my pages")
        self.split_var = tk.StringVar(value="Split automatically")
        fields = [("Book title", None, self.title_var), ("Language", LANGS, self.lang_var),
                  ("Page look", MODES, self.mode_var), ("Page size", SIZES, self.size_var),
                  ("Two-page photos", SPLITS, self.split_var)]
        for i, (label, choices, var) in enumerate(fields):
            cell = ttk.Frame(opts)
            cell.grid(row=i // 4, column=i % 4, sticky="ew", padx=(0, 10), pady=4)
            ttk.Label(cell, text=label).pack(anchor="w")
            if choices is None:
                ttk.Entry(cell, textvariable=var).pack(fill="x")
            else:
                ttk.Combobox(cell, textvariable=var, values=list(choices), state="readonly").pack(fill="x")

        run = ttk.Frame(f)
        run.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        run.columnconfigure(1, weight=1)
        self.go_btn = ttk.Button(run, text="Make book", style="Big.TButton", command=self.start)
        self.go_btn.grid(row=0, column=0, rowspan=2, padx=(0, 12))
        self.progress = ttk.Progressbar(run, maximum=1.0)
        self.progress.grid(row=0, column=1, sticky="ew")
        self.status = ttk.Label(run, text="Add some pages to start.")
        self.status.grid(row=1, column=1, sticky="w", pady=(4, 0))

        self.result_frame = ttk.Frame(f)
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.result_btns = []
        for text, key in (("Open PDF", "pdf"), ("Open folder", "folder"),
                          ("Highlights", "highlights"), ("Text", "text")):
            b = ttk.Button(self.result_frame, text=text, state="disabled",
                           command=lambda k=key: self.open_result(k))
            b.pack(side="left", padx=(0, 8))
            self.result_btns.append(b)

    def _add(self, paths) -> None:
        added = 0
        for p in paths:
            p = Path(p)
            if p.is_dir():
                found = sorted((x for x in p.iterdir() if x.suffix.lower() in IMAGE_EXTS | PDF_EXTS),
                               key=natural_key)
                if found and not self.title_var.get().strip():
                    self.title_var.set(p.name)
                new = found
            elif p.suffix.lower() in IMAGE_EXTS | PDF_EXTS:
                new = [p]
            else:
                new = []
            for x in new:
                if x not in self.files:
                    self.files.append(x)
                    added += 1
        self._refresh_list()
        if not added and paths:
            self.status.configure(text="No photos or PDFs found there.")

    def _refresh_list(self) -> None:
        self.listbox.delete(0, "end")
        for i, p in enumerate(self.files, 1):
            self.listbox.insert("end", f"{i:3d}.  {p.name}      ({p.parent})")
        if not self.busy:
            n = len(self.files)
            self.status.configure(text=f"{n} file(s) ready." if n else "Add some pages to start.")

    def _on_drop(self, event) -> None:
        self._add(self.root.tk.splitlist(event.data))

    def add_files(self) -> None:
        self._add(filedialog.askopenfilenames(title="Choose page photos", filetypes=FILETYPES))

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose a folder of page photos")
        if folder:
            self._add([folder])

    def move(self, step: int) -> None:
        sel = list(self.listbox.curselection())
        if not sel:
            return
        order = sel if step < 0 else reversed(sel)
        for i in order:
            j = i + step
            if 0 <= j < len(self.files) and j not in sel:
                self.files[i], self.files[j] = self.files[j], self.files[i]
                sel[sel.index(i)] = j
        self._refresh_list()
        for i in sel:
            self.listbox.selection_set(i)

    def sort_files(self) -> None:
        self.files.sort(key=natural_key)
        self._refresh_list()

    def remove_selected(self) -> None:
        for i in sorted(self.listbox.curselection(), reverse=True):
            del self.files[i]
        self._refresh_list()

    def clear(self) -> None:
        self.files.clear()
        self._refresh_list()

    def start(self) -> None:
        if self.busy:
            return
        if not self.files:
            messagebox.showinfo("DigiBookMaker", "Add some page photos first.")
            return
        from .pipeline import Options

        title = self.title_var.get().strip() or time.strftime("Book %Y-%m-%d %H%M")
        opts = Options(title=title, lang=LANGS[self.lang_var.get()], mode=MODES[self.mode_var.get()],
                       page_size=SIZES[self.size_var.get()], split=SPLITS[self.split_var.get()])
        self.busy = True
        self.go_btn.configure(state="disabled")
        for b in self.result_btns:
            b.configure(state="disabled")
        self.progress["value"] = 0
        self.status.configure(text="Starting…")
        threading.Thread(target=self._work, args=(list(self.files), opts), daemon=True).start()

    def _work(self, files: list[Path], opts) -> None:
        from .pipeline import make_book

        try:
            res = make_book(files, self.library_dir, opts,
                            progress=lambda s, d, t: self.events.put(("progress", s, d, t)))
            self.events.put(("done", res))
        except Exception as exc:
            traceback.print_exc()
            self.events.put(("error", str(exc) or exc.__class__.__name__))

    def _poll(self) -> None:
        try:
            while True:
                ev = self.events.get_nowait()
                if ev[0] == "progress":
                    _, stage, done, total = ev
                    lo, hi = STAGES.get(stage, (0.0, 1.0))
                    self.progress["value"] = lo + (hi - lo) * (done / max(1, total))
                    self.status.configure(text=f"{stage}… {done}/{total}")
                elif ev[0] == "done":
                    res = ev[1]
                    self.result = res
                    self.busy = False
                    self.go_btn.configure(state="normal")
                    self.progress["value"] = 1.0
                    self.status.configure(
                        text=f"Done in {res.seconds:.0f}s: {res.pages} pages, "
                             f"{res.highlight_count} highlights, {res.figure_count} pictures.  "
                             f"Saved in {res.folder}")
                    for b in self.result_btns:
                        b.configure(state="normal")
                    self.refresh_books()
                elif ev[0] == "error":
                    self.busy = False
                    self.go_btn.configure(state="normal")
                    self.status.configure(text="Something went wrong.")
                    messagebox.showerror("DigiBookMaker", ev[1])
        except queue.Empty:
            pass
        self.root.after(150, self._poll)

    def open_result(self, key: str) -> None:
        if self.result is None:
            return
        open_path({"pdf": self.result.pdf, "folder": self.result.folder,
                   "highlights": self.result.highlights, "text": self.result.text}[key])

    def _check_tesseract(self) -> None:
        from . import ocr

        try:
            have = ocr.available_langs()
        except ocr.TesseractMissing as exc:
            messagebox.showwarning("Text reading is not available", str(exc))
            return
        missing = [name for code, name in (("jpn", "Japanese"), ("jpn_vert", "Japanese vertical"),
                                           ("ara", "Arabic")) if code not in have]
        if missing:
            messagebox.showwarning(
                "Some languages are missing",
                "Tesseract is installed but without: " + ", ".join(missing) + ".\n\n"
                "Reinstall it and tick those under 'Additional language data'.")

    # ------------------------------------------------------------------ search tab

    def _build_search_tab(self) -> None:
        f = self.search_tab
        f.columnconfigure(0, weight=1)
        f.rowconfigure(2, weight=3)
        f.rowconfigure(5, weight=2)

        bar = ttk.Frame(f)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)
        self.q_var = tk.StringVar()
        entry = ttk.Entry(bar, textvariable=self.q_var, font=("Segoe UI", 11))
        entry.grid(row=0, column=0, sticky="ew")
        entry.bind("<Return>", lambda e: self.search())
        ttk.Button(bar, text="Search", command=self.search).grid(row=0, column=1, padx=(8, 0))
        ttk.Label(f, text="Searches every book, in any language. Double-click a result to open the PDF.",
                  style="Hint.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 4))

        self.hits = ttk.Treeview(f, columns=("book", "page", "text"), show="headings", height=8)
        for col, text, w in (("book", "Book", 160), ("page", "Page", 50), ("text", "Text", 520)):
            self.hits.heading(col, text=text)
            self.hits.column(col, width=w, stretch=(col == "text"))
        self.hits.grid(row=2, column=0, sticky="nsew")
        self.hits.bind("<Double-1>", self._open_hit)

        ttk.Label(f, text="Library", font=("Segoe UI", 11, "bold")).grid(row=3, column=0, sticky="w",
                                                                         pady=(14, 4))
        lb = ttk.Frame(f)
        lb.grid(row=4, column=0, sticky="ew")
        for text, cmd in (("Open PDF", lambda: self._book_action("pdf")),
                          ("Open folder", lambda: self._book_action("folder")),
                          ("Delete", lambda: self._book_action("delete")),
                          ("Open library folder", lambda: open_path(self.library_dir))):
            ttk.Button(lb, text=text, command=cmd).pack(side="left", padx=(0, 8))
        self.books = ttk.Treeview(f, columns=("title", "pages", "created"), show="headings", height=6)
        for col, text, w in (("title", "Title", 400), ("pages", "Pages", 60), ("created", "Made", 160)):
            self.books.heading(col, text=text)
            self.books.column(col, width=w, stretch=(col == "title"))
        self.books.grid(row=5, column=0, sticky="nsew", pady=(6, 0))
        self.books.bind("<Double-1>", lambda e: self._book_action("pdf"))

    def search(self) -> None:
        self.hits.delete(*self.hits.get_children())
        q = self.q_var.get().strip()
        if not q:
            return
        hits = self.library.search(q)
        for h in hits:
            self.hits.insert("", "end", iid=f"{h.slug}|{h.page}|{len(self.hits.get_children())}",
                             values=(h.title, h.page, h.snippet))
        if not hits:
            self.hits.insert("", "end", values=("", "", "No matches."))

    def _pdf_of(self, slug: str) -> Path | None:
        pdfs = list(self.library.book_dir(slug).glob("*.pdf"))
        return pdfs[0] if pdfs else None

    def _open_hit(self, _event) -> None:
        sel = self.hits.selection()
        if sel and "|" in sel[0]:
            pdf = self._pdf_of(sel[0].split("|")[0])
            if pdf:
                open_path(pdf)

    def refresh_books(self) -> None:
        self.books.delete(*self.books.get_children())
        for b in self.library.books():
            made = time.strftime("%Y-%m-%d %H:%M", time.localtime(b["created"]))
            self.books.insert("", "end", iid=b["slug"], values=(b["title"], b["pages"], made))

    def _book_action(self, action: str) -> None:
        sel = self.books.selection()
        if not sel:
            messagebox.showinfo("DigiBookMaker", "Select a book in the list first.")
            return
        slug = sel[0]
        if action == "pdf":
            pdf = self._pdf_of(slug)
            if pdf:
                open_path(pdf)
        elif action == "folder":
            open_path(self.library.book_dir(slug))
        elif action == "delete":
            if messagebox.askyesno("Delete book", "Delete this book and all its files?"):
                import shutil

                self.library.remove_book(slug)
                shutil.rmtree(self.library.book_dir(slug), ignore_errors=True)
                self.refresh_books()


def main(library_dir: str | Path | None = None) -> None:
    lib = Path(library_dir) if library_dir else default_library()
    lib.mkdir(parents=True, exist_ok=True)
    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    App(root, lib)
    root.mainloop()


if __name__ == "__main__":
    main()
