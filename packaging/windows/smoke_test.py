"""Check the built DigiBookMaker.exe actually works (run by the Windows build)."""

import subprocess
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from synth import CJK_FONT, ENGLISH, JAPANESE, LATIN_FONT, PHOTO_BOX, photograph, render_page  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console is cp1252
exe = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
photos = work / "photos"
photos.mkdir(parents=True, exist_ok=True)
print("fonts:", LATIN_FONT, CJK_FONT)
pages = [
    render_page(ENGLISH, LATIN_FONT, highlight_lines=(8,), page_no=1),
    render_page(JAPANESE, CJK_FONT, page_no=2),
    render_page(ENGLISH[:6], LATIN_FONT, page_no=3, photo_box=PHOTO_BOX),
]
for i, p in enumerate(pages):
    ok, data = cv2.imencode(".jpg", photograph(p, seed=i))
    (photos / f"IMG_{i + 1:04d}.jpg").write_bytes(data.tobytes())

lib = work / "library"
start = time.time()
proc = subprocess.run([str(exe), "--cli", "--library", str(lib), "make", str(photos), "-t", "Smoke"],
                      timeout=900)
print(f"exit code {proc.returncode} after {time.time() - start:.0f}s")
book = lib / "smoke"
problems = []
if proc.returncode != 0:
    problems.append(f"exit code {proc.returncode}")
text = (book / "text.md").read_text(encoding="utf-8") if (book / "text.md").exists() else ""
hl = (book / "highlights.md").read_text(encoding="utf-8") if (book / "highlights.md").exists() else ""
print(text)
print(hl)
if not (book / "smoke.pdf").is_file():
    problems.append("no PDF")
if "Knowledge is only useful" not in text:
    problems.append("English text not read")
if "歴史" not in text:
    problems.append("Japanese text not read")
if "(yellow) Knowledge is only useful" not in hl:
    problems.append("highlight not found")
if not list((book / "figures").glob("*.jpg")):
    problems.append("picture not extracted")

# The window opens and stays open (no crash on start).
gui = subprocess.Popen([str(exe)])
time.sleep(15)
if gui.poll() is not None:
    problems.append(f"window closed by itself (exit code {gui.returncode})")
gui.kill()

if problems:
    log = Path.home() / "AppData" / "Local" / "DigiBookMaker" / "digibook.log"
    if log.exists():
        print("---- digibook.log ----")
        print(log.read_text(encoding="utf-8", errors="replace"))
    sys.exit("SMOKE TEST FAILED: " + "; ".join(problems))
print("SMOKE TEST PASSED")
