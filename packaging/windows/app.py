"""Entry point of the Windows app (DigiBookMaker.exe).

Double-click: opens the window.
`DigiBookMaker.exe --cli make <photos> ...`: runs the command line (used by the build's self-test).
"""

import multiprocessing
import os
import sys
from pathlib import Path


def _log_to_file() -> None:
    # A windowed .exe has no console: send output to a log file instead of nowhere.
    if sys.stdout is not None and sys.stderr is not None:
        return
    folder = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "DigiBookMaker"
    folder.mkdir(parents=True, exist_ok=True)
    log = open(folder / "digibook.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or log
    sys.stderr = sys.stderr or log


if __name__ == "__main__":
    multiprocessing.freeze_support()  # page workers re-launch this .exe
    _log_to_file()
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        from digibook.cli import main

        sys.exit(main(sys.argv[2:]))
    from digibook.gui import main

    main()
