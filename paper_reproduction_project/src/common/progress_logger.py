"""Small progress logger that writes both console and a heartbeat file."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def emit_progress(message: str, log_path: Optional[Union[str, Path]] = None, console: bool = True) -> None:
    """Emit a progress line immediately and append it to a file when configured."""

    line = "[%s] %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)
    if console:
        print(line, flush=True)
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
