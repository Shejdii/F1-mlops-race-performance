from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    level: str = "INFO", file_path: str | None = None, to_console: bool = True
):
    """Configure logging with optional rotating file + console.
    Creates parent dirs if needed.
    """
    lvl = getattr(logging, str(level).upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    if file_path:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(p, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        handlers.append(fh)

    if to_console:
        ch = logging.StreamHandler()
        ch.setLevel(lvl)
        ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        handlers.append(ch)

    # root config
    logging.basicConfig(level=lvl, handlers=handlers)
    return logging.getLogger("f1ideal")


### {
# from __future__ import annotations
# import logging
# from logging.handlers import RotatingFileHandler
# from pathlib import Path


# def setup_logging(level: str = "INFO", file_path: str | None = None, to_console: bool = True):
# """Configure logging with optional rotating file + console.
# Creates parent dirs if needed.
# """
# lvl = getattr(logging, str(level).upper(), logging.INFO)


# handlers: list[logging.Handler] = []


# if file_path:
# p = Path(file_path)
# p.parent.mkdir(parents=True, exist_ok=True)
# fh = RotatingFileHandler(p, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
# fh.setLevel(lvl)
# fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
# handlers.append(fh)


# if to_console:
# ch = logging.StreamHandler()
# ch.setLevel(lvl)
# ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
# handlers.append(ch)


# root config
# logging.basicConfig(level=lvl, handlers=handlers)
# return logging.getLogger("f1ideal")
### }
