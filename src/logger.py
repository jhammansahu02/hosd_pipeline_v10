"""
Centralised logging: console + rotating file.
"""

import io
import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler


def get_logger(name: str = "hosd", log_dir: Path = Path("./logs")) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ─────────────────────────────────────
    # On Windows the default console codec is cp1252 which cannot
    # encode Unicode box-drawing characters.  Wrap stdout in a
    # UTF-8 TextIOWrapper so any character can be printed; if that
    # fails (e.g. redirected pipe), fall back to errors="replace".
    try:
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
        ch = logging.StreamHandler(utf8_stdout)
    except AttributeError:
        # sys.stdout has no .buffer (already wrapped, Jupyter, etc.)
        ch = logging.StreamHandler(sys.stdout)

    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── File handler (always UTF-8) ──────────────────────────
    fh = RotatingFileHandler(
        log_file, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
