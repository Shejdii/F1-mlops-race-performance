from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger(__name__)


def parse_time_to_seconds(time_str: str | None) -> float:
    """
    Obsługuje czasy w formacie:
    - H:MM:SS.xxx
    - M:SS.xxx
    - SS.xxx

    Zwraca float w sekundach albo np.nan.
    """
    if time_str is None:
        logger.warning("parse_time_to_seconds: received None → returning NaN")
        return np.nan

    s = str(time_str).strip()

    # nic / \N
    if s == "" or s.upper() == r"\N":
        return np.nan

    parts = s.split(":")

    try:
        # Format: H:MM:SS.xxx  (np. "2:05:05.152")
        if len(parts) == 3:
            h, m, sec = parts
            return float(h) * 3600 + float(m) * 60 + float(sec)

        # Format: M:SS.xxx  (np. "1:05.432")
        if len(parts) == 2:
            m, sec = parts
            return float(m) * 60 + float(sec)

        # Format: SS.xxx
        return float(s)

    except Exception as err:
        logger.error(
            "parse_time_to_seconds: failed to parse value '%s' → returning NaN. Error: %s",
            time_str,
            str(err),
        )
        return np.nan
