from __future__ import annotations

import logging
import pandas as pd

from .definitions import (
    C_CONSTRUCTOR_ID,
    F_DRIVER_FORM_AVG,
    F_DRIVER_FORM_STD,
    F_LAP_TIME_DIFF,
    F_LAP_TIME_PREV,
    F_POSITION_PREV,
    F_RELATIVE_PACE,
    F_STINT,
    F_TEAM_FORM_AVG,
    F_TEAM_FORM_STD,
    F_TRACK_EVO,
)
from .base import (
    add_driver_form,
    add_relative_pace,
    add_stint_features,
    add_team_formation,
    add_track_evolution,
    lap_based_features,
)

logger = logging.getLogger(__name__)


def _normalize_position_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ujednolica kolumny pozycji po merge'ach.
    Obsługuje przypadki:
      - position
      - position_x / position_y

    Priorytet:
      1) position
      2) position_x
      3) position_y
    """
    out = df.copy()

    if "position" in out.columns:
        out["position"] = pd.to_numeric(out["position"], errors="coerce")
        return out

    candidates = [c for c in ["position_x", "position_y"] if c in out.columns]
    if not candidates:
        return out

    out["position"] = pd.NA
    for col in candidates:
        out["position"] = out["position"].fillna(out[col])

    out["position"] = pd.to_numeric(out["position"], errors="coerce")
    out.drop(columns=candidates, inplace=True, errors="ignore")

    return out


def build_features(
    df: pd.DataFrame,
    *,
    include_stint: bool | None,
    driver_window: int,
    team_window: int,
) -> pd.DataFrame:
    """
    Składa pełny zestaw feature’ów okrążeń.

    Zasady:
    - kolejność jawna i deterministyczna
    - brak wycieków (shift/rolling w funkcjach bazowych)
    - feature’y opcjonalne tylko jeśli dane istnieją
    """

    out = df.copy()

    # --- 0) Ujednolicenie kolumn po merge'ach ---
    out = _normalize_position_columns(out)

    # --- 1) Track evolution ---
    if F_TRACK_EVO not in out.columns:
        out = add_track_evolution(out)

    # --- 2) Lap-based lags (prev + diff) ---
    need_lap_lags = {F_LAP_TIME_PREV, F_LAP_TIME_DIFF}
    if not need_lap_lags.issubset(out.columns):
        out = lap_based_features(out)

    # --- 3) Driver form (rolling, shifted) ---
    need_driver_form = {F_DRIVER_FORM_AVG, F_DRIVER_FORM_STD}
    if not need_driver_form.issubset(out.columns):
        out = add_driver_form(out, window=driver_window)

    # --- 4) Stint features (opcjonalne) ---
    use_stint = include_stint if include_stint is not None else (F_STINT in out.columns)
    if use_stint:
        if F_STINT not in out.columns:
            raise KeyError("include_stint=True, ale brak kolumny 'stint'")
        out = add_stint_features(out)

    # --- 5) Team form (opcjonalne, tylko jeśli mamy constructorId) ---
    if C_CONSTRUCTOR_ID in out.columns:
        need_team_form = {F_TEAM_FORM_AVG, F_TEAM_FORM_STD}
        if not need_team_form.issubset(out.columns):
            out = add_team_formation(out, window=team_window)

    # --- 6) Relative pace (leave-one-out per race+lap) ---
    if F_RELATIVE_PACE not in out.columns:
        out = add_relative_pace(out)

    # --- 7) Anty-leak: bieżąca pozycja -> tylko lag ---
    if "position" in out.columns:
        out = out.sort_values(["raceId", "driverId", "lap"])
        out[F_POSITION_PREV] = out.groupby(["raceId", "driverId"])["position"].shift(1)
        out.drop(columns=["position"], inplace=True)

    return out
