# src/features/build.py
from __future__ import annotations

import pandas as pd
import logging

from .definitions import (
    C_CONSTRUCTOR_ID,
    F_STINT,
    F_TRACK_EVO,
    F_LAP_TIME_PREV,
    F_LAP_TIME_DIFF,
    F_DRIVER_FORM_AVG,
    F_DRIVER_FORM_STD,
    F_TEAM_FORM_AVG,
    F_TEAM_FORM_STD,
    F_RELATIVE_PACE,
    F_POSITION_PREV,     # <--- dodaj w definitions.py jeśli brak
)


from .base import (
    add_track_evolution,
    lap_based_features,
    add_stint_features,
    add_driver_form,
    add_team_formation,
    add_relative_pace,
)

logger = logging.getLogger(__name__)


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

    # 7) Anty-leak: bieżąca pozycja zawiera skutki okrążenia → używamy tylko laga
    if "position" in out.columns:
        out = out.sort_values(["raceId", "driverId", "lap"])
        out[F_POSITION_PREV] = out.groupby(["raceId", "driverId"])["position"].shift(1)
        out.drop(columns=["position"], inplace=True)

        
    return out
