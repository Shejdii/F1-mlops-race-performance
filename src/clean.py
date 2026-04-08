# clean.py
from __future__ import annotations
import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .features.build import build_features
from .time_parse import parse_time_to_seconds


logger = logging.getLogger(__name__)


# ------------ helpers ------------
def _require(df: pd.DataFrame, cols: Iterable[str], ctx: str) -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise KeyError(f"[{ctx}] brak wymaganych kolumn: {sorted(missing)}")


def _ensure_seconds_column(df: pd.DataFrame, target_col: str = "seconds") -> pd.DataFrame:
    """
    Zapewnia istnienie kolumny z czasem okrążenia w sekundach.
    Obsługuje:
      - 'seconds' (już gotowe),
      - 'time' (np. '1:23.456') -> parse_time_to_seconds,
      - 'milliseconds' -> /1000.
    """
    out = df.copy()

    if target_col in out.columns:
        return out

    if "time" in out.columns:
        logger.debug("Konwersja 'time' -> '%s' przez parse_time_to_seconds", target_col)
        out[target_col] = out["time"].apply(parse_time_to_seconds)

    elif "milliseconds" in out.columns:
        logger.debug("Konwersja 'milliseconds' -> '%s' dzieląc przez 1000", target_col)
        out[target_col] = out["milliseconds"].astype(float) / 1000.0

    else:
        raise KeyError(
            "Brak kolumn 'seconds' / 'time' / 'milliseconds' — nie mogę zbudować kolumny z czasem w sekundach."
        )

    return out


def _attach_race_meta(df: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """
    Dokleja do okrążeń minimalne meta z tabeli races: year, round, circuitId.
    """
    _require(df, ["raceId"], "_attach_race_meta")
    _require(races, ["raceId", "year", "round", "circuitId"], "_attach_race_meta")

    out = df.merge(
        races[["raceId", "year", "round", "circuitId"]],
        on="raceId",
        how="left",
        validate="many_to_one",
    )
    return out


def _add_max_lap(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, ["raceId", "lap"], "_add_max_lap")
    out = df.copy()
    out["max_lap"] = out.groupby("raceId")["lap"].transform("max")
    return out


# ------------ main cleaning function ------------

def clean_lap_data(
    laps: pd.DataFrame,
    races: pd.DataFrame,
    *,
    require_constructor: bool = False,
    compute_features: bool = True,
    rolling_window_driver: int = 5,
    rolling_window_team: int = 5,
    include_stint_features: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Główny „cleaner” okrążeń.
    Kroki:
      1) normalizacja czasu -> kolumna 'seconds'
      2) sanity-clean + typy
      3) merge meta z 'races'
      4) 'max_lap'
      5) (opcjonalnie) cechy pośrednie: evolution, diffs, formy, relative pace

    Parametry:
      - require_constructor: jeśli True, rzuci błąd gdy w wejściu brak constructorId.
      - compute_features: czy obliczać feature'y (True domyślnie).
      - rolling_window_driver/team: okno dla formy kierowcy/zespołu (bez bieżącego okrążenia).
      - include_stint_features:
            True  -> wymaga kolumny 'stint' i dodaje cechy stintu
            False -> pomija cechy stintu
            None  -> auto: doda, jeśli kolumna 'stint' jest dostępna

    Zwraca:
      DataFrame gotowy do modelowania.
    """
    logger.info("Start clean_lap_data")

    # 1) seconds
    out = _ensure_seconds_column(laps, target_col="seconds")

    # 2) sanity + typy
    base_required = {"raceId", "driverId", "lap", "seconds"}
    if require_constructor:
        base_required.add("constructorId")
    _require(out, base_required, "clean_lap_data:base")

    # drop NA w krytycznych, int dla 'lap'
    out = out.dropna(subset=["raceId", "driverId", "lap", "seconds"]).copy()
    out["lap"] = out["lap"].astype(int)

    # 3) meta – DOŁĄCZ TYLKO, JEŚLI BRAKUJE
    if not {"year", "round", "circuitId"} <= set(out.columns):
        out = _attach_race_meta(out, races)


    # 4) max_lap
    out = _add_max_lap(out)

    # 5) features 
    if compute_features:
   
        logger.info("Wyliczam feature'y…")
        out = build_features(
            out,
            include_stint=include_stint_features,
            driver_window=rolling_window_driver,
            team_window=rolling_window_team,
        )


    # 3) Usuń surowe kolumny czasu, jeśli jeszcze gdzieś się zachowały
    for col in ("time", "milliseconds", "lap_times_raw"):
        if col in out.columns:
            out.drop(columns=[col], inplace=True)

    # porządek kolumn (opcjonalnie)
    preferred_order = [
        "raceId", "year", "round", "circuitId",
        "driverId", "constructorId",
        "lap", "max_lap", "seconds",
        "track_evolution_index",
        "lap_time_diff",
        "stint", "stint_change", "stint_lap_number",
        "driver_form_avg", "driver_form_std",
        "team_form_avg", "team_form_std",
        "relative_pace",
    ]
    cols_present = [c for c in preferred_order if c in out.columns]
    other_cols = [c for c in out.columns if c not in cols_present]
    out = out[cols_present + other_cols]

    logger.info("clean_lap_data: gotowe, shape=%s", out.shape)
    return out

# Opcjonalnie: funkcja „lite”, tylko cleaning bez feature’ów:
def clean_lap_data_lite(laps: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """
    Minimalny cleaning bez feature'ów — seconds, meta, max_lap.
    """
    return clean_lap_data(
        laps,
        races,
        require_constructor=False,
        compute_features=False,
    )

