from __future__ import annotations

import pandas as pd
import numpy as np


def _require(df: pd.DataFrame, cols: set[str], fn: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise KeyError(f"{fn}: brak wymaganych kolumn: {sorted(missing)}")


def add_track_evolution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje indeks ewolucji toru w [0,1]: lap / max_lap (zabezpieczenie na dzielenie przez 0/NaN).
    Wymaga: 'lap', 'max_lap'
    """
    _require(df, {"lap", "max_lap"}, "add_track_evolution")
    out = df.copy()
    denom = out["max_lap"].replace(0, np.nan)
    out["track_evolution_index"] = (out["lap"] / denom).clip(0.0, 1.0).fillna(0.0)
    return out


def lap_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje:
      - lap_time_prev: czas poprzedniego okrążenia kierowcy w tym wyścigu
      - lap_time_diff: seconds - lap_time_prev

    Wymaga: 'raceId', 'driverId', 'lap', 'seconds'
    """
    _require(df, {"raceId", "driverId", "lap", "seconds"}, "lap_based_features")
    out = df.copy()
    out = out.sort_values(by=["raceId", "driverId", "lap"])

    out["lap_time_prev"] = out.groupby(["raceId", "driverId"])["seconds"].shift(1)
    out["lap_time_diff"] = out["seconds"] - out["lap_time_prev"]

    return out


def add_stint_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje:
      - 'stint_change' 0/1 (czy zaczyna się nowy stint; pierwszy wiersz = 0),
      - 'stint_lap_number' (kolejny numer okrążenia w stincie, start od 1).
    Wymaga: 'raceId', 'driverId', 'lap', 'stint'
    """
    _require(df, {"raceId", "driverId", "lap", "stint"}, "add_stint_features")
    out = df.copy().sort_values(by=["raceId", "driverId", "lap"])
    prev_stint = out.groupby(["raceId", "driverId"])["stint"].shift(1)
    out["stint_change"] = (
        (out["stint"] != prev_stint).astype(int).where(prev_stint.notna(), 0)
    )
    out["stint_lap_number"] = (
        out.groupby(["raceId", "driverId", "stint"]).cumcount() + 1
    )
    return out


def add_driver_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Rolling forma kierowcy po 'seconds' z poprzednich okrążeń (shift(1)):
      - 'driver_form_avg'
      - 'driver_form_std'

    Liczone w poprawnym porządku czasowym:
      driverId -> year -> round -> lap

    Wymaga: 'driverId', 'year', 'round', 'lap', 'seconds'
    """
    _require(
        df,
        {"driverId", "year", "round", "lap", "seconds"},
        "add_driver_form",
    )

    out = df.copy().sort_values(["driverId", "year", "round", "lap"])

    prev_seconds = out.groupby("driverId")["seconds"].shift(1)

    out["driver_form_avg"] = prev_seconds.groupby(out["driverId"]).transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

    out["driver_form_std"] = (
        prev_seconds.groupby(out["driverId"])
        .transform(lambda x: x.rolling(window, min_periods=1).std())
        .fillna(0.0)
    )

    return out


def add_team_formation(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Rolling forma zespołu (constructorId) po 'seconds' z poprzednich okrążeń (shift(1)):
      - 'team_form_avg'
      - 'team_form_std'

    Liczone w poprawnym porządku czasowym:
      constructorId -> year -> round -> lap

    Wymaga: 'constructorId', 'year', 'round', 'lap', 'seconds'
    """
    _require(
        df,
        {"constructorId", "year", "round", "lap", "seconds"},
        "add_team_formation",
    )

    out = df.copy().sort_values(["constructorId", "year", "round", "lap"])

    prev_seconds = out.groupby("constructorId")["seconds"].shift(1)

    out["team_form_avg"] = prev_seconds.groupby(out["constructorId"]).transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

    out["team_form_std"] = (
        prev_seconds.groupby(out["constructorId"])
        .transform(lambda x: x.rolling(window, min_periods=1).std())
        .fillna(0.0)
    )

    return out


def add_relative_pace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Relatywne tempo względem średniej okrążenia w danym wyścigu i danym numerze okrążenia,
    w wariancie leave-one-out (bez własnego czasu):
      relative_pace = seconds - mean_others(raceId, lap)

    Wymaga: 'raceId', 'lap', 'driverId', 'seconds'
    """
    _require(df, {"raceId", "lap", "driverId", "seconds"}, "add_relative_pace")
    out = df.copy().sort_values(by=["raceId", "lap", "driverId"])

    grp = out.groupby(["raceId", "lap"])["seconds"]
    sum_all = grp.transform("sum")
    cnt_all = grp.transform("count")

    mean_others = (sum_all - out["seconds"]) / (cnt_all - 1).replace(0, np.nan)
    mean_others = mean_others.fillna(out["seconds"])

    out["relative_pace"] = out["seconds"] - mean_others
    return out
