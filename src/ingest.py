from __future__ import annotations
import pandas as pd
from .time_parse import parse_time_to_seconds


def lap_times_raw(laps: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """
    Buduje bazowy DataFrame okrążeń z lap_times i races:
      - standaryzuje czas do kolumny 'seconds'
      - dołącza meta: year, round, circuitId
      - dodaje 'max_lap' per race
      - nie zostawia kolumn pomocniczych (time/milliseconds/lap_times_raw)
    """
    out = laps.copy()

    # --- 1) Czas okrążenia -> 'seconds' (docelowa kolumna)
    if "seconds" not in out.columns:
        if "time" in out.columns:
            # 'time' jest stringiem "m:ss.xxx" lub "ss.xxx"
            out["seconds"] = out["time"].apply(parse_time_to_seconds)
        elif "milliseconds" in out.columns:
            out["seconds"] = pd.to_numeric(out["milliseconds"], errors="coerce") / 1000.0
        else:
            raise KeyError("Brak kolumny czasu: oczekiwano 'seconds' lub 'time' lub 'milliseconds' w lap_times")

    # --- 2) Oczyszczanie i standardyzacja
    out = out.dropna(subset=["raceId", "driverId", "lap", "seconds"]).copy()
    out["lap"] = pd.to_numeric(out["lap"], errors="coerce").astype(int)
    out["seconds"] = pd.to_numeric(out["seconds"], errors="coerce")

    # --- 3) Dołącz meta wyścigu (tylko potrzebne kolumny, bez *_x/_y)
    races_min = races[["raceId", "year", "round", "circuitId"]].copy()
    out = out.merge(races_min, on="raceId", how="left")

    # --- 4) Dodaj max_lap dla każdego wyścigu
    max_lap = out.groupby("raceId")["lap"].max().rename("max_lap")
    out = out.merge(max_lap, on="raceId", how="left")

    # --- 5) Usuń surowe kolumny czasu jeśli były (redukcja śmieci)
    for col in ("time", "milliseconds", "lap_times_raw"):
        if col in out.columns:
            out.drop(columns=[col], inplace=True)

    return out