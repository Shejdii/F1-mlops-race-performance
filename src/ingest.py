from __future__ import annotations

import pandas as pd

from .time_parse import parse_time_to_seconds


def lap_times_raw(
    laps: pd.DataFrame,
    races: pd.DataFrame,
    results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Buduje bazowy DataFrame okrążeń z lap_times i races:
      - standaryzuje czas do kolumny 'seconds'
      - dołącza meta: year, round, circuitId
      - opcjonalnie dołącza context wyników: constructorId, position
      - dodaje 'max_lap' per race
      - nie zostawia kolumn pomocniczych (time/milliseconds/lap_times_raw)

    Klucze merge dla contextu wyników:
      - raceId
      - driverId
    """

    out = laps.copy()

    # --- 1) Czas okrążenia -> 'seconds' (docelowa kolumna)
    if "seconds" not in out.columns:
        if "time" in out.columns:
            out["seconds"] = out["time"].apply(parse_time_to_seconds)
        elif "milliseconds" in out.columns:
            out["seconds"] = (
                pd.to_numeric(out["milliseconds"], errors="coerce") / 1000.0
            )
        else:
            raise KeyError(
                "Brak kolumny czasu: oczekiwano 'seconds' lub 'time' lub 'milliseconds' w lap_times"
            )

    # --- 2) Oczyszczanie i standaryzacja
    out = out.dropna(subset=["raceId", "driverId", "lap", "seconds"]).copy()
    out["raceId"] = pd.to_numeric(out["raceId"], errors="coerce").astype(int)
    out["driverId"] = pd.to_numeric(out["driverId"], errors="coerce").astype(int)
    out["lap"] = pd.to_numeric(out["lap"], errors="coerce").astype(int)
    out["seconds"] = pd.to_numeric(out["seconds"], errors="coerce")

    # --- 3) Meta wyścigu
    races_min = races[["raceId", "year", "round", "circuitId"]].copy()
    races_min["raceId"] = pd.to_numeric(races_min["raceId"], errors="coerce").astype(
        int
    )
    out = out.merge(races_min, on="raceId", how="left")

    # --- 4) Context wyników (opcjonalny): constructorId, position
    if results is not None:
        required_cols = {"raceId", "driverId", "constructorId", "position"}
        missing = required_cols - set(results.columns)
        if missing:
            raise KeyError(
                f"results missing required columns for lap context merge: {sorted(missing)}"
            )

        results_min = results[
            ["raceId", "driverId", "constructorId", "position"]
        ].copy()
        results_min["raceId"] = pd.to_numeric(
            results_min["raceId"], errors="coerce"
        ).astype(int)
        results_min["driverId"] = pd.to_numeric(
            results_min["driverId"], errors="coerce"
        ).astype(int)
        results_min["constructorId"] = pd.to_numeric(
            results_min["constructorId"], errors="coerce"
        )
        results_min["position"] = pd.to_numeric(
            results_min["position"], errors="coerce"
        )

        # jeden rekord per (raceId, driverId)
        results_min = results_min.drop_duplicates(subset=["raceId", "driverId"])

        out = out.merge(results_min, on=["raceId", "driverId"], how="left")

    # --- 5) Dodaj max_lap dla każdego wyścigu
    max_lap = out.groupby("raceId")["lap"].max().rename("max_lap")
    out = out.merge(max_lap, on="raceId", how="left")

    # --- 6) Usuń surowe kolumny czasu jeśli były
    for col in ("time", "milliseconds", "lap_times_raw"):
        if col in out.columns:
            out.drop(columns=[col], inplace=True)

    return out
