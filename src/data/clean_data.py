# src/clean.py
import logging
import numpy as np
import pandas as pd


# ---- helpers ----
def _time_to_seconds(x):
    if x is None or pd.isna(x) or x == r"\N":
        return np.nan
    s = str(x).strip()
    parts = s.split(":")
    try:
        if len(parts) == 2:  # "m:ss.xxx"
            m, sec = parts
            return float(m) * 60 + float(sec)
        return float(s)  # "ss.xxx"
    except Exception:
        return np.nan


def clean_lap_times(lap_times: pd.DataFrame) -> pd.DataFrame:
    """
    Buduje kolumnę 'seconds' (float) dla czasów okrążeń.
    Preferuje 'milliseconds' (stabilne numerycznie), fallback na parsowanie 'time'.
    Usuwa braki i wartości nielogiczne (<= 0).
    """
    df = lap_times.copy()

    sec_from_ms = None
    if "milliseconds" in df.columns:
        ms = pd.to_numeric(df["milliseconds"], errors="coerce")
        sec_from_ms = ms / 1000.0

    sec_from_time = None
    if "time" in df.columns:
        sec_from_time = df["time"].apply(_time_to_seconds)

    if sec_from_ms is not None and sec_from_time is not None:
        # ms jako źródło prawdy; uzupełnij brakujące z 'time'
        df["seconds"] = sec_from_ms.fillna(sec_from_time)
        # ostrzeż, jeśli duże rozbieżności (>2 ms)
        diff = (sec_from_ms - sec_from_time).abs()
        bad = diff[(~diff.isna()) & (diff > 0.002)].shape[0]
        if bad > 0:
            logging.warning(
                f"[clean_lap_times] {bad} różnic >2ms między 'time' a 'milliseconds'"
            )
    elif sec_from_ms is not None:
        df["seconds"] = sec_from_ms
    elif sec_from_time is not None:
        df["seconds"] = sec_from_time
    else:
        raise KeyError("Brak kolumn 'milliseconds' i 'time' w lap_times")

    # sanity: usuń NaN i wartości nielogiczne
    before = df.shape[0]
    df = df.dropna(subset=["seconds"])
    df = df[df["seconds"] > 0]
    logging.info(
        f"[clean_lap_times] rows_before={before}, rows_after={df.shape[0]}, seconds_na={df['seconds'].isna().sum()}"
    )
    return df


def clean_qualifying(qualifying: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje:
      - q1_s/q2_s/q3_s (sekundy),
      - quali_best_s = czas z ostatniej osiągniętej sesji (Q3 > Q2 > Q1).
    """
    q = qualifying.copy()
    for col in ["q1", "q2", "q3"]:
        if col in q.columns:
            q[f"{col}_s"] = q[col].apply(_time_to_seconds)

    # Priorytet: Q3 -> Q2 -> Q1 (ostatnia osiągnięta sesja)
    cols_priority = [c for c in ["q3_s", "q2_s", "q1_s"] if c in q.columns]
    if cols_priority:
        q["quali_best_s"] = q[cols_priority].bfill(axis=1).iloc[:, 0]
    else:
        q["quali_best_s"] = np.nan

    logging.info(
        "[clean_qualifying] shape=%s | NaNs: q1_s=%s q2_s=%s q3_s=%s best=%s",
        q.shape,
        q.get("q1_s", pd.Series(dtype=float)).isna().sum() if "q1_s" in q else "n/a",
        q.get("q2_s", pd.Series(dtype=float)).isna().sum() if "q2_s" in q else "n/a",
        q.get("q3_s", pd.Series(dtype=float)).isna().sum() if "q3_s" in q else "n/a",
        q["quali_best_s"].isna().sum(),
    )
    return q


def clean_races(races: pd.DataFrame) -> pd.DataFrame:
    """Lekki porządek: podstawowe kolumny, year jako int, filtr >=1950."""
    req = ["raceId", "year", "round", "name"]
    for c in req:
        if c not in races.columns:
            raise KeyError(f"races brakuje kolumny '{c}'")
    df = races.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"])
    df = df[df["year"] >= 1950]
    logging.info(f"[clean_races] shape={df.shape}")
    return df


if __name__ == "__main__":
    import sys

    # jeśli podasz ścieżkę do pliku w terminalu, użyje jej
    # jeśli nie — weźmie domyślnie lap_times.csv
    path = sys.argv[1] if len(sys.argv) > 1 else "data/lap_times.csv"

    df = pd.read_csv(path)

    print("=== Przed czyszczeniem ===")
    print(df.head())
    df.info()

    df_clean = clean_lap_times(df)

    print("\n=== Po czyszczeniu ===")
    print(df_clean.head())
    df_clean.info()
