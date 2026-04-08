# src/features/definitions.py
from __future__ import annotations
import pandas as pd

# ---- Wersja schemy ----
FEATURES_VERSION = "v0.1-laps"

# ---- Nazwy kolumn jako zwykłe stringi ----
# Klucze / meta 
C_RACE_ID        = "raceId"
C_YEAR           = "year"
C_ROUND          = "round"
C_CIRCUIT_ID     = "circuitId"
C_DRIVER_ID      = "driverId"
C_CONSTRUCTOR_ID = "constructorId"   # opcjonalne (może nie być)
C_LAP            = "lap"
C_MAX_LAP        = "max_lap"

# Sygnały podstawowe
F_SECONDS        = "seconds"

# Feature engineering
F_TRACK_EVO      = "track_evolution_index"
F_LAP_TIME_PREV  = "lap_time_prev"
F_LAP_TIME_DIFF  = "lap_time_diff"

F_DRIVER_FORM_AVG = "driver_form_avg"
F_DRIVER_FORM_STD = "driver_form_std"

F_TEAM_FORM_AVG   = "team_form_avg"   # opcjonalne
F_TEAM_FORM_STD   = "team_form_std"   # opcjonalne

F_RELATIVE_PACE   = "relative_pace"

F_POSITION_PREV   = "position_prev"   # opcjonalne (u Ciebie jest)

# Stint (opcjonalne)
F_STINT           = "stint"
F_STINT_CHANGE    = "stint_change"
F_STINT_LAP_NO    = "stint_lap_number"

# ---- Minimalny zestaw kolumn artefaktu ----
REQUIRED_COLUMNS = [
    C_RACE_ID, C_YEAR, C_ROUND, C_CIRCUIT_ID,
    C_DRIVER_ID, C_LAP, C_MAX_LAP,
    F_SECONDS,
    F_TRACK_EVO,
    F_LAP_TIME_PREV, F_LAP_TIME_DIFF,
    F_DRIVER_FORM_AVG, F_DRIVER_FORM_STD,
    F_RELATIVE_PACE,
]

# ---- Kolumny opcjonalne  ----
OPTIONAL_COLUMNS = [
    C_CONSTRUCTOR_ID,
    F_TEAM_FORM_AVG, F_TEAM_FORM_STD,
    F_POSITION_PREV,
    F_STINT, F_STINT_CHANGE, F_STINT_LAP_NO,
]

# ---- Pełna lista schemy  ----
ALL_KNOWN_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def assert_feature_schema(df: pd.DataFrame) -> None:
    """
    Sprawdza czy DF zawiera wszystkie wymagane kolumny.
    Nie wymusza kolumn opcjonalnych.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")


def warn_unknown_columns(df: pd.DataFrame) -> list[str]:
    """
    Zwraca listę kolumn, które nie są znane w schemie.
    (Nie rzuca błędu – to tylko sygnał, że rośnie 'śmietnik').
    """
    unknown = [c for c in df.columns if c not in ALL_KNOWN_COLUMNS]
    return unknown
