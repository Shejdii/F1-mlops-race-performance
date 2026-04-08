# src/data.py
import os
import logging
import pandas as pd
from src.config import (
    LAP_TIMES, RACES, DRIVERS, DRIVER_STANDINGS, QUALIFYING,
    LOG_FILE, LOG_LEVEL
)

# Helper: sprawdzenie istnienia plików
def _check_exists(path: str) -> None:
    if not os.path.exists(path):
        logging.error(f"Brakuje wymaganego pliku: {path}")
        raise FileNotFoundError(f"Brakuje wymaganego pliku: {path}")

# Główny loader
def load_data():
    """Load and validate required CSV files for the GOAT F1 project."""
    required_files = {
        "lap_times": LAP_TIMES,
        "races": RACES,
        "drivers": DRIVERS,
        "driver_standings": DRIVER_STANDINGS,
        "qualifying": QUALIFYING,
    }

    # Walidacja
    for name, path in required_files.items():
        _check_exists(path)
        logging.info(f"Znaleziono plik: {path}")

    # Wczytanie
    lap_times = pd.read_csv(LAP_TIMES, low_memory=False)
    races = pd.read_csv(RACES, low_memory=False)
    drivers = pd.read_csv(DRIVERS, low_memory=False)
    driver_standings = pd.read_csv(DRIVER_STANDINGS, low_memory=False)
    qualifying = pd.read_csv(QUALIFYING, low_memory=False)

    logging.info(f"Wczytano lap_times: {lap_times.shape}")
    logging.info(f"Wczytano races: {races.shape}")
    logging.info(f"Wczytano drivers: {drivers.shape}")
    logging.info(f"Wczytano driver_standings: {driver_standings.shape}")
    logging.info(f"Wczytano qualifying: {qualifying.shape}")

    # Zwracamy w ustalonej kolejności (krotka)
    return lap_times, races, drivers, driver_standings, qualifying


if __name__ == "__main__":
    logging.basicConfig(
        filename=LOG_FILE,
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    lap_times, races, drivers, driver_standings, qualifying = load_data()

    print("\n=== lap_times ===")
    print(lap_times.head())
    lap_times.info()

    print("\n=== races ===")
    print(races.head())
    races.info()

    print("\n=== drivers ===")
    print(drivers.head())
    drivers.info()

    print("\n=== driver_standings ===")
    print(driver_standings.head())
    driver_standings.info()

    print("\n=== qualifying ===")
    print(qualifying.head())
    qualifying.info()