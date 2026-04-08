from __future__ import annotations
from pathlib import Path
import json
import pandas as pd


def ensure_dirs(*paths: str | Path):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_data(data_dir: str | Path):
    data_dir = Path(data_dir)
    laps = pd.read_csv(data_dir / "lap_times.csv")
    races = pd.read_csv(data_dir / "races.csv")
    drivers = pd.read_csv(data_dir / "drivers.csv")
    constructors = pd.read_csv(data_dir / "constructors.csv")
    results = pd.read_csv(data_dir / "results.csv")
    pits = data_dir / "pit_stops.csv"
    pit_stops = pd.read_csv(pits) if pits.exists() else None

    return laps, races, drivers, constructors, results, pit_stops


def save_json(path: str | Path, **payload):
    Path(path).write_text(json.dumps(payload, indent=2))   