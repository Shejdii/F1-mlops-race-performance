from __future__ import annotations

import argparse
from pathlib import Path

import config


def ensure_dirs() -> None:
    Path(config.ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.ART_FEATURES).parent.mkdir(parents=True, exist_ok=True)
    Path(config.ART_MODELS).mkdir(parents=True, exist_ok=True)
    Path(config.ART_REPORTS).mkdir(parents=True, exist_ok=True)
    Path(config.OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)


def features() -> int:
    """Buduje artifacts/features/features.parquet z danych CSV."""
    from src import io_utils, ingest, clean

    ensure_dirs()

    data_dir = Path(config.DATA_DIR)

    laps, races, drivers, constructors, results, pit_stops = io_utils.load_data(
        data_dir
    )
    base = ingest.lap_times_raw(laps, races)
    df = clean.clean_lap_data(base, races, compute_features=True)

    df.to_parquet(config.ART_FEATURES, index=False)

    print("OK: zapisano", config.ART_FEATURES)
    print("Shape:", df.shape)
    print("Kolumny:", list(df.columns))

    expected = set(config.EXPECTED_FEATURE_COLUMNS)
    present = set(df.columns)
    missing = sorted(expected - present)
    if missing:
        print("\nWARN: Brakuje oczekiwanych kolumn:")
        for col in missing:
            print("  -", col)

    return 0


def train() -> int:
    """Trening modelu (czyta config.py)."""
    ensure_dirs()
    from src.model.train import main as train_main

    return int(train_main())


def predict() -> int:
    """Ranking skill / residuale (czyta config.py)."""
    ensure_dirs()
    from src.model.predict import main as predict_main

    return int(predict_main())


def run_all() -> int:
    """Pełny pipeline: features -> train -> predict."""
    ensure_dirs()
    features()
    train()
    predict()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="f1ml", description="Runner: features -> train -> predict"
    )
    parser.add_argument("cmd", choices=["features", "train", "predict", "all"])
    args = parser.parse_args()

    if args.cmd == "features":
        return features()
    if args.cmd == "train":
        return train()
    if args.cmd == "predict":
        return predict()
    return run_all()


if __name__ == "__main__":
    raise SystemExit(main())
