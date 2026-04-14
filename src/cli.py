from __future__ import annotations

import argparse
from pathlib import Path

import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dirs() -> None:
    Path(config.ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.ART_MODELS).mkdir(parents=True, exist_ok=True)
    Path(config.ART_REPORTS).mkdir(parents=True, exist_ok=True)
    Path(config.OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)


def features() -> int:
    from src import clean, ingest, io_utils

    ensure_dirs()

    data_dir = Path(config.DATA_DIR)

    laps, races, drivers, constructors, results, pit_stops = io_utils.load_data(
        data_dir
    )
    base = ingest.lap_times_raw(laps, races)
    df = clean.clean_lap_data(base, races, compute_features=True)

    Path(config.ART_FEATURES).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.ART_FEATURES, index=False)

    print("OK: saved", config.ART_FEATURES)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    expected = set(config.EXPECTED_FEATURE_COLUMNS)
    present = set(df.columns)
    missing = sorted(expected - present)
    if missing:
        print("\nWARN: Missing expected feature columns:")
        for col in missing:
            print(" -", col)

    return 0


def train() -> int:
    ensure_dirs()
    from src.model.train import main as train_main

    return int(train_main())


def predict() -> int:
    ensure_dirs()
    from src.model.predict import main as predict_main

    return int(predict_main())


def all_steps() -> int:
    rc = features()
    if rc != 0:
        return rc

    rc = train()
    if rc != 0:
        return rc

    rc = predict()
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="f1ml",
        description="Runner: features -> train -> predict",
    )
    parser.add_argument(
        "cmd",
        choices=["features", "train", "predict", "all"],
    )
    args = parser.parse_args()

    if args.cmd == "features":
        return features()
    if args.cmd == "train":
        return train()
    if args.cmd == "predict":
        return predict()
    return all_steps()


if __name__ == "__main__":
    raise SystemExit(main())
