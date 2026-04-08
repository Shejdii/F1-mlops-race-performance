from __future__ import annotations

import argparse
from pathlib import Path

import config


# folder projektu = tam gdzie jest cli.py
DOM = Path(__file__).parent


def ensure_dirs() -> None:
    # tworzymy katalogi wg config (bez dokładania DOM drugi raz)
    Path(config.ARTIFACTS_DIR).mkdir(exist_ok=True)
    Path(config.ART_MODELS).mkdir(parents=True, exist_ok=True)
    Path(config.ART_REPORTS).mkdir(parents=True, exist_ok=True)


def features() -> int:
    """Buduje artifacts/features.parquet z danych CSV."""
    from src import io_utils, ingest, clean

    ensure_dirs()

    # data dir z config; tu DOM jest OK, bo DATA_DIR jest "DataBase"
    data_dir = DOM / config.DATA_DIR

    laps, races, drivers, constructors, results, pit_stops = io_utils.load_data(data_dir)
    base = ingest.lap_times_raw(laps, races)
    df = clean.clean_lap_data(base, races, compute_features=True)

    # zapis wg config (bez DOM / ...)
    df.to_parquet(config.ART_FEATURES, index=False)

    print("OK: zapisano", config.ART_FEATURES)
    print("Shape:", df.shape)
    print("Kolumny:", list(df.columns))

    # prosty guard: czy mamy spodziewane kolumny
    expected = set(config.EXPECTED_FEATURE_COLUMNS)
    present = set(df.columns)
    missing = sorted(expected - present)
    if missing:
        print("\nWARN: Brakuje oczekiwanych kolumn:")
        for c in missing:
            print("  -", c)

    return 0


def train() -> int:
    """Trening TF (czyta config.py)."""
    ensure_dirs()
    from src.model.train import main as train_main
    return int(train_main())


def predict() -> int:
    """Ranking skill (residual) (czyta config.py)."""
    ensure_dirs()
    from src.model.predict import main as predict_main
    return int(predict_main())


def main() -> int:
    p = argparse.ArgumentParser(prog="f1ml", description="Runner: features -> train -> predict")
    p.add_argument("cmd", choices=["features", "train", "predict"])
    args = p.parse_args()

    if args.cmd == "features":
        return features()
    if args.cmd == "train":
        return train()
    return predict()


if __name__ == "__main__":
    raise SystemExit(main())

