from pathlib import Path

import pandas as pd

import config


SOURCE_PATH = Path("artifacts/features/features.parquet")
OUTPUT_PATH = Path("tests/fixtures/features_sample.parquet")


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing source dataset: {SOURCE_PATH}")

    target = config.MODEL_TARGET
    feature_columns = config.get_model_features()

    required_columns = ["raceId", "driverId"] + feature_columns + [target]

    df = pd.read_parquet(SOURCE_PATH)

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in source dataset: {missing}")

    df = df[required_columns].copy()
    df = df.dropna(subset=required_columns)

    race_counts = df["raceId"].value_counts().sort_values(ascending=False)
    selected_races = race_counts.head(6).index.tolist()
    sample = df[df["raceId"].isin(selected_races)].copy()

    driver_counts = sample["driverId"].value_counts().sort_values(ascending=False)
    selected_drivers = driver_counts.head(8).index.tolist()
    sample = sample[sample["driverId"].isin(selected_drivers)].copy()

    sample = (
        sample.sort_values(["raceId", "driverId"])
        .groupby(["raceId", "driverId"], as_index=False, group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )

    if sample.empty:
        raise ValueError("Sample dataset is empty after filtering.")
    if sample["raceId"].nunique() < 3:
        raise ValueError("Sample must contain at least 3 unique raceId values.")
    if sample["driverId"].nunique() < 4:
        raise ValueError("Sample must contain at least 4 unique driverId values.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved sample dataset to: {OUTPUT_PATH}")
    print(f"Shape: {sample.shape}")
    print(f"Unique raceId: {sample['raceId'].nunique()}")
    print(f"Unique driverId: {sample['driverId'].nunique()}")
    print(f"Target: {target}")
    print(f"Feature count: {len(feature_columns)}")


if __name__ == "__main__":
    main()