from pathlib import Path

import pandas as pd

import config


SAMPLE_PATH = Path("tests/fixtures/features_sample.parquet")


def get_required_columns() -> list[str]:
    return ["raceId", "driverId"] + config.get_model_features() + [config.MODEL_TARGET]


def test_features_sample_exists() -> None:
    assert SAMPLE_PATH.exists(), f"Missing sample dataset: {SAMPLE_PATH}"


def test_features_sample_has_required_columns() -> None:
    df = pd.read_parquet(SAMPLE_PATH)
    required = set(get_required_columns())
    missing = required - set(df.columns)
    assert not missing, f"Missing required columns: {sorted(missing)}"


def test_features_sample_is_not_empty() -> None:
    df = pd.read_parquet(SAMPLE_PATH)
    assert not df.empty, "Sample dataset is empty."


def test_features_sample_has_minimum_shape() -> None:
    df = pd.read_parquet(SAMPLE_PATH)
    assert len(df) >= 100, f"Expected at least 100 rows, got {len(df)}."
    assert len(df.columns) == len(get_required_columns()), (
        f"Expected {len(get_required_columns())} columns, got {len(df.columns)}."
    )


def test_features_sample_has_enough_groups_for_split() -> None:
    df = pd.read_parquet(SAMPLE_PATH)
    assert df["raceId"].nunique() >= 3, "Need at least 3 unique raceId values."
    assert df["driverId"].nunique() >= 4, "Need at least 4 unique driverId values."


def test_features_sample_has_no_missing_values_in_required_columns() -> None:
    df = pd.read_parquet(SAMPLE_PATH)
    required = get_required_columns()
    missing_counts = df[required].isna().sum()
    bad = missing_counts[missing_counts > 0]
    assert bad.empty, f"Found missing values in required columns: {bad.to_dict()}"


def test_target_is_numeric() -> None:
    df = pd.read_parquet(SAMPLE_PATH)
    assert pd.api.types.is_numeric_dtype(df[config.MODEL_TARGET]), (
        f"Column '{config.MODEL_TARGET}' must be numeric."
    )


def test_feature_columns_are_numeric_except_identifiers() -> None:
    df = pd.read_parquet(SAMPLE_PATH)

    non_numeric = [
        col
        for col in config.get_model_features() + [config.MODEL_TARGET]
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    assert not non_numeric, f"Expected numeric dtypes for columns: {non_numeric}"