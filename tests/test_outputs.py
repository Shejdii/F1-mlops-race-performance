from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_FILE = PROJECT_ROOT / "artifacts" / "features" / "features.parquet"
TF_MODEL_FILE = PROJECT_ROOT / "artifacts" / "models" / "tf_model.keras"
REPORT_FILE = PROJECT_ROOT / "artifacts" / "reports" / "driver_skill.csv"
PREVIEW_FILE = PROJECT_ROOT / "artifacts" / "reports" / "predictions_preview.csv"


def test_features_output_contract() -> None:
    assert FEATURES_FILE.exists(), f"Missing features file: {FEATURES_FILE}"

    df = pd.read_parquet(FEATURES_FILE)

    print(f"\n[FEATURES] rows={len(df)}, cols={len(df.columns)}")

    expected = set(config.EXPECTED_FEATURE_COLUMNS)
    present = set(df.columns)
    missing = sorted(expected - present)

    assert not missing, f"Missing expected feature columns: {missing}"
    assert not df.empty, "features.parquet is empty"


def test_model_output_exists() -> None:
    if not TF_MODEL_FILE.exists():
        pytest.skip("TF model not found yet. Run `make train` first.")

    size_mb = TF_MODEL_FILE.stat().st_size / (1024 * 1024)

    print(f"\n[MODEL] path={TF_MODEL_FILE}, size={size_mb:.2f} MB")

    assert TF_MODEL_FILE.stat().st_size > 0


def test_report_outputs_exist() -> None:
    if not REPORT_FILE.exists() or not PREVIEW_FILE.exists():
        pytest.skip("Report outputs not found yet. Run `make predict` first.")

    assert REPORT_FILE.is_file(), f"Missing report file: {REPORT_FILE}"
    assert PREVIEW_FILE.is_file(), f"Missing preview file: {PREVIEW_FILE}"
    assert REPORT_FILE.stat().st_size > 0, "driver_skill.csv is empty"
    assert PREVIEW_FILE.stat().st_size > 0, "predictions_preview.csv is empty"
