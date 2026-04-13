from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_FILE = PROJECT_ROOT / "artifacts" / "reports" / "driver_skill.csv"


def test_driver_skill_report_contract() -> None:
    if not REPORT_FILE.exists():
        pytest.skip("driver_skill.csv not found yet. Run `make predict` first.")

    df = pd.read_csv(REPORT_FILE)

    required_cols = [
        "driverId",
        "driver_name",
        "skill_score",
        "mean_residual",
        "std_residual",
        "bad_rate",
        "elite_rate",
        "peak_residual",
        "speed_component",
        "consistency_component",
        "error_component",
        "peak_component",
        "elite_component",
        "confidence",
        "sample_penalty",
        "races",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    assert not missing, (
        f"Missing required report columns.\n"
        f"Missing: {missing}\n"
        f"Actual:  {list(df.columns)}"
    )

    print(f"\n[REPORT] rows={len(df)}, cols={len(df.columns)}")
    assert not df.empty, "driver_skill.csv is empty"


def test_driver_skill_report_values_make_sense() -> None:
    if not REPORT_FILE.exists():
        pytest.skip("driver_skill.csv not found yet. Run `make predict` first.")

    df = pd.read_csv(REPORT_FILE)

    assert df["driverId"].notna().all(), "driverId contains nulls"
    assert df["driver_name"].notna().all(), "driver_name contains nulls"
    assert (df["races"] > 0).all(), "Found non-positive race counts"
    assert ((df["bad_rate"] >= 0) & (df["bad_rate"] <= 1)).all(), (
        "bad_rate must be between 0 and 1"
    )
    assert ((df["elite_rate"] >= 0) & (df["elite_rate"] <= 1)).all(), (
        "elite_rate must be between 0 and 1"
    )
    assert ((df["confidence"] >= 0) & (df["confidence"] <= 1)).all(), (
        "confidence must be between 0 and 1"
    )


def test_driver_skill_report_sorted_by_skill_score() -> None:
    if not REPORT_FILE.exists():
        pytest.skip("driver_skill.csv not found yet. Run `make predict` first.")

    df = pd.read_csv(REPORT_FILE)

    scores = df["skill_score"].to_list()
    assert scores == sorted(scores), "driver_skill.csv is not sorted ascending by skill_score"


def test_driver_skill_top_preview() -> None:
    if not REPORT_FILE.exists():
        pytest.skip("driver_skill.csv not found yet.")

    df = pd.read_csv(REPORT_FILE)

    top = df.head(3)

    print("\n[TOP DRIVERS]")
    print(top.to_string(index=False))

    assert len(top) > 0