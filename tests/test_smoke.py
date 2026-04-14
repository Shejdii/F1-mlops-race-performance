from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_FILE = PROJECT_ROOT / "artifacts" / "features" / "features.parquet"


def test_features_smoke() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "features"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "features command failed\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    assert FEATURES_FILE.exists(), f"Missing features file: {FEATURES_FILE}"

    df = pd.read_parquet(FEATURES_FILE)
    assert not df.empty, "features.parquet is empty"
    assert len(df.columns) >= 10, "Too few columns in features.parquet"
