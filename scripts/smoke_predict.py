from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

import config


def main() -> int:
    model_path = Path(config.RIDGE_MODEL_FILE)
    features_path = Path(config.ART_FEATURES)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")

    # Load model
    model = joblib.load(model_path)

    # Load features
    df = pd.read_parquet(features_path)

    # We only need one valid row
    if df.empty:
        raise RuntimeError("Features dataset is empty")

    # Build sample consistent with training
    target = config.MODEL_TARGET
    feats = config.get_model_features()

    # ensure columns exist
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns: {missing}")

    sample = df[feats].iloc[[0]].copy()

    # Predict
    _ = model.predict(sample)

    # If no exception → success
    print(json.dumps({"success": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())