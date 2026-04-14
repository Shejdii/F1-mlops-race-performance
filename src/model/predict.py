from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import config

MIN_RACES = 30
PRIOR_STRENGTH = 120

ELITE_QUANTILE = 0.10
BAD_QUANTILE = 0.90

W_SPEED = 0.36
W_PEAK = 0.28
W_ELITE = 0.16
W_CONSISTENCY = 0.12
W_ERROR = 0.08

UNCERTAINTY_PENALTY_SCALE = 12.0


def load_df(target: str, feats: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(config.ART_FEATURES)
    df = df[["raceId", "driverId", target] + feats].copy()

    for col in [target] + feats:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[target] + feats)

    return df


def percentile_score(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True, ascending=True) * 100.0


def confidence_label(x: float) -> str:
    if x >= 0.85:
        return "HIGH"
    if x >= 0.60:
        return "MEDIUM"
    return "LOW"


def predict_with_backend(x: np.ndarray) -> np.ndarray:
    backend = str(config.MODEL_BACKEND).lower()

    if backend == "tensorflow":
        model = tf.keras.models.load_model(config.TF_MODEL_FILE)
        return model.predict(x, verbose=0).reshape(-1)

    if backend == "ridge":
        model = joblib.load(config.RIDGE_MODEL_FILE)
        return model.predict(x).reshape(-1)

    raise ValueError(f"Unsupported MODEL_BACKEND={config.MODEL_BACKEND}")


def build_driver_ranking(
    df: pd.DataFrame,
    y: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    residual = y - y_pred

    tmp = pd.DataFrame(
        {
            "driverId": df["driverId"].astype(int).to_numpy(),
            "raceId": df["raceId"].astype(int).to_numpy(),
            "residual": residual,
            "y": y,
            "y_pred": y_pred,
        }
    )

    race_bias = tmp.groupby("raceId")["residual"].transform("mean").to_numpy()
    tmp["residual_adj"] = tmp["residual"].to_numpy() - race_bias

    per_race = tmp.groupby(["driverId", "raceId"], as_index=False).agg(
        residual_race=("residual_adj", "mean")
    )

    elite_threshold = float(per_race["residual_race"].quantile(ELITE_QUANTILE))
    bad_threshold = float(per_race["residual_race"].quantile(BAD_QUANTILE))

    per_race["elite_race"] = (per_race["residual_race"] <= elite_threshold).astype(int)
    per_race["bad_race"] = (per_race["residual_race"] >= bad_threshold).astype(int)

    skill = per_race.groupby("driverId", as_index=False).agg(
        mean_residual=("residual_race", "mean"),
        std_residual=("residual_race", "std"),
        races=("residual_race", "count"),
        bad_rate=("bad_race", "mean"),
        elite_rate=("elite_race", "mean"),
        peak_residual=("residual_race", lambda s: s.quantile(0.10)),
    )

    skill = skill[skill["races"] >= MIN_RACES].copy()
    skill["std_residual"] = skill["std_residual"].fillna(0.0)
    skill["bad_rate"] = skill["bad_rate"].fillna(0.0)
    skill["elite_rate"] = skill["elite_rate"].fillna(0.0)
    skill["peak_residual"] = skill["peak_residual"].fillna(skill["mean_residual"])

    shrink_weight = skill["races"] / (skill["races"] + PRIOR_STRENGTH)

    global_mean_residual = float(skill["mean_residual"].mean())
    global_std_residual = float(skill["std_residual"].mean())
    global_bad_rate = float(skill["bad_rate"].mean())
    global_elite_rate = float(skill["elite_rate"].mean())
    global_peak_residual = float(skill["peak_residual"].mean())

    skill["shrunk_mean_residual"] = (
        shrink_weight * skill["mean_residual"]
        + (1.0 - shrink_weight) * global_mean_residual
    )
    skill["shrunk_std_residual"] = (
        shrink_weight * skill["std_residual"]
        + (1.0 - shrink_weight) * global_std_residual
    )
    skill["shrunk_bad_rate"] = (
        shrink_weight * skill["bad_rate"] + (1.0 - shrink_weight) * global_bad_rate
    )
    skill["shrunk_elite_rate"] = (
        shrink_weight * skill["elite_rate"] + (1.0 - shrink_weight) * global_elite_rate
    )
    skill["shrunk_peak_residual"] = (
        shrink_weight * skill["peak_residual"]
        + (1.0 - shrink_weight) * global_peak_residual
    )

    skill["speed_strength"] = -skill["shrunk_mean_residual"]
    skill["peak_strength"] = -skill["shrunk_peak_residual"]
    skill["elite_strength"] = skill["shrunk_elite_rate"]
    skill["consistency_strength"] = -np.log1p(skill["shrunk_std_residual"])
    skill["error_control_strength"] = -skill["shrunk_bad_rate"]

    skill["speed_score"] = percentile_score(skill["speed_strength"])
    skill["peak_score"] = percentile_score(skill["peak_strength"])
    skill["elite_score"] = percentile_score(skill["elite_strength"])
    skill["consistency_score"] = percentile_score(skill["consistency_strength"])
    skill["error_score"] = percentile_score(skill["error_control_strength"])

    skill["confidence"] = shrink_weight.clip(lower=0.0, upper=1.0)
    skill["confidence_label"] = skill["confidence"].apply(confidence_label)

    skill["talent_score"] = (
        W_SPEED * skill["speed_score"]
        + W_PEAK * skill["peak_score"]
        + W_ELITE * skill["elite_score"]
        + W_CONSISTENCY * skill["consistency_score"]
        + W_ERROR * skill["error_score"]
    )

    skill["uncertainty_penalty"] = (
        1.0 - skill["confidence"]
    ) * UNCERTAINTY_PENALTY_SCALE
    skill["adjusted_talent_score"] = (
        skill["talent_score"] - skill["uncertainty_penalty"]
    )

    # lower is better for sorting compatibility
    skill["skill_score"] = -skill["adjusted_talent_score"]

    drivers = pd.read_csv(Path(config.DATA_DIR) / "drivers.csv")
    drivers["driverId"] = drivers["driverId"].astype(int)
    drivers["driver_name"] = (
        drivers["forename"].astype(str) + " " + drivers["surname"].astype(str)
    )

    skill = skill.merge(drivers[["driverId", "driver_name"]], on="driverId", how="left")
    skill = skill.sort_values(["skill_score", "races"], ascending=[True, False])

    return skill, tmp


def main() -> int:
    target = config.MODEL_TARGET
    variant = config.MODEL_VARIANT
    feats = config.get_model_features()

    df = load_df(target, feats)
    x = df[feats].to_numpy(np.float32)
    y = df[target].to_numpy(np.float32)

    y_pred = predict_with_backend(x)
    residual = y - y_pred

    mae_model = float(np.mean(np.abs(residual)))
    mae_zero = float(np.mean(np.abs(y)))

    df_tmp = pd.DataFrame(
        {
            "driverId": df["driverId"].astype(int).to_numpy(),
            "y": y,
        }
    )
    y_driver_mean = df_tmp.groupby("driverId")["y"].transform("mean").to_numpy()
    mae_driver_mean = float(np.mean(np.abs(y - y_driver_mean)))

    print(f"\n=== BASELINE SMOKE TEST ({config.MODEL_BACKEND}) ===")
    print(f"MAE model         : {mae_model:.4f}")
    print(f"MAE zero baseline : {mae_zero:.4f}")
    print(f"MAE per-driver mean (cheat-ish): {mae_driver_mean:.4f}")

    skill, preview_base = build_driver_ranking(df, y, y_pred)

    cols = [
        "driverId",
        "driver_name",
        "skill_score",
        "adjusted_talent_score",
        "talent_score",
        "uncertainty_penalty",
        "mean_residual",
        "std_residual",
        "bad_rate",
        "elite_rate",
        "peak_residual",
        "speed_score",
        "peak_score",
        "elite_score",
        "consistency_score",
        "error_score",
        "confidence",
        "confidence_label",
        "races",
    ]
    skill = skill[cols]

    skill.to_csv(config.SKILL_REPORT, index=False)

    preview = preview_base[
        ["driverId", "raceId", "y", "y_pred", "residual", "residual_adj"]
    ].head(200)
    preview.to_csv(config.PREDICTIONS_OUT, index=False)

    print("\nOK: predict done")
    print(
        f"Target: {target} | Variant: {variant} | MODEL_BACKEND: {config.MODEL_BACKEND}"
    )
    print(
        f"Weights -> speed: {W_SPEED}, peak: {W_PEAK}, elite: {W_ELITE}, "
        f"consistency: {W_CONSISTENCY}, error: {W_ERROR}"
    )
    print(
        f"Shrinkage -> prior_strength: {PRIOR_STRENGTH} | "
        f"uncertainty_penalty_scale: {UNCERTAINTY_PENALTY_SCALE}"
    )
    print("Saved:", config.SKILL_REPORT)

    print("\n=== SUPER KIEROWCY (TOP 10 TALENT) ===")
    print(
        skill.head(10).to_string(index=False)
        if len(skill)
        else "Brak kierowców spełniających MIN_RACES."
    )

    print("Saved preview:", config.PREDICTIONS_OUT)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
