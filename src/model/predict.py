from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

import config


def load_df(target: str, feats: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(config.ART_FEATURES)
    df = df[["raceId", "driverId", target] + feats].copy()

    for col in [target] + feats:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[target] + feats)

    return df


def main() -> int:
    target = config.MODEL_TARGET
    variant = config.MODEL_VARIANT
    feats = config.get_model_features()

    # --- parametry rankingu ---
    min_races = 60

    elite_quantile = 0.10
    bad_quantile = 0.90
    confidence_full_at = 100
    sample_penalty_weight = 1.25

    # wagi score
    w_speed = 1.00
    w_consistency = 0.25
    w_error = 1.25
    w_peak = 0.55
    w_elite = 0.40

    df = load_df(target, feats)

    x = df[feats].to_numpy(np.float32)
    y = df[target].to_numpy(np.float32)

    model = tf.keras.models.load_model(config.TF_MODEL_FILE)
    y_pred = model.predict(x, verbose=0).reshape(-1)

    # residual: ujemny = szybciej niż baseline
    residual = y - y_pred

    # ============================================================
    # SMOKE TEST
    # ============================================================
    mae_model = float(np.mean(np.abs(residual)))
    mae_zero = float(np.mean(np.abs(y)))  # baseline = 0 dla relative_pace

    df_tmp = pd.DataFrame(
        {
            "driverId": df["driverId"].astype(int).to_numpy(),
            "y": y,
        }
    )
    y_driver_mean = df_tmp.groupby("driverId")["y"].transform("mean").to_numpy()
    mae_driver_mean = float(np.mean(np.abs(y - y_driver_mean)))

    print("\n=== BASELINE SMOKE TEST (relative_pace) ===")
    print(f"MAE model         : {mae_model:.4f}")
    print(f"MAE zero baseline : {mae_zero:.4f}")
    print(f"MAE per-driver mean (cheat-ish): {mae_driver_mean:.4f}")

    # ============================================================
    # RACE DE-BIAS
    # ============================================================
    tmp = pd.DataFrame(
        {
            "driverId": df["driverId"].astype(int).to_numpy(),
            "raceId": df["raceId"].astype(int).to_numpy(),
            "residual": residual,
        }
    )

    race_bias = tmp.groupby("raceId")["residual"].transform("mean").to_numpy()
    tmp["residual_adj"] = tmp["residual"].to_numpy() - race_bias

    # każdy wyścig waży podobnie
    per_race = tmp.groupby(["driverId", "raceId"], as_index=False).agg(
        residual_race=("residual_adj", "mean")
    )

    # ============================================================
    # KLASY ZACHOWAŃ: elite / bad
    # ============================================================
    elite_threshold = float(per_race["residual_race"].quantile(elite_quantile))
    bad_threshold = float(per_race["residual_race"].quantile(bad_quantile))

    per_race["elite_race"] = (per_race["residual_race"] <= elite_threshold).astype(int)
    per_race["bad_race"] = (per_race["residual_race"] >= bad_threshold).astype(int)

    # ============================================================
    # AGREGACJA PO KIEROWCY
    # ============================================================
    skill = per_race.groupby("driverId", as_index=False).agg(
        mean_residual=("residual_race", "mean"),
        std_residual=("residual_race", "std"),
        races=("residual_race", "count"),
        bad_rate=("bad_race", "mean"),
        elite_rate=("elite_race", "mean"),
        peak_residual=("residual_race", lambda s: s.quantile(0.10)),
    )

    skill = skill[skill["races"] >= min_races].copy()
    skill["std_residual"] = skill["std_residual"].fillna(0.0)
    skill["bad_rate"] = skill["bad_rate"].fillna(0.0)
    skill["elite_rate"] = skill["elite_rate"].fillna(0.0)
    skill["peak_residual"] = skill["peak_residual"].fillna(skill["mean_residual"])

    # ============================================================
    # SKŁADNIKI SCORE
    # ============================================================
    # im bardziej ujemny mean_residual, tym lepiej
    skill["speed_component"] = -skill["mean_residual"]

    # łagodniejsza kara za spread
    skill["consistency_component"] = np.log1p(skill["std_residual"])

    # im mniejszy bad_rate, tym lepiej
    skill["error_component"] = skill["bad_rate"]

    # im bardziej ujemny peak_residual, tym lepiej
    skill["peak_component"] = -skill["peak_residual"]

    # im wyższy elite_rate, tym lepiej
    skill["elite_component"] = skill["elite_rate"]

    # confidence: małe próbki dostają karę, ale nie przez dzielenie
    skill["confidence"] = (skill["races"] / confidence_full_at).clip(upper=1.0)
    skill["confidence"] = skill["confidence"].clip(lower=0.30)

    # ============================================================
    # FINAL SCORE
    # niższy = lepszy
    # ============================================================
    raw_score = (
        -w_speed * skill["speed_component"]
        + w_consistency * skill["consistency_component"]
        + w_error * skill["error_component"]
        - w_peak * skill["peak_component"]
        - w_elite * skill["elite_component"]
    )

    skill["sample_penalty"] = (1.0 - skill["confidence"]) * sample_penalty_weight
    skill["skill_score"] = raw_score + skill["sample_penalty"]

    skill = skill.sort_values(["skill_score", "races"], ascending=[True, False])

    # ============================================================
    # NAZWISKA
    # ============================================================
    drivers = pd.read_csv(config.DATA_DIR / "drivers.csv")
    drivers["driverId"] = drivers["driverId"].astype(int)
    drivers["driver_name"] = (
        drivers["forename"].astype(str) + " " + drivers["surname"].astype(str)
    )

    skill = skill.merge(drivers[["driverId", "driver_name"]], on="driverId", how="left")

    cols = [
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
    skill = skill[cols]

    # ============================================================
    # ZAPIS
    # ============================================================
    skill.to_csv(config.SKILL_REPORT, index=False)

    print("\nOK: predict done")
    print(f"Target: {target} | Variant: {variant} | MIN_RACES: {min_races}")
    print(
        f"elite_q: {elite_quantile} | bad_q: {bad_quantile} | "
        f"elite_threshold: {elite_threshold:.4f} | bad_threshold: {bad_threshold:.4f}"
    )
    print(
        f"Weights -> speed: {w_speed}, consistency: {w_consistency}, "
        f"error: {w_error}, peak: {w_peak}, elite: {w_elite}"
    )
    print(
        f"Confidence -> full_at: {confidence_full_at}, "
        f"sample_penalty_weight: {sample_penalty_weight}"
    )
    print("Saved:", config.SKILL_REPORT)

    print("\n=== SUPER KIEROWCY (TOP 10) ===")
    print(
        skill.head(10).to_string(index=False)
        if len(skill)
        else "Brak kierowców spełniających MIN_RACES."
    )

    preview = pd.DataFrame(
        {
            "driverId": tmp["driverId"].to_numpy(),
            "raceId": tmp["raceId"].to_numpy(),
            "y": y,
            "y_pred": y_pred,
            "residual": residual,
            "residual_adj": tmp["residual_adj"].to_numpy(),
        }
    ).head(200)
    preview.to_csv(config.PREDICTIONS_OUT, index=False)
    print("Saved preview:", config.PREDICTIONS_OUT)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
