from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

import config


def load_df(target: str, feats: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(config.ART_FEATURES)
    df = df[["raceId", "driverId", target] + feats].copy()

    # numeric + drop braków (prosto)
    for col in [target] + feats:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[target] + feats)

    return df


def main() -> int:
    target = config.MODEL_TARGET
    variant = config.MODEL_VARIANT
    feats = config.get_model_features()

    # --- 3 elementy: szybkość / stabilność / "pewność z błędów" ---
    min_races = 30
    alpha = 0.15
    gamma = 2.00

    df = load_df(target, feats)

    x = df[feats].to_numpy(np.float32)
    y = df[target].to_numpy(np.float32)

    model = tf.keras.models.load_model(config.TF_MODEL_FILE)
    y_pred = model.predict(x, verbose=0).reshape(-1)

    # residual: ujemny = szybciej niż baseline
    residual = y - y_pred

    # ============================================================
    # SMOKE TEST: czy model bije proste baseline’y?
    # ============================================================
    mae_model = float(np.mean(np.abs(residual)))
    mae_zero = float(np.mean(np.abs(y)))  # baseline = 0, a mean(y)≈0

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
    # RACE DE-BIAS: odejmij średni błąd w danym wyścigu
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

    # ===== per-race residual (każdy wyścig waży podobnie) =====
    per_race = tmp.groupby(["driverId", "raceId"], as_index=False).agg(
        residual_race=("residual_adj", "mean")
    )

    # ============================================================
    # "Pewność z błędów", nie z kariery
    # ============================================================
    bad_threshold = float(per_race["residual_race"].quantile(0.90))
    per_race["bad_race"] = (per_race["residual_race"] >= bad_threshold).astype(int)

    # ===== agregacja po kierowcy =====
    skill = per_race.groupby("driverId", as_index=False).agg(
        mean_residual=("residual_race", "mean"),
        std_residual=("residual_race", "std"),
        races=("residual_race", "count"),
        bad_rate=("bad_race", "mean"),
    )

    skill = skill[skill["races"] >= min_races].copy()
    skill["std_residual"] = skill["std_residual"].fillna(0.0)
    skill["bad_rate"] = skill["bad_rate"].fillna(0.0)

    # ===== score (3 składniki, bez karania za krótką karierę) =====
    skill["skill_score"] = (
        skill["mean_residual"]
        + alpha * skill["std_residual"]
        + gamma * skill["bad_rate"]
    )

    skill = skill.sort_values(["skill_score", "races"], ascending=[True, False])

    # ===== nazwiska =====
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
        "races",
    ]
    skill = skill[cols]

    # ===== zapis + print =====
    skill.to_csv(config.SKILL_REPORT, index=False)

    print("\nOK: predict done")
    print(f"Target: {target} | Variant: {variant} | MIN_RACES: {min_races}")
    print(
        f"ALPHA (stability): {alpha} | GAMMA (bad_rate): {gamma} | "
        f"BAD_THRESHOLD(p90): {bad_threshold:.4f}"
    )
    print("Saved:", config.SKILL_REPORT)

    print("\n=== SUPER KIEROWCY (TOP 10) ===")
    print(
        skill.head(10).to_string(index=False)
        if len(skill)
        else "Brak kierowców spełniających MIN_RACES."
    )

    # preview (debug)
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