from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf

import config


def load_df(target: str, feats: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(config.ART_FEATURES)
    df = df[["raceId", "driverId", target] + feats].copy()

    # numeric + drop braków (prosto)
    for c in [target] + feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[target] + feats)

    return df


def main() -> int:
    target = config.MODEL_TARGET
    variant = config.MODEL_VARIANT
    feats = config.get_model_features()

    # --- 3 elementy: szybkość / stabilność / "pewność z błędów" ---
    MIN_RACES = 30  # filtr minimalnej liczby wyścigów (żeby nie wygrywał 1 weekend)
    ALPHA = 0.15  # kara za niestabilność (std)
    GAMMA = 2.00  # kara za "bad races" (błędy / dropy / chaos)

    df = load_df(target, feats)

    X = df[feats].to_numpy(np.float32)
    y = df[target].to_numpy(np.float32)

    model = tf.keras.models.load_model(config.TF_MODEL_FILE)
    y_pred = model.predict(X, verbose=0).reshape(-1)

    # residual: ujemny = szybciej niż baseline
    residual = y - y_pred

    # ============================================================
    # SMOKE TEST: czy model bije proste baseline’y?
    # - global mean
    # - per-race mean (mega mocny baseline; jeśli TF tego nie bije -> problem)
    # ============================================================
    mae_model = float(np.mean(np.abs(residual)))

    y_mean = float(np.mean(y))
    mae_mean = float(np.mean(np.abs(y - y_mean)))

    # baseline: średnia targetu per raceId
    df_tmp = pd.DataFrame({"raceId": df["raceId"].astype(int).to_numpy(), "y": y})
    y_race_mean = df_tmp.groupby("raceId")["y"].transform("mean").to_numpy()
    mae_race_mean = float(np.mean(np.abs(y - y_race_mean)))
    print("\n=== BASELINE SMOKE TEST ===")
    mae_model = float(np.mean(np.abs(residual)))
    mae_zero = float(np.mean(np.abs(y)))  # bo baseline = 0, a mean(y)≈0
    df_tmp = pd.DataFrame({"driverId": df["driverId"].astype(int).to_numpy(), "y": y})
    y_driver_mean = df_tmp.groupby("driverId")["y"].transform("mean").to_numpy()
    mae_driver_mean = float(np.mean(np.abs(y - y_driver_mean)))

    print("\n=== BASELINE SMOKE TEST (relative_pace) ===")
    print(f"MAE model         : {mae_model:.4f}")
    print(f"MAE zero baseline : {mae_zero:.4f}")
    print(f"MAE per-driver mean (cheat-ish): {mae_driver_mean:.4f}")

    # ============================================================
    # RACE DE-BIAS: odejmij średni błąd w danym wyścigu
    # (czyli: w GP liczy się kto był lepszy od reszty wg baseline)
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
    # "Pewność z błędów", nie z kariery:
    # liczymy % "bad races" = wyścigi gdzie driver był mocno wolniejszy niż baseline
    # próg bierzemy globalnie (np. 90% najgorszych residual_race)
    # ============================================================
    BAD_THRESHOLD = float(per_race["residual_race"].quantile(0.90))
    per_race["bad_race"] = (per_race["residual_race"] >= BAD_THRESHOLD).astype(int)

    # ===== agregacja po kierowcy =====
    skill = per_race.groupby("driverId", as_index=False).agg(
        mean_residual=("residual_race", "mean"),  # SZYBKOŚĆ (ujemny lepszy)
        std_residual=("residual_race", "std"),  # STABILNOŚĆ (mniej = lepiej)
        races=("residual_race", "count"),  # ile wyścigów
        bad_rate=("bad_race", "mean"),  # PEWNOŚĆ z błędów (mniej = lepiej)
    )

    skill = skill[skill["races"] >= MIN_RACES].copy()
    skill["std_residual"] = skill["std_residual"].fillna(0.0)
    skill["bad_rate"] = skill["bad_rate"].fillna(0.0)

    # ===== score (3 składniki, bez karania za krótką karierę) =====
    # - mean_residual: im bardziej ujemny, tym lepiej
    # - std_residual: kara za chaos
    # - bad_rate: kara za częste "katastrofalne" wyścigi (błędy/outliery)
    skill["skill_score"] = (
        skill["mean_residual"]
        + ALPHA * skill["std_residual"]
        + GAMMA * skill["bad_rate"]
    )

    skill = skill.sort_values(["skill_score", "races"], ascending=[True, False])

    # ===== nazwiska =====
    drivers = pd.read_csv(config.DATA_DIR + "/drivers.csv")
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
    print(f"Target: {target} | Variant: {variant} | MIN_RACES: {MIN_RACES}")
    print(
        f"ALPHA (stability): {ALPHA} | GAMMA (bad_rate): {GAMMA} | BAD_THRESHOLD(p90): {BAD_THRESHOLD:.4f}"
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
