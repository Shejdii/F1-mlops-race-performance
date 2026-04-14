from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

EXPERIMENT_NAME = "f1-goat-skill-model"
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0]


def load_df(target: str, feats: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(config.ART_FEATURES)
    needed = ["raceId"] + feats + [target]
    df = df[needed].copy()

    for col in feats + [target]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=feats + [target])
    return df


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
    }


def build_tf_model(input_dim: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Normalization(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )


def train_best_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Pipeline, dict[str, float], float, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    best_model: Pipeline | None = None
    best_metrics: dict[str, float] | None = None
    best_alpha: float | None = None

    for alpha in RIDGE_ALPHAS:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(x_train, y_train)
        y_val_pred = model.predict(x_val)
        metrics = evaluate_regression(y_val, y_val_pred)

        rows.append(
            {
                "model_name": f"ridge_alpha_{alpha}",
                "alpha": alpha,
                "val_mae": metrics["mae"],
                "val_mse": metrics["mse"],
            }
        )

        if best_metrics is None or metrics["mae"] < best_metrics["mae"]:
            best_model = model
            best_metrics = metrics
            best_alpha = alpha

    assert best_model is not None
    assert best_metrics is not None
    assert best_alpha is not None

    summary = (
        pd.DataFrame(rows).sort_values("val_mae", ascending=True).reset_index(drop=True)
    )
    return best_model, best_metrics, best_alpha, summary


def train_hgbr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[HistGradientBoostingRegressor, dict[str, float]]:
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=20,
        random_state=config.RANDOM_SEED,
    )
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    metrics = evaluate_regression(y_val, y_val_pred)
    return model, metrics


def train_tf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[tf.keras.Model, dict[str, float], dict[str, float]]:
    model = build_tf_model(input_dim=x_train.shape[1])

    norm_layer = model.layers[0]
    if not isinstance(norm_layer, tf.keras.layers.Normalization):
        raise TypeError("Expected first layer to be tf.keras.layers.Normalization")

    norm_layer.adapt(x_train)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.TF_LR)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.TF_EARLYSTOP_PATIENCE,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.TF_EPOCHS,
        batch_size=config.TF_BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    y_val_pred = model.predict(x_val, verbose=0).reshape(-1)
    metrics = evaluate_regression(y_val, y_val_pred)

    history_metrics = {
        "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "best_val_mae_hist": float(np.min(history.history["val_mae"])),
    }

    return model, metrics, history_metrics


def main() -> int:
    target = config.MODEL_TARGET
    variant = config.MODEL_VARIANT
    feats = config.get_model_features()

    df = load_df(target, feats)

    x = df[feats].to_numpy(np.float32)
    y = df[target].to_numpy(np.float32)
    groups = df["raceId"].to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )
    train_idx, val_idx = next(splitter.split(x, y, groups=groups))

    x_train = x[train_idx]
    x_val = x[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    train_races = set(groups[train_idx].tolist())
    val_races = set(groups[val_idx].tolist())
    overlap = train_races & val_races
    if overlap:
        raise ValueError(
            f"Group split leakage: overlapping raceId values: {sorted(list(overlap))[:10]}"
        )

    Path(config.ART_MODELS).mkdir(parents=True, exist_ok=True)
    Path(config.ART_REPORTS).mkdir(parents=True, exist_ok=True)

    benchmark_path = Path(config.ART_REPORTS) / "train_benchmark_summary.csv"
    ridge_sweep_path = Path(config.ART_REPORTS) / "ridge_alpha_sweep.csv"

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name=f"benchmark_{target}_{variant}_grouped_by_race"
    ) as run:
        mlflow.log_param("target", target)
        mlflow.log_param("variant", variant)
        mlflow.log_param("features", ",".join(feats))
        mlflow.log_param("feature_count", len(feats))
        mlflow.log_param("test_size", config.TEST_SIZE)
        mlflow.log_param("random_seed", config.RANDOM_SEED)
        mlflow.log_param("split_strategy", "GroupShuffleSplit")
        mlflow.log_param("split_group", "raceId")

        mlflow.log_param("ridge_alphas", ",".join(str(a) for a in RIDGE_ALPHAS))
        mlflow.log_param("tf_lr", config.TF_LR)
        mlflow.log_param("tf_epochs", config.TF_EPOCHS)
        mlflow.log_param("tf_batch_size", config.TF_BATCH_SIZE)
        mlflow.log_param("tf_earlystop_patience", config.TF_EARLYSTOP_PATIENCE)

        mlflow.log_metric("train_rows", int(len(x_train)))
        mlflow.log_metric("val_rows", int(len(x_val)))
        mlflow.log_metric("train_races", int(len(train_races)))
        mlflow.log_metric("val_races", int(len(val_races)))

        ridge_model, ridge_metrics, ridge_best_alpha, ridge_sweep = train_best_ridge(
            x_train, y_train, x_val, y_val
        )
        mlflow.log_metric("ridge_val_mae", ridge_metrics["mae"])
        mlflow.log_metric("ridge_val_mse", ridge_metrics["mse"])
        mlflow.log_param("ridge_best_alpha", ridge_best_alpha)

        for _, row in ridge_sweep.iterrows():
            alpha_str = str(row["alpha"]).replace(".", "_")
            mlflow.log_metric(f"ridge_alpha_{alpha_str}_val_mae", float(row["val_mae"]))
            mlflow.log_metric(f"ridge_alpha_{alpha_str}_val_mse", float(row["val_mse"]))

        ridge_sweep.to_csv(ridge_sweep_path, index=False)
        mlflow.log_artifact(str(ridge_sweep_path))

        hgbr_model, hgbr_metrics = train_hgbr(x_train, y_train, x_val, y_val)
        mlflow.log_metric("hgbr_val_mae", hgbr_metrics["mae"])
        mlflow.log_metric("hgbr_val_mse", hgbr_metrics["mse"])

        tf_model, tf_metrics, tf_history_metrics = train_tf(
            x_train, y_train, x_val, y_val
        )
        mlflow.log_metric("tf_val_mae", tf_metrics["mae"])
        mlflow.log_metric("tf_val_mse", tf_metrics["mse"])
        mlflow.log_metric("tf_best_epoch", tf_history_metrics["best_epoch"])
        mlflow.log_metric("tf_best_val_loss", tf_history_metrics["best_val_loss"])
        mlflow.log_metric(
            "tf_best_val_mae_hist", tf_history_metrics["best_val_mae_hist"]
        )
        mlflow.log_metric("tf_num_params", int(tf_model.count_params()))

        joblib.dump(ridge_model, config.RIDGE_MODEL_FILE)
        mlflow.log_artifact(str(config.RIDGE_MODEL_FILE))

        tf_model.save(config.TF_MODEL_FILE)
        mlflow.keras.log_model(tf_model, name="tf_model")
        mlflow.log_artifact(str(config.TF_MODEL_FILE))

        summary = pd.DataFrame(
            [
                {
                    "model_name": "ridge",
                    "val_mae": ridge_metrics["mae"],
                    "val_mse": ridge_metrics["mse"],
                },
                {
                    "model_name": "hist_gradient_boosting",
                    "val_mae": hgbr_metrics["mae"],
                    "val_mse": hgbr_metrics["mse"],
                },
                {
                    "model_name": "tensorflow",
                    "val_mae": tf_metrics["mae"],
                    "val_mse": tf_metrics["mse"],
                },
            ]
        ).sort_values("val_mae", ascending=True)

        summary.to_csv(benchmark_path, index=False)
        mlflow.log_artifact(str(benchmark_path))

        winner = summary.iloc[0]["model_name"]
        winner_mae = float(summary.iloc[0]["val_mae"])

        mlflow.log_param("benchmark_winner", winner)
        mlflow.log_metric("benchmark_best_val_mae", winner_mae)

        print("\n=== RIDGE ALPHA SWEEP ===")
        print(ridge_sweep.to_string(index=False))

        print("\n=== MODEL BENCHMARK (GroupShuffleSplit by raceId) ===")
        print(summary.to_string(index=False))

        print("\nOK: training benchmark done")
        print(f"Target: {target} | Variant: {variant}")
        print(f"Train races: {len(train_races)} | Val races: {len(val_races)}")
        print(f"Best Ridge alpha: {ridge_best_alpha}")
        print(f"Saved Ridge model: {config.RIDGE_MODEL_FILE}")
        print(f"Saved TF model: {config.TF_MODEL_FILE}")
        print(f"Saved Ridge sweep: {ridge_sweep_path}")
        print(f"Saved benchmark summary: {benchmark_path}")
        print(f"MLflow run logged in experiment: {EXPERIMENT_NAME}")
        print(f"MLflow run_id: {run.info.run_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
