from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config


def load_df(target: str, feats: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(config.ART_FEATURES)
    df = df[feats + [target]].copy()

    for col in feats + [target]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=feats + [target])
    return df


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Normalization(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    return model


def main() -> int:
    target = config.MODEL_TARGET
    variant = config.MODEL_VARIANT
    feats = config.get_model_features()

    df = load_df(target, feats)

    x = df[feats].to_numpy(np.float32)
    y = df[target].to_numpy(np.float32)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )

    model = build_model(input_dim=x_train.shape[1])

    norm_layer = model.layers[0]
    if not isinstance(norm_layer, tf.keras.layers.Normalization):
        raise TypeError("Expected first layer to be tf.keras.layers.Normalization")

    norm_layer.adapt(x_train)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.TF_LR)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.TF_EARLYSTOP_PATIENCE,
            restore_best_weights=True,
        )
    ]

    Path(config.ART_MODELS).mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("f1-goat-skill-model")

    with mlflow.start_run(run_name=f"tf_{target}_{variant}"):
        mlflow.log_param("target", target)
        mlflow.log_param("variant", variant)
        mlflow.log_param("features", ",".join(feats))
        mlflow.log_param("feature_count", len(feats))
        mlflow.log_param("test_size", config.TEST_SIZE)
        mlflow.log_param("random_seed", config.RANDOM_SEED)

        mlflow.log_param("tf_lr", config.TF_LR)
        mlflow.log_param("tf_epochs", config.TF_EPOCHS)
        mlflow.log_param("tf_batch_size", config.TF_BATCH_SIZE)
        mlflow.log_param("tf_earlystop_patience", config.TF_EARLYSTOP_PATIENCE)

        mlflow.log_metric("train_rows", int(len(x_train)))
        mlflow.log_metric("val_rows", int(len(x_val)))
        mlflow.log_metric("num_params", int(model.count_params()))

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
        val_mae = float(np.mean(np.abs(y_val - y_val_pred)))
        val_mse = float(np.mean((y_val - y_val_pred) ** 2))

        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        best_val_loss = float(np.min(history.history["val_loss"]))
        best_val_mae_hist = float(np.min(history.history["val_mae"]))

        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_mse", val_mse)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_val_mae_hist", best_val_mae_hist)

        model.save(config.TF_MODEL_FILE)
        mlflow.keras.log_model(model, name="model")

        mlflow.log_artifact(str(config.TF_MODEL_FILE))

        print("\nOK: TF trained")
        print(f"Target: {target} | Variant: {variant}")
        print(f"Val MAE: {val_mae:.4f}")
        print(f"Saved: {config.TF_MODEL_FILE}")
        print(f"MLflow run logged in experiment: f1-goat-skill-model")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())