from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf

import config


def load_df(target: str, feats: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(config.ART_FEATURES)

    df = df[["raceId", "driverId", target] + feats].copy()

    # liczby + drop braków
    for c in [target] + feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[target] + feats)
    return df


def main() -> int:
    target = config.MODEL_TARGET
    variant = config.MODEL_VARIANT
    feats = config.get_model_features()

    df = load_df(target, feats)

    X = df[feats].to_numpy(np.float32)
    y = df[target].to_numpy(np.float32)
    groups = df["raceId"].to_numpy()

    split = GroupShuffleSplit(
        n_splits=1,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )
    tr_idx, va_idx = next(split.split(X, y, groups=groups))

    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val, y_val = X[va_idx], y[va_idx]

    norm = tf.keras.layers.Normalization()
    norm.adapt(X_train)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        norm,
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.TF_LR),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.TF_EPOCHS,
        batch_size=config.TF_BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_mae",
            patience=config.TF_EARLYSTOP_PATIENCE,
            restore_best_weights=True,
        )],
        verbose=2,
    )

    _, mae = model.evaluate(X_val, y_val, verbose=0)

    print("\nOK: TF trained")
    print(f"Target: {target} | Variant: {variant}")
    print(f"Val MAE: {float(mae):.4f}")

    model.save(config.TF_MODEL_FILE)
    print("Saved:", config.TF_MODEL_FILE)

    return 0



if __name__ == "__main__":
    raise SystemExit(main())