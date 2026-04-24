from __future__ import annotations

import json
from pathlib import Path

from mlflow.tracking import MlflowClient

METRICS_OUT = Path("artifacts/reports/metrics.json")

EXPERIMENT_NAME = "f1-goat-skill-model"


def get_latest_run():
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {EXPERIMENT_NAME}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError(f"No MLflow runs found for experiment: {EXPERIMENT_NAME}")

    return runs[0]


def _metric(metrics: dict[str, float], name: str) -> float | None:
    value = metrics.get(name)
    return float(value) if value is not None else None


def main() -> int:
    run = get_latest_run()
    metrics = run.data.metrics

    if "ridge_val_mse" not in metrics:
        raise RuntimeError(
            "'ridge_val_mse' not found in MLflow run. "
            f"Available metrics: {list(metrics.keys())}"
        )

    ridge_mse = float(metrics["ridge_val_mse"])
    ridge_rmse = ridge_mse**0.5

    payload = {
        "primary_metric": {
            "name": "rmse",
            "value": ridge_rmse,
        },
        "secondary_metrics": {
            "mae": _metric(metrics, "ridge_val_mae"),
            "mse": ridge_mse,
            "ridge_val_mae": _metric(metrics, "ridge_val_mae"),
            "ridge_val_mse": ridge_mse,
            "tf_val_mae": _metric(metrics, "tf_val_mae"),
            "tf_val_mse": _metric(metrics, "tf_val_mse"),
            "hgbr_val_mae": _metric(metrics, "hgbr_val_mae"),
            "hgbr_val_mse": _metric(metrics, "hgbr_val_mse"),
        },
        "metadata": {
            "mlflow_run_id": run.info.run_id,
            "mlflow_experiment_id": run.info.experiment_id,
            "source": "mlflow",
            "selected_backend": "ridge",
        },
    }

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())