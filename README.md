# F1 MLOps Race Performance

End-to-end MLOps pipeline for modeling Formula 1 race performance and estimating driver skill.

---

## Core idea

Raw race results do **not** reflect true driver skill.

This project models performance relative to race conditions and uses **residual-based analysis** to approximate driver ability — aiming toward a data-driven GOAT evaluation.

---

## What this shows

* Full ML pipeline: **features → train → predict**
* **Leakage-safe validation** (GroupShuffleSplit by raceId)
* **Model benchmarking** (Ridge, HistGB, TensorFlow)
* **MLflow experiment tracking**
* **CI pipeline (GitHub Actions)** running full workflow on sample data

---

## Why this is interesting

* Avoids common ML mistake: **data leakage between laps from the same race**
* Uses **residuals instead of raw performance** to estimate skill
* Separates:

  * heavy pipeline (real data)
  * lightweight CI validation (sample dataset)

---

## Example output

Top drivers by model-based skill:

```text id="z4n1ld"
Rubens Barrichello
Jacques Villeneuve
Jenson Button
Felipe Massa
Fernando Alonso
```

---

## Pipeline

```text id="o7p9vy"
features → train → predict
```

---

## Run locally

```bash
make install
make all
```

---

## Tech

Python • pandas • scikit-learn • TensorFlow • MLflow • pytest • GitHub Actions

---

## Key takeaway

This project focuses on **building a reproducible ML system**, not just training a model.
