# Formula 1 GOAT Driver Modeling (ML)

End-to-end Machine Learning project aimed at estimating **driver skill independent of machinery**, with the long-term goal of identifying a **GOAT (Greatest Of All Time) driver** using data-driven modeling.

This repository contains **Version 1 (baseline)** — a fully working pipeline proving the concept.

---

## Core Idea (GOAT Hypothesis)

Raw race results do **not** measure driver skill directly.  
Performance is influenced by:
- car performance
- track conditions
- race chaos
- reliability

**Hypothesis:**  
If we normalize lap-time data to race-relative pace and model residual performance over many races, we can approximate **true driver skill**.

---

## Project Goal (V1)

Build a **reproducible ML baseline** that:
- processes raw Formula 1 lap-time data
- derives race-relative pace features
- trains a regression model
- evaluates it against strong baselines
- produces interpretable driver-level outputs

This version focuses on **engineering correctness and stability**, not peak model performance.

---

## Data

Source: Kaggle — Formula 1 World Championship (1950–2020)

Main tables used:
- lap_times
- races
- drivers
- results

---

## Pipeline Overview

1. **Ingest**
   - Load raw CSV data
   - Parse lap times into seconds
   - Merge race metadata

2. **Clean**
   - Remove invalid laps
   - Normalize lap indexing
   - Filter unreliable samples

3. **Feature Engineering**
   - Race-relative pace
   - Track evolution index
   - Stability metrics across races

4. **Baseline Modeling**
   - Regression model predicting relative pace
   - Designed to capture consistency, not outliers

5. **Evaluation & Smoke Tests**
   - Zero baseline
   - Per-driver mean baseline (cheat-ish)
   - Model MAE comparison
   - Stability checks (MIN_RACES threshold)

6. **Artifacts**
   - Driver skill report (CSV)
   - Model predictions
   - Evaluation summaries

---

## Baselines & Validation

Baselines implemented:
- Zero predictor
- Per-driver mean predictor
- ML regression baseline

Validation includes:
- Mean Absolute Error comparison
- Driver-level stability constraints
- Sanity checks on data leakage

The ML model **outperforms the zero baseline** and remains stable across drivers with sufficient race history.

---

## Tech Stack

- Python
- Pandas / NumPy
- TensorFlow (Keras)
- CLI-style pipeline
- Reproducible artifacts
- Logging & configuration separation

---

## Current Status

**Version 1 — Baseline Complete**

✔ End-to-end pipeline  
✔ Stable baseline model  
✔ Validated against simple heuristics  
✔ Ready for iteration  

This version confirms that the **GOAT driver modeling approach is viable**.

---

## Future Work (V2+)

- Better car-performance disentanglement
- Season-aware normalization
- Model versioning & experiment tracking
- Advanced models (tree-based, ensembles)
- Cross-era driver comparison

---

## Why This Project

This project is intentionally **engineering-first**:
- correctness > complexity
- baselines > fancy models
- reproducibility > leaderboard chasing

It represents how a real ML system is built — iteratively.

---

### Versioning Note

This repository reflects **V1**.  
Future iterations will extend, not replace, this baseline.