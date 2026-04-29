# 🏎️ F1 Driver Skill Modeling (MLOps Pipeline)

End-to-end MLOps pipeline for estimating **driver skill independent of machinery**, using race-level normalization and leakage-safe validation.

---

## ⚡ What this project does

This system:

* builds **lap-level and race-relative features** from raw F1 data  
* trains and benchmarks multiple models (Ridge, HistGB, TensorFlow)  
* evaluates performance using **leakage-aware validation (GroupSplit by raceId)**  
* estimates driver skill via **residual modeling**  
* tracks experiments with MLflow  
* runs full pipeline reproducibly via CLI + Makefile + CI  

---

## 🎯 Core idea

Raw race results do not reflect true driver skill.

Performance is influenced by:

* car performance  
* track conditions  
* race dynamics  

To reduce these effects, the system models relative performance within each race:

```text
relative_pace = driver_lap_time - mean_lap_time_of_others
                (same race, same lap)

Driver skill is then estimated by aggregating **model residuals across races (mean residual per driver)**.
```
---

## 📊 Model Performance

Evaluation uses **GroupShuffleSplit (grouped by raceId)** to prevent leakage.

### Model Benchmark

| Model | MAE ↓ | MSE ↓ |
|------|------|------|
| Ridge | **3.448** | 736.74 |
| TensorFlow | 3.669 | **706.71** |
| HistGradientBoosting | 3.932 | 774.68 |

### Baselines

| Baseline | MAE ↓ |
|----------|------|
| Zero baseline | 3.110 |
| Per-driver mean (leakage-prone) | 2.964 |

**Interpretation:**

* The model improves over a naive baseline  
* A leakage-prone baseline still performs better  
* This demonstrates how easily performance can be inflated by leakage  
* Proper validation leads to more realistic (and lower) performance estimates  

---

## 🧪 Example Output (Model-Based Ranking)

Top drivers by model-based skill:

Rubens Barrichello  
Lewis Hamilton  
Jenson Button  
Jacques Villeneuve  
Felipe Massa  
David Coulthard  
Fernando Alonso  
Ralf Schumacher  
Michael Schumacher  
Nico Rosberg  

**Note:**

This ranking is derived from aggregated residual performance and is sensitive to modeling assumptions.  
It should not be interpreted as an absolute measure of driver ability.

---

## ⚠️ Problem Difficulty

Estimating driver skill from race data is inherently noisy:

* strong confounding from car performance  
* race-specific dynamics  
* incomplete observability  

Even simple leakage-prone baselines can outperform properly validated models.

This project explicitly prioritizes:

* correctness of validation  
* robustness of pipeline  
* reproducibility  

over raw metric optimization.

---

## ⚠️ Limitations

* car performance is only indirectly modeled  
* no explicit team/car disentanglement  
* race strategies and external events are not fully captured  
* model performance depends heavily on feature design and normalization assumptions  

This model estimates **relative skill under noisy conditions**, not absolute ability.

---

## 🧱 Pipeline

ingest → preprocess → feature engineering → train → evaluate → predict → report

Artifacts produced:

* features (features.parquet)  
* trained models (ridge.joblib, tf_model.keras)  
* benchmark reports (train_benchmark_summary.csv)  
* driver skill rankings (driver_skill.csv)  

---

## 🧪 Validation Strategy

Key design decision:

GroupShuffleSplit (grouped by raceId)

This prevents:

* leakage between laps from the same race  
* overly optimistic evaluation  

---

## 📈 Experiment Tracking

MLflow tracks:

* model runs  
* metrics (MAE, MSE)  
* hyperparameters  
* model comparisons  

---

## 🛠️ Run locally

```bash
make install
make all
make test

---

## 🧪 CI (GitHub Actions)

CI pipeline runs:

* tests on sample dataset  
* pipeline execution (train + predict)

This ensures:

* reproducibility  
* fast validation  
* pipeline integrity  

---

## 💡 What this demonstrates

* end-to-end ML pipeline design  
* leakage-aware validation (GroupSplit)  
* realistic model benchmarking (with baselines)  
* experiment tracking (MLflow)  
* reproducible workflows (CLI + Makefile + CI)  
* handling of noisy, confounded real-world data  

---

## 🧠 What makes this project different

This project explicitly prioritizes **correct evaluation over metric performance**.

Instead of optimizing for the lowest error:

- it enforces leakage-safe validation  
- it compares against leakage-prone baselines  
- it highlights how misleading naive evaluation can be  

This reflects real-world ML challenges, where validation strategy is often more important than the model itself.

## 📌 Key takeaway

This project is not about predicting race results.

It is about:

designing a system that estimates signal in a noisy, biased environment

and doing it in a reproducible, MLOps-oriented way.
