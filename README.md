# 🏎️ F1 Driver Skill Modeling (MLOps Pipeline)

End-to-end MLOps pipeline for estimating **driver performance beyond raw race results**, using race-relative features, grouped validation and residual analysis.

---

## ⚡ What this project does

This system:

- builds **lap-level and race-relative features** from historical F1 data
- trains and benchmarks multiple models (Ridge, HistGradientBoosting, TensorFlow)
- evaluates performance using **GroupShuffleSplit grouped by `raceId`**
- compares models against a zero baseline on the same validation races
- estimates driver performance via **residual analysis**
- tracks experiments with MLflow
- runs the full pipeline reproducibly via CLI + Makefile + CI

---

## 🧠 What makes this project different

This project explicitly prioritizes **correct evaluation over headline metric performance**.

Instead of optimizing only for error:

- it keeps laps from the same race on one side of the train/validation split
- it compares models against an explicit baseline evaluated on the same validation data
- it separates formal validation results from diagnostic outputs
- it keeps limitations visible instead of presenting every trained model as an improvement

This reflects a common real-world ML challenge, where validation strategy can matter as much as the model itself.

---

## 🎯 Core idea

Raw race results do not directly represent driver skill.

Performance is influenced by:

- car performance
- track conditions
- race dynamics
- strategy and incidents

To reduce some of these effects, the system models relative performance within each race:

```text
relative_pace = driver_lap_time - mean_lap_time_of_others
                (same race, same lap)
```

The model learns expected relative pace from engineered race, driver-form and team-form features.

Driver performance is then explored by aggregating **model residuals across races**.

---

## 📊 Model Performance

Evaluation uses **GroupShuffleSplit grouped by `raceId`** so laps from the same race stay on one side of the train/validation split.

### Model Benchmark

| Model | Validation MAE ↓ | Validation MSE ↓ |
|---|---:|---:|
| Ridge | **3.448** | 736.74 |
| Zero baseline | 3.572 | 740.64 |
| TensorFlow | 3.701 | **701.44** |
| HistGradientBoosting | 3.932 | 774.68 |

**Interpretation:**

- Ridge achieved the best validation MAE
- Ridge beat the zero baseline, but only by a modest margin
- TensorFlow achieved lower MSE but worse MAE
- HistGradientBoosting did not outperform the zero baseline on MAE
- The engineered features contain some predictive signal, but the overall problem remains difficult and noisy

---

## 🧪 Example Output (Model-Based Ranking)

Top drivers by current model-based ranking:

```text
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
```

**Note:**

This ranking is derived from aggregated model residuals and is sensitive to modelling assumptions.

The current ranking uses residuals generated across the full feature dataset, so it should be treated as exploratory rather than as a fully out-of-sample estimate of driver ability.

---

## ⚠️ Problem Difficulty

Estimating driver skill from race data is inherently noisy:

- strong confounding from car performance
- race-specific dynamics
- strategy and incidents
- incomplete information about machinery performance

Even with race-grouped validation, the model only improves modestly over the zero baseline.

This project therefore prioritizes:

- correctness of validation
- explicit baseline comparison
- robustness of pipeline
- reproducibility

over raw metric optimization.

---

## ⚠️ Limitations

- car performance is only indirectly modeled
- no explicit team/car disentanglement
- race strategies and external events are not fully captured
- model performance depends heavily on feature design and assumptions
- the current driver ranking uses residuals across the full feature dataset
- the grouped validation set is also used for model selection and TensorFlow early stopping
- there is currently no separate untouched final test set

This model estimates **relative performance under noisy conditions**, not absolute driver ability.

---

## 🧱 Pipeline

```text
ingest → preprocess → feature engineering → train → evaluate → predict → report
```

Artifacts produced:

- features (`features.parquet`)
- trained models (`ridge.joblib`, `tf_model.keras`)
- Ridge hyperparameter sweep
- benchmark reports (`train_benchmark_summary.csv`)
- driver rankings (`driver_skill.csv`)

---

## 🧪 Validation Strategy

Key design decision:

```text
GroupShuffleSplit (grouped by raceId)
```

This prevents:

- laps from the same race appearing in both training and validation data
- overly optimistic results from a random lap-level split

The validation split is currently used for:

- Ridge alpha selection
- model comparison
- TensorFlow early stopping
- validation metric reporting

It should therefore be treated as a **development validation set**, not as a completely untouched final test set.

---

## 📈 Experiment Tracking

MLflow tracks:

- model runs
- metrics (MAE, MSE)
- hyperparameters
- model comparisons
- train and validation sizes
- generated artifacts

---

## 🛠️ Run locally

```bash
make install
make all
make test
```

---

## 🧪 CI (GitHub Actions)

CI runs:

- a sample-data contract test
- training on a small fixture dataset
- prediction on a small fixture dataset

This helps ensure:

- reproducibility
- testability
- pipeline integrity

---

## 💡 What this demonstrates

- end-to-end ML pipeline design
- race-grouped validation
- explicit baseline comparison
- model benchmarking
- experiment tracking with MLflow
- reproducible workflows (CLI + Makefile + CI)
- residual-based driver analysis
- handling noisy, confounded real-world data
- documenting limitations alongside results

---

## 📌 Key takeaway

This project is not about producing a definitive ranking of Formula 1 drivers.

It is about:

> designing a reproducible system that tries to extract useful signal from a noisy and heavily confounded environment

and doing it in a **reproducible, MLOps-oriented way**.