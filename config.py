# =============================================================================
# 1) ŚCIEŻKI I LOGI
# =============================================================================

from pathlib import Path

# Root projektu = folder, w którym leży config.py
PROJECT_ROOT = Path(__file__).resolve().parent

# Folder z CSV
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Artefakty pipeline
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ART_FEATURES = PROJECT_ROOT / "artifacts" / "features" / "features.parquet"

ART_MODELS = PROJECT_ROOT / "artifacts" / "models"
ART_REPORTS = PROJECT_ROOT / "artifacts" / "reports"

TF_MODEL_FILE = PROJECT_ROOT / "artifacts" / "models" / "tf_model.keras"
RIDGE_MODEL_FILE = PROJECT_ROOT / "artifacts" / "models" / "ridge.joblib"

SKILL_REPORT = PROJECT_ROOT / "artifacts" / "reports" / "driver_skill.csv"
PREDICTIONS_OUT = PROJECT_ROOT / "artifacts" / "reports" / "predictions_preview.csv"

MODEL_BACKEND = "ridge"  # "ridge" | "tensorflow"

# Outputy / logi
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOG_LEVEL = "INFO"  # "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL"
LOG_FILE = PROJECT_ROOT / "outputs" / "run.log"
LOG_TO_CONSOLE = True


# =============================================================================
# 2) DANE / CLEANING / FEATURE ENGINEERING
# =============================================================================

USE_PITS = True
STINT_GAP_SECONDS = 5.0
ROLLING_WINDOW_DRIVER = 5
ROLLING_WINDOW_TEAM = 5


# =============================================================================
# 3) MODELING – CANONICAL KOLUMNY
# =============================================================================

MODEL_TARGET = "relative_pace"  # "seconds" | "relative_pace"
MODEL_VARIANT = "skill"  # "forecast" | "skill"

MODEL_MIN_LAPS = 1000

TEST_SIZE = 0.20
RANDOM_SEED = 42

MISSING_NUM_STRATEGY = "median"  # "median" | "mean"
FILL_NAN_NUMERIC = 0.0

MODEL_FEATURES_BASE = [
    "track_evolution_index",
    "driver_form_avg",
    "driver_form_std",
    "team_form_avg",
    "team_form_std",
    "position_prev",
    "lap",
    "max_lap",
    "round",
    "year",
]

MODEL_FEATURES_FORECAST_EXTRA = [
    "lap_time_prev",
    "lap_time_diff",
]

EXPECTED_FEATURE_COLUMNS = [
    "raceId",
    "driverId",
    "year",
    "round",
    "circuitId",
    "lap",
    "max_lap",
    "seconds",
    "track_evolution_index",
    "lap_time_prev",
    "lap_time_diff",
    "driver_form_avg",
    "driver_form_std",
    "team_form_avg",
    "team_form_std",
    "relative_pace",
    "position_prev",
]


def get_model_features() -> list[str]:
    """Zwraca listę cech zależnie od MODEL_VARIANT."""
    feats = list(MODEL_FEATURES_BASE)
    if MODEL_VARIANT == "forecast":
        return list(MODEL_FEATURES_FORECAST_EXTRA) + feats
    if MODEL_VARIANT == "skill":
        return feats
    raise ValueError(f"Unknown MODEL_VARIANT={MODEL_VARIANT}")


# =============================================================================
# 3b) TF – parametry treningu
# =============================================================================

TF_LR = 1e-3
TF_EPOCHS = 50
TF_BATCH_SIZE = 2048
TF_EARLYSTOP_PATIENCE = 5


# =============================================================================
# 4) PARAMETRY POD SKILL / IDEAL DRIVER
# =============================================================================

RACE_TAX_TRIM = 0.10
DSS_QUANTILE = 0.10
IDEAL_PERCENTILE = 0.95
