# =============================================================================
# 1) ŚCIEŻKI I LOGI
# =============================================================================

# Folder z CSV
DATA_DIR = "DataBase"

# Artefakty pipeline
ARTIFACTS_DIR = "artifacts"
ART_FEATURES  = "artifacts/features.parquet"

ART_MODELS    = "artifacts/models"
ART_REPORTS   = "artifacts/reports"

# Konkretne pliki wyjściowe (żeby train.py i predict.py nie zgadywały)
TF_MODEL_FILE   = "artifacts/models/tf_model.keras"
SKILL_REPORT    = "artifacts/reports/driver_skill.csv"
PREDICTIONS_OUT = "artifacts/reports/predictions_preview.csv"  # opcjonalnie: do debug / wglądu

# Outputy / logi
OUTPUTS_DIR = "outputs"
LOG_LEVEL = "INFO"                 # "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL"
LOG_FILE = "outputs/run.log"
LOG_TO_CONSOLE = True


# =============================================================================
# 2) DANE / CLEANING / FEATURE ENGINEERING
# =============================================================================

USE_PITS = True
STINT_GAP_SECONDS = 5.0
ROLLING_WINDOW_DRIVER = 5
ROLLING_WINDOW_TEAM = 5


# =============================================================================
# 3) MODELING – CANONICAL KOLUMNY (zgodne z artifacts/features.parquet)
# =============================================================================

# Targety dostępne w obecnym features.parquet
MODEL_TARGET = "relative_pace"           # "seconds" | "relative_pace"

# Variant:
# - forecast: wolno użyć lap_time_prev / lap_time_diff (predykcja)
# - skill:    bez prev/diff (residual lepiej niesie skill)
MODEL_VARIANT = "skill"            # "forecast" | "skill"

# Ranking: minimalna liczba okrążeń na kierowcę
MODEL_MIN_LAPS = 1000

# Split / seed
TEST_SIZE = 0.20
RANDOM_SEED = 42

# Prosta imputacja (startowa)
MISSING_NUM_STRATEGY = "median"    # "median" | "mean"
FILL_NAN_NUMERIC = 0.0

# Feature sets (canonical)
MODEL_FEATURES_BASE = [
    "track_evolution_index",
    "driver_form_avg",
    "driver_form_std",
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
# 3b) TF – parametry treningu (żeby nie były zakopane w train.py)
# =============================================================================

TF_LR = 1e-3
TF_EPOCHS = 50
TF_BATCH_SIZE = 2048
TF_EARLYSTOP_PATIENCE = 5


# =============================================================================
# 4) PARAMETRY POD SKILL / IDEAL DRIVER (na później)
# =============================================================================

RACE_TAX_TRIM = 0.10
DSS_QUANTILE = 0.10
IDEAL_PERCENTILE = 0.95