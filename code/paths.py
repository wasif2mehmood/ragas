"""Path configurations for the project."""
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
CODE_DIR = PROJECT_ROOT / "code"
DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = CODE_DIR / "metrics"

# Data files
GOLDEN_DATASET_CSV = DATA_DIR / "golden_dataset_with_references.csv"
GOLDEN_DATASET_JSON = DATA_DIR / "golden_dataset.json"
TEST_SET_CSV = DATA_DIR / "test_set.csv"
EVALUATION_RESULTS_CSV = DATA_DIR / "evaluation_results.csv"
COMPLETE_EVALUATION_RESULTS_CSV = DATA_DIR / "complete_evaluation_results.csv"

# Convert to strings for compatibility
GOLDEN_DATASET_CSV_STR = str(GOLDEN_DATASET_CSV)
GOLDEN_DATASET_JSON_STR = str(GOLDEN_DATASET_JSON)
TEST_SET_CSV_STR = str(TEST_SET_CSV)
EVALUATION_RESULTS_CSV_STR = str(EVALUATION_RESULTS_CSV)
COMPLETE_EVALUATION_RESULTS_CSV_STR = str(COMPLETE_EVALUATION_RESULTS_CSV)