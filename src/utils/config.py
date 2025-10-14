import os
from pathlib import Path

# ============ PATHS ============
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "gtzan" / "genres_original"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_CSV_DIR = PROCESSED_DATA_DIR / "features_mfcc.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for dir_path in [PROCESSED_DATA_DIR, MODELS_DIR / "sklearn", MODELS_DIR / "scratch", 
                 RESULTS_DIR / "figures", RESULTS_DIR / "metrics"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============ AUDIO PROCESSING ============
SAMPLE_RATE = 22050
DURATION = 30  # seconds
N_MFCC = 20
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# ============ DATA SPLIT ============
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2  # From training set
RANDOM_STATE = 42

# ============ MODEL HYPERPARAMETERS ============
# These are starting points - we'll tune them later!

# KNN
KNN_NEIGHBORS = [3, 5, 7, 9, 11]
KNN_WEIGHTS = ['uniform', 'distance']

# Logistic Regression
LR_C = [0.1, 1, 10, 100]
LR_SOLVER = ['liblinear', 'lbfgs']
LR_MAX_ITER = 1000

# Decision Tree
DT_MAX_DEPTH = [10, 20, 30, None]
DT_MIN_SAMPLES_SPLIT = [2, 5, 10]
DT_MIN_SAMPLES_LEAF = [1, 2, 4]

# Random Forest
RF_N_ESTIMATORS = [100, 200, 300]
RF_MAX_DEPTH = [10, 20, 30]
RF_MIN_SAMPLES_SPLIT = [2, 5]

# SVM
SVM_C = [1, 10, 100]
SVM_GAMMA = ['scale', 0.01, 0.001]
SVM_KERNEL = ['rbf']

# Gradient Boosting
GB_N_ESTIMATORS = [100, 200]
GB_LEARNING_RATE = [0.05, 0.1, 0.2]
GB_MAX_DEPTH = [3, 5, 7]

# Naive Bayes
NB_VAR_SMOOTHING = [1e-9, 1e-8, 1e-7]

# ANN (Neural Network)
ANN_HIDDEN_LAYERS = [64, 32]
ANN_LEARNING_RATE = 0.001
ANN_EPOCHS = 100
ANN_BATCH_SIZE = 32
ANN_ACTIVATION = 'relu'

# ============ GENRE LABELS ============
GENRES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

# ============ EVALUATION ============
CV_FOLDS = 5
METRICS = ['accuracy', 'precision', 'recall', 'f1']

# ============ VISUALIZATION ============
FIGSIZE = (12, 8)
DPI = 300
CMAP = 'viridis'

# ============ LOGGING ============
VERBOSE = True
SAVE_MODELS = True