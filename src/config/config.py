"""
Configuration settings for the UBA ML project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset settings
DATASET_URLS = {
    "CICIDS2017": "https://www.unb.ca/cic/datasets/ids-2017.html",
    "NSL-KDD": "https://www.unb.ca/cic/datasets/nsl.html",
    "UNSW-NB15": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
}

# Model parameters
MODEL_PARAMS = {
    "isolation_forest": {
        "n_estimators": 100,
        "max_samples": "auto",
        "contamination": "auto",
        "random_state": 42
    },
    "one_class_svm": {
        "kernel": "rbf",
        "gamma": "scale",
        "nu": 0.01
    },
    "local_outlier_factor": {
        "n_neighbors": 20,
        "contamination": "auto"
    },
    "autoencoder": {
        "encoding_dim": 32,
        "epochs": 50,
        "batch_size": 256,
        "learning_rate": 0.001
    },
    "lstm": {
        "units": 64,
        "dropout": 0.2,
        "recurrent_dropout": 0.2,
        "epochs": 30,
        "batch_size": 128
    }
}

# Feature engineering settings
FEATURE_ENGINEERING = {
    "time_window": 300,  # 5 minutes in seconds
    "statistical_features": ["mean", "std", "min", "max", "count"],
    "categorical_encoding": "one-hot"
}

# Evaluation settings
EVALUATION = {
    "test_size": 0.2,
    "validation_size": 0.2,
    "random_state": 42,
    "metrics": ["accuracy", "precision", "recall", "f1", "auc", "false_positive_rate"]
}

# Visualization settings
VISUALIZATION = {
    "color_palette": "viridis",
    "figure_size": (12, 8),
    "dpi": 100
} 