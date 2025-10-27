# src/config.py
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- Paths ---
DATA_DIR = './data/'
IMAGE_DIR = './data/images/'
FEATURE_DIR = './features/'
MODEL_DIR = './models/'

PROJECT_TRAIN_CSV = './data/project_train.csv'
PROJECT_TEST_CSV = './data/project_test.csv'

# --- Multimodal Model Config ---
TEXT_MODEL_NAME = 'distilbert-base-uncased'
IMAGE_MODEL_NAME = 'efficientnet_b0'
IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 256

# --- Training Config ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
LEARNING_RATE = 5e-6
EPOCHS = 15 # Max epochs
PATIENCE = 3 # Early stopping patience
MODEL_SAVE_PATH = f'{MODEL_DIR}/main_multimodal_model.pth'

# --- Baseline Model Config ---
BASELINE_LGBM_MODEL_PATH = f'{MODEL_DIR}/baseline_lgbm_model.joblib'
BASELINE_FEATURES_TRAIN = f'{FEATURE_DIR}/train_baseline_features.npz'
BASELINE_FEATURES_TEST = f'{FEATURE_DIR}/test_baseline_features.npz'

# --- Hybrid Model Config ---
HYBRID_LGBM_PARAMS = { # Best params from Optuna (or defaults)
    'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 3500, # Example
    'learning_rate': 0.015, 'num_leaves': 200, 'max_depth': 10,
    'min_child_samples': 50, 'feature_fraction': 0.7, 'bagging_fraction': 0.5,
    'bagging_freq': 3, 'lambda_l1': 1e-05, 'lambda_l2': 0.1,
    'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt',
}
HYBRID_N_SPLITS = 5
HYBRID_MODELS_FOR_FEATURES = { # Models used for feature extraction
    'efficientnet_b0': f'{MODEL_DIR}/efficientnet_b0_feature_extractor.pth', # Needs training
    'resnet50': f'{MODEL_DIR}/resnet50_feature_extractor.pth' # Needs training
}


# --- General ---
RANDOM_STATE = 42
NUM_WORKERS = 4