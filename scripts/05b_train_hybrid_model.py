# scripts/05b_train_hybrid_model.py
import sys
import os
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import joblib
import gc

# Import from project structure
from src.utils import smape
import src.config as config

# --- Configuration from src/config.py ---
FEATURE_DIR = config.FEATURE_DIR
MODEL_DIR = config.MODEL_DIR
TRAIN_CSV = config.PROJECT_TRAIN_CSV
# Use model architecture names defined in config to find feature files
MODELS_FOR_FEATURES = list(config.HYBRID_MODELS_FOR_FEATURES.keys())
N_SPLITS = config.HYBRID_N_SPLITS
PARAMS = config.HYBRID_LGBM_PARAMS # Use the defined hybrid params
RANDOM_STATE = config.RANDOM_STATE

os.makedirs(MODEL_DIR, exist_ok=True) # Ensure model directory exists

print("--- Training Hybrid K-Fold LightGBM Model ---")

# 1. Load all pre-extracted features
print("Loading features...")
try:
    train_ipq = np.load(os.path.join(FEATURE_DIR, 'train_ipq.npy'))
    # Load deep features based on the model names in config
    train_deep_features = [np.load(os.path.join(FEATURE_DIR, f'train_deep_features_{m}.npy'))
                           for m in MODELS_FOR_FEATURES]
    X = np.hstack(train_deep_features + [train_ipq]) # Combine deep features + IPQ

    train_df = pd.read_csv(TRAIN_CSV)
    y = np.log1p(train_df['price']) # Target: log-transformed price
    y_true_orig = train_df['price'].values # Original price for SMAPE calculation
except FileNotFoundError as e:
    print(f"Error loading features or train CSV: {e}")
    print("Ensure '04_create_deep_features.py' (or equivalent) was run successfully.")
    exit()

# Clean up memory
del train_deep_features, train_ipq, train_df
gc.collect()

print(f"Full training feature matrix shape: {X.shape}")

# 2. Set up K-Fold Cross-Validation
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(X)) # To store out-of-fold predictions
trained_models = [] # To store trained models for prediction later

print(f"\nStarting training with {N_SPLITS}-Fold Cross-Validation...")

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Initialize and train LightGBM model for this fold
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='mae', # Use MAE on log scale as proxy
              callbacks=[lgb.early_stopping(100, verbose=100)]) # Stop if no improvement

    # Store validation predictions (on original scale) for OOF score
    val_preds_log = model.predict(X_val)
    oof_preds[val_index] = np.expm1(val_preds_log) # Convert back from log scale

    # Save the trained model for this fold
    model_filename = os.path.join(MODEL_DIR, f'hybrid_lgbm_model_fold_{fold+1}.joblib')
    joblib.dump(model, model_filename)
    trained_models.append(model_filename) # Store the path
    print(f"Fold {fold+1} model saved to {model_filename}")

    # Clean up memory for next fold
    del X_train, X_val, y_train, y_val, model
    gc.collect()

# 3. Calculate Final OOF Score
# Clip OOF predictions to avoid issues with SMAPE calculation if needed
oof_preds = np.clip(oof_preds, 0, None)
final_oof_smape = smape(y_true_orig, oof_preds)
print(f"\n--- 📈 Cross-Validation Finished ---")
print(f"Final OOF (Out-of-Fold) SMAPE: {final_oof_smape:.4f}")
print("-----------------------------------")

# Note: This script focuses on training and saving the K-Fold models.
# The `06_evaluate_models.py` script will handle loading these models
# and generating the final predictions on the project_test set.

print("--- Hybrid K-Fold Model Training Complete ---")
