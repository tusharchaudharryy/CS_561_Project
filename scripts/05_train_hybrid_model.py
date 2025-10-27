import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import joblib
import gc
import os

# Import from project structure
from src.utils import smape
import src.config as config

# Configuration from src/config.py
FEATURE_DIR = config.FEATURE_DIR
MODEL_DIR = config.MODEL_DIR
TRAIN_CSV = config.PROJECT_TRAIN_CSV
MODELS_TO_USE = list(config.IMAGE_MODELS_FOR_FEATURES.keys()) # Get model names from config
N_SPLITS = config.LGBM_N_SPLITS
PARAMS = config.LGBM_PARAMS
RANDOM_STATE = config.RANDOM_STATE

os.makedirs(MODEL_DIR, exist_ok=True)

print("--- Training Hybrid K-Fold LightGBM Model ---")

# 1. Load all pre-extracted features
print("Loading features...")
try:
    train_ipq = np.load(os.path.join(FEATURE_DIR, 'train_ipq.npy'))
    train_deep_features = [np.load(os.path.join(FEATURE_DIR, f'train_deep_features_{m}.npy')) for m in MODELS_TO_USE]
    X = np.hstack(train_deep_features + [train_ipq])

    train_df = pd.read_csv(TRAIN_CSV)
    y = np.log1p(train_df['price'])
    y_true_orig = train_df['price'].values
except FileNotFoundError as e:
    print(f"Error loading features or train CSV: {e}")
    print("Ensure '04_create_deep_features.py' was run successfully.")
    exit()

del train_deep_features, train_ipq, train_df # Free up memory
gc.collect()

print(f"Full training feature matrix shape: {X.shape}")

# 2. Set up K-Fold Cross-Validation
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(X))
trained_models = [] # To save models from each fold

print(f"\nStarting training with {N_SPLITS}-Fold Cross-Validation...")

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='mae',
              callbacks=[lgb.early_stopping(100, verbose=100)]) # Print progress less often

    # Store validation predictions for OOF score
    val_preds_log = model.predict(X_val)
    oof_preds[val_index] = np.expm1(val_preds_log)

    # Save the trained model for this fold
    model_filename = os.path.join(MODEL_DIR, f'hybrid_lgbm_model_fold_{fold+1}.joblib')
    joblib.dump(model, model_filename)
    trained_models.append(model) # Keep in memory if RAM allows, or just save path
    print(f"Fold {fold+1} model saved to {model_filename}")

    del X_train, X_val, y_train, y_val
    gc.collect()

# 3. Calculate Final OOF Score
final_oof_smape = smape(y_true_orig, oof_preds)
print(f"\n--- 📈 Cross-Validation Finished ---")
print(f"Final OOF (Out-of-Fold) SMAPE: {final_oof_smape:.4f}")
print("-----------------------------------")

# Optional: Train final model on all data (if not averaging fold predictions later)
# print("\nTraining final model on all data...")
# final_model = lgb.LGBMRegressor(**PARAMS)
# final_model.fit(X, y)
# final_model_path = os.path.join(MODEL_DIR, 'hybrid_lgbm_model_final.joblib')
# joblib.dump(final_model, final_model_path)
# print(f"Final model saved to: {final_model_path}")

print("--- Hybrid K-Fold Model Training Complete ---")