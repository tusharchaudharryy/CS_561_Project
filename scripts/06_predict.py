import numpy as np
import pandas as pd
import joblib
import gc
import os
from src.utils import smape
import src.config as config

FEATURE_DIR = config.FEATURE_DIR
MODEL_DIR = config.MODEL_DIR
TEST_CSV = config.PROJECT_TEST_CSV
MODELS_TO_USE = list(config.IMAGE_MODELS_FOR_FEATURES.keys())
N_SPLITS = config.LGBM_N_SPLITS
PREDICTIONS_FILE = 'project_predictions.csv'

print(" Generating Predictions and Evaluating ")

print("Loading test features...")
try:
    test_ipq = np.load(os.path.join(FEATURE_DIR, 'test_ipq.npy'))
    test_deep_features = [np.load(os.path.join(FEATURE_DIR, f'test_deep_features_{m}.npy')) for m in MODELS_TO_USE]
    X_test = np.hstack(test_deep_features + [test_ipq])
    test_df = pd.read_csv(TEST_CSV)
    y_test_true = test_df['price'].values
except FileNotFoundError as e:
    print(f"Error loading features or test CSV: {e}")
    print("Ensure '04_create_deep_features.py' was run successfully.")
    exit()

print(f"Test feature matrix shape: {X_test.shape}")

del test_deep_features, test_ipq
gc.collect()

print(f"Loading {N_SPLITS} K-Fold models and predicting")
test_preds_sum = np.zeros(len(X_test))
models_loaded = 0

for fold in range(N_SPLITS):
    model_filename = os.path.join(MODEL_DIR, f'hybrid_lgbm_model_fold_{fold+1}.joblib')
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
        test_preds_log = model.predict(X_test)
        test_preds_sum += np.expm1(test_preds_log)
        models_loaded += 1
        print(f"Loaded and predicted with fold {fold+1} model.")
    else:
        print(f"Warning: Model file not found for fold {fold+1}. Skipping.")

if models_loaded == 0:
    print("Error: No models were loaded. Cannot generate predictions.")
    exit()

final_predictions = test_preds_sum / models_loaded
final_predictions = np.clip(final_predictions, 0, None)

print("Averaged predictions from K-Fold models.")

final_test_smape = smape(y_test_true, final_predictions)
print(f"\n Final Project Test SMAPE: {final_test_smape:.4f} ")

submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': final_predictions})
submission_df.to_csv(PREDICTIONS_FILE, index=False)
print(f"Predictions saved to: {PREDICTIONS_FILE}")

print(" Prediction and Evaluation Complete ")
