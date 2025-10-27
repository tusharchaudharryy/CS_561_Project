import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz
import joblib
import os
from src.utils import smape

DATA_DIR = './data/'
FEATURE_DIR = './features/'
MODEL_DIR = './models/'
TRAIN_CSV = os.path.join(DATA_DIR, 'project_train.csv')
TRAIN_FEATURES = os.path.join(FEATURE_DIR, 'train_baseline_features.npz')
MODEL_PATH = os.path.join(MODEL_DIR, 'baseline_lgbm_model.joblib')

os.makedirs(MODEL_DIR, exist_ok=True)

print(" Training Baseline LightGBM Model ")

try:
    X = load_npz(TRAIN_FEATURES)
    train_df = pd.read_csv(TRAIN_CSV)
    y = np.log1p(train_df['price'])
    y_true_orig = train_df['price'].values
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()

print(f"Loaded features shape: {X.shape}")

X_train, X_val, y_train, y_val, _, y_val_true_orig = train_test_split(
    X, y, y_true_orig, test_size=0.15, random_state=42
)

params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

print("Training model")
model = lgb.LGBMRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(100, verbose=True)]
)

val_preds_log = model.predict(X_val)
val_preds = np.expm1(val_preds_log)
validation_smape = smape(y_val_true_orig, val_preds)
print(f"\n Baseline Model Validation SMAPE: {validation_smape:.4f} ")

joblib.dump(model, MODEL_PATH)
print(f"Baseline model saved to: {MODEL_PATH}")
print(" Baseline Model Training Complete ")
