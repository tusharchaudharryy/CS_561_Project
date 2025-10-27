import torch

DATA_DIR = './data/'
IMAGE_DIR = './data/images/'
FEATURE_DIR = './features/'
MODEL_DIR = './models/'

PROJECT_TRAIN_CSV = './data/project_train.csv'
PROJECT_TEST_CSV = './data/project_test.csv'

TEXT_MODEL_NAME = 'distilbert-base-uncased'
IMAGE_MODELS_FOR_FEATURES = {
    'efficientnet_b0': 'best_model_efficientnet_b0_lr5e-06.pth',
    'resnet50': 'best_model_resnet50_lr5e-06.pth'
}
IMAGE_SIZE = 224
MAX_TEXT_LENGTH = 256
FEATURE_EXTRACTION_BATCH_SIZE = 64

LGBM_N_SPLITS = 5
LGBM_PARAMS = {
    'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 3576,
    'learning_rate': 0.0141, 'num_leaves': 210, 'max_depth': 10,
    'min_child_samples': 50, 'feature_fraction': 0.657, 'bagging_fraction': 0.402,
    'bagging_freq': 3, 'lambda_l1': 2.11e-05, 'lambda_l2': 0.170,
    'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt',
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_STATE = 42
