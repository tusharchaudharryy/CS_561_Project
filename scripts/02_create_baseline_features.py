import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, save_npz
import os
from src.utils import extract_ipq

DATA_DIR = './data/'
FEATURE_DIR = './features/'
TRAIN_CSV = os.path.join(DATA_DIR, 'project_train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'project_test.csv')

os.makedirs(FEATURE_DIR, exist_ok=True)

print(" Creating Baseline Features (TF-IDF + IPQ) ")

try:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
except FileNotFoundError as e:
    print(f"Error loading CSV: {e}")
    exit()

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

full_df = pd.concat([train_df.drop(columns=['price']), test_df], axis=0, ignore_index=True)

print("Calculating TF-IDF features")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, stop_words='english')
text_features = tfidf.fit_transform(full_df['catalog_content'].fillna(''))

print("Extracting IPQ features")
ipq_features = extract_ipq(full_df['catalog_content'])

print("Combining features")
X_full = hstack([text_features, ipq_features]).tocsr()

X_train_baseline = X_full[:len(train_df)]
X_test_baseline = X_full[len(train_df):]

print(f"Train feature shape: {X_train_baseline.shape}")
print(f"Test feature shape: {X_test_baseline.shape}")

train_feat_path = os.path.join(FEATURE_DIR, 'train_baseline_features.npz')
test_feat_path = os.path.join(FEATURE_DIR, 'test_baseline_features.npz')
save_npz(train_feat_path, X_train_baseline)
save_npz(test_feat_path, X_test_baseline)

print(f"Baseline training features saved to: {train_feat_path}")
print(f"Baseline test features saved to: {test_feat_path}")
print(" Baseline Feature Creation Complete ")
