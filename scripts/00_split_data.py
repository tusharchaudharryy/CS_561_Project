import pandas as pd
from sklearn.model_selection import train_test_split
import os

ORIGINAL_TRAIN_PATH = './data/train.csv'
PROJECT_TRAIN_PATH = './data/project_train.csv'
PROJECT_TEST_PATH = './data/project_test.csv'
TEST_SPLIT_SIZE = 0.2 
RANDOM_STATE = 42

print(" Splitting Data ")

if not os.path.exists(ORIGINAL_TRAIN_PATH):
    print(f"Error: Original train file not found at {ORIGINAL_TRAIN_PATH}")
else:
    original_train_df = pd.read_csv(ORIGINAL_TRAIN_PATH)
    print(f"Loaded original training data: {len(original_train_df)} rows")

    project_train_df, project_test_df = train_test_split(
        original_train_df,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"Splitting data: {len(project_train_df)} train rows, {len(project_test_df)} test rows")

    project_train_df.to_csv(PROJECT_TRAIN_PATH, index=False)
    project_test_df.to_csv(PROJECT_TEST_PATH, index=False)

    print(f"New training set saved to: {PROJECT_TRAIN_PATH}")
    print(f"New test set saved to: {PROJECT_TEST_PATH}")
    print(" Data Splitting Complete ")