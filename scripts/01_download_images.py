import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import download_images_from_df

DATA_DIR = './data/'
IMAGE_SAVE_DIR = os.path.join(DATA_DIR, 'images')
TRAIN_CSV = os.path.join(DATA_DIR, 'project_train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'project_test.csv')
ORIGINAL_TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

print(" Downloading Images ")

try:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
except FileNotFoundError as e:
    print(f"Error loading CSV file: {e}")
    exit()

all_links_df = pd.concat([
    train_df[['image_link']],
    test_df[['image_link']]
], ignore_index=True).drop_duplicates()

download_images_from_df(all_links_df, IMAGE_SAVE_DIR)

print(" Image Download Complete ")
