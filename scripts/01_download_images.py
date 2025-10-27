import pandas as pd
from src.utils import download_images_from_df
import os

# Configuration
DATA_DIR = './data/'
IMAGE_SAVE_DIR = os.path.join(DATA_DIR, 'images')
TRAIN_CSV = os.path.join(DATA_DIR, 'project_train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'project_test.csv')
ORIGINAL_TEST_CSV = os.path.join(DATA_DIR, 'test.csv') # Optional: if you need original test images

print("--- Downloading Images ---")

# Load dataframes
try:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    # original_test_df = pd.read_csv(ORIGINAL_TEST_CSV) # Optional
except FileNotFoundError as e:
    print(f"Error loading CSV file: {e}")
    exit()

# Combine unique image links to avoid duplicate downloads
all_links_df = pd.concat([
    train_df[['image_link']],
    test_df[['image_link']],
    # original_test_df[['image_link']] # Optional
], ignore_index=True).drop_duplicates()

# Download images
download_images_from_df(all_links_df, IMAGE_SAVE_DIR)

print("--- Image Download Complete ---")