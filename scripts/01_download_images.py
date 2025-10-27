# scripts/01_download_images.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.utils import download_images_from_df
import src.config as config

print("--- Downloading Images ---")

# Load dataframes
try:
    train_df = pd.read_csv(config.PROJECT_TRAIN_CSV)
    test_df = pd.read_csv(config.PROJECT_TEST_CSV)
    # You might want original test images later for prediction, include if needed
    # original_test_df = pd.read_csv(os.path.join(config.DATA_DIR, 'test.csv'))
except FileNotFoundError as e:
    print(f"Error loading CSV file: {e}")
    print("Ensure '00_split_data.py' has been run.")
    exit()

# Combine unique image links and IDs
all_data_df = pd.concat([
    train_df[['sample_id', 'image_link']],
    test_df[['sample_id', 'image_link']],
    # original_test_df[['sample_id', 'image_link']] # Optional
], ignore_index=True).drop_duplicates(subset=['sample_id']) # Use sample_id to name files

# Download images
download_images_from_df(all_data_df, config.IMAGE_DIR)

print("--- Image Download Complete ---")