import pandas as pd
import sys
import os

sys.path.append(os.path.abspath('src'))

from competition_utils import download_images

IMAGE_SAVE_DIR = './data/images/'
TRAIN_CSV_PATH = './data/train.csv'
TEST_CSV_PATH = './data/test.csv'

if __name__ == "__main__":
    print(" Starting Image Download ")
    
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    print(f"Downloading {len(train_df)} training images")
    train_image_links = train_df['image_link'].tolist()
    download_images(train_image_links, IMAGE_SAVE_DIR)
    
    print(f"\nDownloading {len(test_df)} testing images")
    test_image_links = test_df['image_link'].tolist()
    download_images(test_image_links, IMAGE_SAVE_DIR)
    
    print("\n Image Download Complete ")