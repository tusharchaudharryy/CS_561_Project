import numpy as np
import os
import pandas as pd
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from functools import partial
import urllib.request
import re

# --- Evaluation Metric ---
def smape(y_true, y_pred):
    """Calculates Symmetric Mean Absolute Percentage Error (SMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100

# --- Image Downloading ---
def _download_single_image(image_link, savefolder):
    """Downloads a single image if it doesn't exist."""
    if isinstance(image_link, str):
        # Extract filename robustly, handle potential issues
        try:
            filename = Path(image_link).name
            # Further sanitize filename if needed (e.g., remove query params)
            filename = filename.split('?')[0]
        except Exception:
            print(f"Warning: Could not extract filename from {image_link}")
            return # Skip if filename extraction fails

        image_save_path = os.path.join(savefolder, filename)
        if not os.path.exists(image_save_path):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)
            except Exception as ex:
                print(f'Warning: Not able to download - {image_link}\n{ex}')
    return

def download_images_from_df(df, download_folder, num_workers=16):
    """Downloads images specified in a DataFrame column."""
    if 'image_link' not in df.columns:
        print("Error: 'image_link' column not found in DataFrame.")
        return

    os.makedirs(download_folder, exist_ok=True)
    image_links = df['image_link'].dropna().tolist()

    print(f"Starting download of {len(image_links)} images to {download_folder}...")

    # Use partial to pass the download_folder argument to the worker function
    download_func = partial(_download_single_image, savefolder=download_folder)

    # Use multiprocessing Pool for parallel downloads
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap(download_func, image_links), total=len(image_links), desc="Downloading Images"))

    print("Image download process complete.")

# --- Feature Extraction Helpers ---
def extract_ipq(text_series):
    """Extracts Item Pack Quantity from a pandas Series of text."""
    if not isinstance(text_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    # Extract digits after 'Item Pack Quantity:', default to 1.0 if not found/invalid
    ipq = text_series.str.extract(r'Item Pack Quantity:\s*(\d+)', expand=False)
    ipq = pd.to_numeric(ipq, errors='coerce').fillna(1.0)
    return ipq.values.reshape(-1, 1) # Reshape for hstack