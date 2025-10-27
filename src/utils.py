import numpy as np
import os
import pandas as pd
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from functools import partial
import urllib.request
import re

def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(ratio) * 100

def _download_single_image(image_link, savefolder):
    if isinstance(image_link, str):
        try:
            filename = Path(image_link).name
            filename = filename.split('?')[0]
        except Exception:
            print(f"Warning: Could not extract filename from {image_link}")
            return

        image_save_path = os.path.join(savefolder, filename)
        if not os.path.exists(image_save_path):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)
            except Exception as ex:
                print(f'Warning: Not able to download - {image_link}\n{ex}')
    return

def download_images_from_df(df, download_folder, num_workers=16):
    if 'image_link' not in df.columns:
        print("Error: 'image_link' column not found in DataFrame.")
        return

    os.makedirs(download_folder, exist_ok=True)
    image_links = df['image_link'].dropna().tolist()
    print(f"Starting download of {len(image_links)} images to {download_folder}...")

    download_func = partial(_download_single_image, savefolder=download_folder)

    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap(download_func, image_links), total=len(image_links), desc="Downloading Images"))

    print("Image download process complete.")

def extract_ipq(text_series):
    if not isinstance(text_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    ipq = text_series.str.extract(r'Item Pack Quantity:\s*(\d+)', expand=False)
    ipq = pd.to_numeric(ipq, errors='coerce').fillna(1.0)
    return ipq.values.reshape(-1, 1)
