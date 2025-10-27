# scripts/04_create_deep_features.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import re

from src.model import MultimodalPricePredictor
from src.utils import extract_ipq
import src.config as config

print("--- Extracting Deep Features ---")
print("!!! This script requires pre-trained PyTorch models to exist.")
print("!!! Ensure '05_train_main_model.py' (or similar) has been run to create .pth files.")

DEVICE = config.DEVICE
IMAGE_DIR = config.IMAGE_DIR
FEATURE_DIR = config.FEATURE_DIR
MODEL_DIR = config.MODEL_DIR
TRAIN_CSV = config.PROJECT_TRAIN_CSV
TEST_CSV = config.PROJECT_TEST_CSV
TEXT_MODEL_NAME = config.TEXT_MODEL_NAME
# Use the *actual* model config used for training (e.g., efficientnet_b0)
IMAGE_MODELS_ARCH = {
    'efficientnet_b0': config.MODEL_SAVE_PATH # Assuming main model uses efficientnet_b0
    # Add other trained model paths here if you train multiple architectures
    # 'resnet50': os.path.join(MODEL_DIR, 'resnet50_model.pth')
}
IMAGE_SIZE = config.IMAGE_SIZE
MAX_TEXT_LENGTH = config.MAX_TEXT_LENGTH
BATCH_SIZE = 64 # Can often use larger batch size for inference

os.makedirs(FEATURE_DIR, exist_ok=True)

# --- Dataset Definition (Copied from src/dataset.py for simplicity in this script) ---
class FeatureDataset(torch.utils.data.Dataset):
    # (Same as in previous extract_features.py example) ...
    def __init__(self, df, tokenizer, image_transform, image_dir):
        self.df = df
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['catalog_content']) if pd.notna(row['catalog_content']) else ''
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_TEXT_LENGTH)
        image_path = os.path.join(self.image_dir, f"{row['sample_id']}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))
        except Exception as e:
            print(f"Warning: Error loading image {image_path}. Using placeholder. Error: {e}", file=sys.stderr)
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'image': self.image_transform(image),
            'sample_id': str(row['sample_id'])
        }
# --- End Dataset Definition ---

def get_deep_features(model, dataloader):
    # (Same as in previous extract_features.py example) ...
    model.eval()
    all_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Deep Features"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            image = batch['image'].to(DEVICE)
            text_output = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = text_output.last_hidden_state[:, 0, :]
            image_features = model.image_model(image)
            combined = torch.cat([text_features, image_features], dim=1).cpu().numpy()
            all_features.append(combined)
    return np.vstack(all_features)


# --- Main Execution ---
if __name__ == "__main__":
    try:
        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)
    except FileNotFoundError as e:
        print(f"Error loading CSV: {e}")
        exit()

    tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)
    image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = FeatureDataset(train_df, tokenizer, image_transform, IMAGE_DIR)
    test_dataset = FeatureDataset(test_df, tokenizer, image_transform, IMAGE_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # Extract and save IPQ (can reuse from baseline or extract again)
    print("Extracting and saving IPQ features...")
    train_ipq = extract_ipq(train_df['catalog_content'])
    test_ipq = extract_ipq(test_df['catalog_content'])
    np.save(os.path.join(FEATURE_DIR, 'train_ipq.npy'), train_ipq)
    np.save(os.path.join(FEATURE_DIR, 'test_ipq.npy'), test_ipq)
    print("IPQ features saved.")

    for model_arch, model_path in IMAGE_MODELS_ARCH.items():
        print(f"\nProcessing features using {model_arch}")
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}. Skipping.")
            continue

        # Instantiate model based on the architecture name used during training
        model = MultimodalPricePredictor(image_model_name=model_arch, text_model_name=TEXT_MODEL_NAME).to(DEVICE)
        try:
            # Load the state dict saved by the training script
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except RuntimeError as e:
             print(f"Error loading model state dict for {model_arch}: {e}")
             print("Ensure the model architecture matches the saved checkpoint.")
             continue

        train_features = get_deep_features(model, train_loader)
        np.save(os.path.join(FEATURE_DIR, f'train_deep_features_{model_arch}.npy'), train_features)
        print(f"Train deep features for {model_arch} saved.")

        test_features = get_deep_features(model, test_loader)
        np.save(os.path.join(FEATURE_DIR, f'test_deep_features_{model_arch}.npy'), test_features)
        print(f"Test deep features for {model_arch} saved.")

    print("\n--- Deep Feature Extraction Complete ---")