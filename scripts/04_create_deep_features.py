import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from torchvision import transforms
from tqdm import tqdm
import os
import re
from PIL import Image

# Import from project structure
from src.model import MultimodalPricePredictor
from src.utils import extract_ipq
import src.config as config

# Configuration from src/config.py
DEVICE = config.DEVICE
IMAGE_DIR = config.IMAGE_DIR
FEATURE_DIR = config.FEATURE_DIR
MODEL_DIR = config.MODEL_DIR
TRAIN_CSV = config.PROJECT_TRAIN_CSV
TEST_CSV = config.PROJECT_TEST_CSV
TEXT_MODEL_NAME = config.TEXT_MODEL_NAME
IMAGE_MODELS = config.IMAGE_MODELS_FOR_FEATURES
IMAGE_SIZE = config.IMAGE_SIZE
MAX_TEXT_LENGTH = config.MAX_TEXT_LENGTH
BATCH_SIZE = config.FEATURE_EXTRACTION_BATCH_SIZE

os.makedirs(FEATURE_DIR, exist_ok=True)

# Custom Dataset for feature extraction
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, image_transform, image_dir):
        self.df = df
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['catalog_content']
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_TEXT_LENGTH)
        image_path = os.path.join(self.image_dir, f"{row['sample_id']}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'image': self.image_transform(image),
            'sample_id': row['sample_id'] # Keep ID if needed later, though not used in extraction here
        }

def get_deep_features(model, dataloader):
    """Extracts combined text and image features before the regressor head."""
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
    print("--- Extracting Deep Features ---")
    try:
        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)
    except FileNotFoundError as e:
        print(f"Error loading CSV: {e}")
        exit()

    tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)
    image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = FeatureDataset(train_df, tokenizer, image_transform, IMAGE_DIR)
    test_dataset = FeatureDataset(test_df, tokenizer, image_transform, IMAGE_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Extract and save IPQ (do this once)
    print("Extracting and saving IPQ features...")
    train_ipq = extract_ipq(train_df['catalog_content'])
    test_ipq = extract_ipq(test_df['catalog_content'])
    np.save(os.path.join(FEATURE_DIR, 'train_ipq.npy'), train_ipq)
    np.save(os.path.join(FEATURE_DIR, 'test_ipq.npy'), test_ipq)
    print("IPQ features saved.")

    # Extract deep features for each specified model
    for model_arch, model_filename in IMAGE_MODELS.items():
        print(f"\nProcessing features using {model_arch}")
        model_path = os.path.join(MODEL_DIR, model_filename)

        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}. Skipping feature extraction for {model_arch}.")
            continue

        model = MultimodalPricePredictor(image_model_name=model_arch, text_model_name=TEXT_MODEL_NAME).to(DEVICE)
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except RuntimeError as e:
             print(f"Error loading model state dict for {model_arch}: {e}")
             print("Ensure the model architecture in src/model.py matches the saved checkpoint.")
             continue


        # Extract and save features
        train_features = get_deep_features(model, train_loader)
        np.save(os.path.join(FEATURE_DIR, f'train_deep_features_{model_arch}.npy'), train_features)
        print(f"Train deep features for {model_arch} saved.")

        test_features = get_deep_features(model, test_loader)
        np.save(os.path.join(FEATURE_DIR, f'test_deep_features_{model_arch}.npy'), test_features)
        print(f"Test deep features for {model_arch} saved.")

    print("\n--- Deep Feature Extraction Complete ---")