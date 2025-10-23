# main_train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from torchvision import transforms
from tqdm import tqdm
import os
import argparse
import csv

from src.dataset import ProductDataset
from src.model import MultimodalPricePredictor
from src.utils import smape

parser = argparse.ArgumentParser(description='Train a multimodal price prediction model.')
parser.add_argument('--image_model', type=str, default='efficientnet_b0', help='Name of the image model from timm library (e.g., efficientnet_b0, resnet50).')
parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
args = parser.parse_args()

IMAGE_DIR = './data/images/'
MODEL_SAVE_PATH = './saved_models/'
LOG_FILE = 'training_log.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


log_header = ['image_model', 'epoch', 'train_loss', 'val_smape']
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_header)


print("Loading and preprocessing data...")
df = pd.read_csv('./data/train.csv')
df['log_price'] = np.log1p(df['price'])
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


train_dataset = ProductDataset(train_df, tokenizer, image_transform, IMAGE_DIR)
val_dataset = ProductDataset(val_df, tokenizer, image_transform, IMAGE_DIR)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


print(f"Using device: {DEVICE}")
print(f"Training with image model: {args.image_model}")

model = MultimodalPricePredictor(image_model_name=args.image_model).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

best_smape = float('inf')

for epoch in range(args.epochs):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        image = batch['image'].to(DEVICE)
        ipq = batch['ipq'].to(DEVICE)         
        targets = batch['target'].to(DEVICE)

        outputs = model(input_ids, attention_mask, image, ipq) 
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

    avg_train_loss = total_loss / len(train_loader)
    scheduler.step() 

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            image = batch['image'].to(DEVICE)
            ipq = batch['ipq'].to(DEVICE)      
            outputs = model(input_ids, attention_mask, image, ipq) 
            val_preds.extend(np.expm1(outputs.cpu().numpy()))
            val_targets.extend(np.expm1(batch['target'].cpu().numpy()))

    current_smape = smape(np.array(val_targets), np.array(val_preds))
    print(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f} - Validation SMAPE: {current_smape:.4f}")


    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args.image_model, epoch + 1, avg_train_loss, current_smape])

    if current_smape < best_smape:
        best_smape = current_smape
        model_filename = f'best_model_{args.image_model}.pth'
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, model_filename))
        print(f"New best model saved as {model_filename} with SMAPE: {best_smape:.4f}")

print("Training finished successfully")
