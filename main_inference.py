import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from torchvision import transforms
from tqdm import tqdm
import os
import argparse 

from src.dataset import ProductDataset
from src.model import MultimodalPricePredictor

parser = argparse.ArgumentParser(description='Generate predictions using a trained model.')
parser.add_argument('--image_model', type=str, default='efficientnet_b0', help='Name of the image model used for training.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference.')
args = parser.parse_args()


IMAGE_DIR = './data/images/'
MODEL_SAVE_PATH = './saved_models/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_FILENAME = f'best_model_{args.image_model}.pth'
MODEL_PATH = os.path.join(MODEL_SAVE_PATH, MODEL_FILENAME)

print("Loading test data...")
test_df = pd.read_csv('./data/test.csv')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = ProductDataset(test_df, tokenizer, image_transform, IMAGE_DIR, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


print(f"Using device: {DEVICE}")
print(f"Loading model from: {MODEL_PATH}")
model = MultimodalPricePredictor(image_model_name=args.image_model).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


all_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting on test set"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        image = batch['image'].to(DEVICE)
        ipq = batch['ipq'].to(DEVICE)  

        outputs = model(input_ids, attention_mask, image, ipq) 
        preds = np.expm1(outputs.cpu().numpy())  
        all_preds.extend(preds)


submission_df = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': all_preds
})
submission_df['price'] = submission_df['price'].clip(lower=0)
submission_df.to_csv('test_out.csv', index=False)

print("Submission file 'test_out.csv' created successfully")
