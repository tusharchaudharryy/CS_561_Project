import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import re

class ProductDataset(Dataset):
    def __init__(self, df, tokenizer, image_transform, image_dir, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.image_dir = image_dir
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        text = row['catalog_content']
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=256
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        ipq = 1.0  
        match = re.search(r"Item Pack Quantity:\s*(\d+)", text)
        if match:
            ipq = float(match.group(1))

        image_path = os.path.join(self.image_dir, f"{row['sample_id']}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), (255, 255, 255))
        
        image = self.image_transform(image)

        if self.is_test:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'image': image,
                'ipq': torch.tensor(ipq, dtype=torch.float32)
            }
        else:
            target = torch.tensor(row['log_price'], dtype=torch.float32)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'image': image,
                'target': target,
                'ipq': torch.tensor(ipq, dtype=torch.float32)
            }
