import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

class SignLanguageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]  # Use .iloc for positional indexing
        label = int(row.iloc[0])  # Corrected to use .iloc for accessing the label
        image = np.array(row[1:], dtype=np.uint8).reshape(28, 28)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label