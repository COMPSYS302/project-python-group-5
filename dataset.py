from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class SignLanguageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        qimage = self.images[idx][1]  # Get the QImage part
        label = int(self.labels[idx])  # Ensure label is an int
        try:
            image = self.qimage_to_pil(qimage)
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")
            raise

    @staticmethod
    def qimage_to_pil(qimage):
        buffer = qimage.bits().asstring(qimage.byteCount())
        pil_image = Image.frombuffer("RGB", (qimage.width(), qimage.height()), buffer, "raw", "RGB", 0, 1)
        return pil_image
