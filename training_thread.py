import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from PyQt5.QtCore import QThread, pyqtSignal
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class CustomLabelEncoder:
    def __init__(self):
        self.label_mapping = {chr(i): i - 65 for i in range(65, 91)}  # A-Z -> 0-25
        self.label_mapping.update({chr(i): i - 22 for i in range(48, 58)})  # 0-9 -> 26-35

    def transform(self, label):
        if label not in self.label_mapping:
            raise ValueError(f"Label '{label}' not found in label mapping.")
        return self.label_mapping[label]


class SubBatchDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        label, image = self.dataset[self.indices[idx]]
        return label, image


class TrainingThread(QThread):
    progress = pyqtSignal(int)
    epoch_progress = pyqtSignal(int, float, float)
    preparing = pyqtSignal(int, int)
    finished = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, model, train_dataset, val_loader, epochs, original_batch_size, stop_event):
        super(TrainingThread, self).__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.val_loader = val_loader
        self.epochs = epochs
        self.original_batch_size = original_batch_size
        self.stop_event = stop_event
        self.scaler = GradScaler()
        self.label_encoder = CustomLabelEncoder()

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Adjust batch size to fit in GPU memory
        max_sub_batch_size = self.find_max_sub_batch_size(device)
        self.preparing.emit(100, max_sub_batch_size)

        # Main training loop
        try:
            for epoch in range(self.epochs):
                if self.stop_event.is_set():
                    break

                running_loss = 0.0
                correct = 0
                total = 0

                # Generate sub-batches
                indices = list(range(len(self.train_dataset)))
                sub_batches = [indices[i:i + max_sub_batch_size] for i in range(0, len(indices), max_sub_batch_size)]

                for batch_idx, sub_batch_indices in enumerate(sub_batches):
                    sub_batch_dataset = SubBatchDataset(self.train_dataset, sub_batch_indices)
                    sub_batch_loader = DataLoader(sub_batch_dataset, batch_size=max_sub_batch_size, shuffle=True)

                    for labels, images in sub_batch_loader:
                        images = images.to(device)
                        # Convert labels to tensor and move to device
                        if isinstance(labels, tuple):
                            labels = torch.tensor([self.label_encoder.transform(label) if isinstance(label, str) else label for label in labels]).to(device)
                        else:
                            labels = torch.tensor(self.label_encoder.transform(labels) if isinstance(labels, str) else labels).to(device)
                        optimizer.zero_grad()

                        with autocast():
                            outputs = self.model(images)
                            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                                raise ValueError("NaN or Inf in outputs")
                            loss = criterion(outputs, labels)
                            if torch.isnan(loss).any() or torch.isinf(loss).any():
                                raise ValueError("NaN or Inf in loss")

                        self.scaler.scale(loss).backward()

                        # Check for NaNs in gradients
                        for param in self.model.parameters():
                            if param.grad is not None and (
                                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                raise ValueError("NaN or Inf in gradients")

                        self.scaler.step(optimizer)
                        self.scaler.update()

                        running_loss += loss.item() * images.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        # Calculate progress
                        current_step = epoch * len(self.train_dataset) + batch_idx * max_sub_batch_size
                        total_steps = self.epochs * len(self.train_dataset)
                        self.progress.emit(int((current_step / total_steps) * 100))

                epoch_loss = running_loss / total
                epoch_accuracy = (correct / total) * 100
                self.epoch_progress.emit(epoch + 1, epoch_loss, epoch_accuracy)

            if not self.stop_event.is_set():
                torch.save(self.model.state_dict(), 'model.pth')
                self.finished.emit()

        except Exception as e:
            self.error_signal.emit(str(e))
            self.finished.emit()

    def find_max_sub_batch_size(self, device):
        for batch_size in range(self.original_batch_size, 0, -1):
            if self.test_memory(batch_size, device):
                return batch_size
        return 1

    def test_memory(self, batch_size, device):
        try:
            sub_batch_indices = list(range(0, min(batch_size, len(self.train_dataset))))
            sub_batch_dataset = SubBatchDataset(self.train_dataset, sub_batch_indices)
            sub_batch_loader = DataLoader(sub_batch_dataset, batch_size=batch_size, shuffle=False)

            for labels, images in sub_batch_loader:
                images = images.to(device)
                # Convert labels to tensor and move to device
                if isinstance(labels, tuple):
                    labels = torch.tensor([self.label_encoder.transform(label) if isinstance(label, str) else label for label in labels]).to(device)
                else:
                    labels = torch.tensor(self.label_encoder.transform(labels) if isinstance(labels, str) else labels).to(device)
                with autocast():
                    _ = self.model(images)
                break
            return True
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                return False
            else:
                raise e
