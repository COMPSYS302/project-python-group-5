import torch
import torch.nn as nn
import torch.optim as optim
from PyQt5.QtCore import QThread, pyqtSignal
import time


import torch
import torch.nn as nn
import torch.optim as optim
from PyQt5.QtCore import QThread, pyqtSignal
import time

class TrainingThread(QThread):
    progress = pyqtSignal(int)
    loss_signal = pyqtSignal(float)
    acc_signal = pyqtSignal(float)
    error_signal = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model, train_loader, val_loader, epochs, batch_size, stop_event):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.batch_size = batch_size
        self.stop_event = stop_event

    def run(self):
        try:
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

            for epoch in range(self.epochs):
                if self.stop_event.is_set():
                    self.finished.emit()
                    return

                self.model.train()
                running_loss = 0.0
                total_samples = 0
                correct_samples = 0

                for images, labels in self.train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_samples += labels.size(0)
                    correct_samples += (predicted == labels).sum().item()

                epoch_loss = running_loss / len(self.train_loader.dataset)
                epoch_accuracy = (correct_samples / total_samples) * 100
                self.loss_signal.emit(epoch_loss)
                self.acc_signal.emit(epoch_accuracy)

                self.progress.emit(int((epoch + 1) / self.epochs * 100))

            self.finished.emit()

        except Exception as e:
            self.error_signal.emit(str(e))
            self.finished.emit()
