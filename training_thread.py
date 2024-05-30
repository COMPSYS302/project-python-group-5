import torch
import torch.nn as nn
import torch.optim as optim
from PyQt5.QtCore import QThread, pyqtSignal

class TrainingThread(QThread):
    progress = pyqtSignal(int)
    epoch_progress = pyqtSignal(int, float, float)
    finished = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, model, train_loader, val_loader, epochs, batch_size, stop_event):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.batch_size = batch_size
        self.stop_event = stop_event

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        try:
            total_steps = len(self.train_loader) * self.epochs
            current_step = 0
            for epoch in range(self.epochs):
                if self.stop_event.is_set():
                    break
                running_loss = 0.0
                correct = 0
                total = 0
                for images, labels in self.train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    current_step += 1
                    self.progress.emit(int((current_step / total_steps) * 100))

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = (correct / total) * 100
                self.epoch_progress.emit(epoch + 1, epoch_loss, epoch_accuracy)

            if not self.stop_event.is_set():
                torch.save(self.model.state_dict(), 'trained_model.pth')
                print("Model saved to trained_model.pth")
            self.finished.emit()
        except Exception as e:
            self.error_signal.emit(str(e))
            self.finished.emit()