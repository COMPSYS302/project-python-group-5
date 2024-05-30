import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QProgressBar, QHBoxLayout, QMessageBox
from PyQt5.QtCore import Qt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pyqtgraph as pg
from models import ModifiedAlexNet, ModifiedResNet, CustomCNN  # Make sure these are correctly imported
from dataset import SignLanguageDataset
from training_thread import TrainingThread
from threading import Event
import time
import torch

class Train(QWidget):
    def __init__(self, csv_file='path_to_your_csv_file.csv'):  # Default path included here
        super().__init__()
        self.csv_file = csv_file
        self.initUI()
        self.train_thread = None
        self.stop_event = Event()
        self.start_time = None

    def initUI(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)

        self.modelSelector = QComboBox()
        self.modelSelector.addItems(["Modified AlexNet", "Modified ResNet", "Custom CNN"])
        self.modelSelectorLabel = QLabel("Select Model:")
        modelSelectorLayout = QVBoxLayout()
        modelSelectorLayout.addWidget(self.modelSelectorLabel)
        modelSelectorLayout.addWidget(self.modelSelector)
        self.layout.addLayout(modelSelectorLayout)

        self.trainRatioSlider, self.trainRatioLabel = self.addSlider("Train/Validation Ratio", 50, 100, 50)
        self.batchSizeSlider, self.batchSizeLabel = self.addSlider("Batch size (as % of total data)", 1, 100, 10)
        self.epochsSlider, self.epochsLabel = self.addSlider("Epochs", 1, 40, 1)

        self.trainButton = QPushButton("Train Data", self)
        self.trainButton.clicked.connect(self.onTrainDataClicked)
        self.layout.addWidget(self.trainButton)

        self.stopButton = QPushButton("Stop Training", self)
        self.stopButton.clicked.connect(self.onStopTrainingClicked)
        self.stopButton.setEnabled(False)
        self.layout.addWidget(self.stopButton)

        self.progressBar = QProgressBar()
        self.layout.addWidget(self.progressBar)

        self.lossPlotWidget = pg.PlotWidget(title="Training Loss")
        self.lossPlotWidget.addLegend()
        self.lossCurve = self.lossPlotWidget.plot(pen='r', name='Training Loss')

        self.accPlotWidget = pg.PlotWidget(title="Validation Accuracy")
        self.accPlotWidget.addLegend()
        self.accCurve = self.accPlotWidget.plot(pen='b', name='Validation Accuracy')

        self.statusLabel = QLabel("Ready")
        self.layout.addWidget(self.statusLabel)

        self.setLayout(self.layout)

    def addSlider(self, label, min_value, max_value, tick_interval):
        layout = QHBoxLayout()
        sliderLabel = QLabel(label)
        valueLabel = QLabel(f"{(min_value + max_value) // 2}%")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue((min_value + max_value) // 2)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(tick_interval)
        slider.valueChanged.connect(lambda value, label=valueLabel: label.setText(f"{value}%"))
        layout.addWidget(sliderLabel)
        layout.addWidget(slider)
        layout.addWidget(valueLabel)
        self.layout.addLayout(layout)
        return slider, valueLabel

    def onTrainDataClicked(self):
        self.disableControls()
        self.start_time = time.time()
        model_name = self.modelSelector.currentText()
        validation_ratio = self.trainRatioSlider.value()
        train_ratio = 100 - validation_ratio

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = SignLanguageDataset(self.csv_file, transform=transform)
        train_size = int(train_ratio / 100 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Dynamically reduce batch size to avoid OOM errors
        batch_size = int((self.batchSizeSlider.value() / 100) * len(dataset))
        while batch_size > 1:
            try:
                torch.cuda.empty_cache()
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                break
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    batch_size //= 2
                else:
                    raise e

        epochs = self.epochsSlider.value()
        model = {'Modified AlexNet': ModifiedAlexNet, 'Modified ResNet': ModifiedResNet, 'Custom CNN': CustomCNN}[model_name]()

        self.train_thread = TrainingThread(model, train_loader, val_loader, epochs, batch_size, self.stop_event)
        self.train_thread.progress.connect(self.updateProgress)
        self.train_thread.epoch_progress.connect(self.updateEpochProgress)
        self.train_thread.finished.connect(self.onTrainingFinished)
        self.train_thread.error_signal.connect(self.handleErrors)
        self.train_thread.start()

    def updateProgress(self, percentage):
        elapsed_time = time.time() - self.start_time
        estimated_total_time = elapsed_time / (percentage / 100) if percentage > 0 else 0
        remaining_time = estimated_total_time - elapsed_time
        self.statusLabel.setText(f"Progress: {percentage}%, Elapsed Time: {elapsed_time:.2f}s, Remaining Time: {remaining_time:.2f}s")
        self.progressBar.setValue(percentage)

    def updateEpochProgress(self, epoch, loss, acc):
        self.lossCurve.setData(range(epoch), [loss] * epoch)  # Update the loss curve
        self.accCurve.setData(range(epoch), [acc] * epoch)  # Update the accuracy curve
        self.statusLabel.setText(f"Epoch: {epoch}, Loss: {loss:.4f}, Acc: {acc:.2f}%")

    def disableControls(self):
        self.trainButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.modelSelector.setEnabled(False)
        self.trainRatioSlider.setEnabled(False)
        self.batchSizeSlider.setEnabled(False)
        self.epochsSlider.setEnabled(False)

    def enableControls(self):
        self.trainButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.modelSelector.setEnabled(True)
        self.trainRatioSlider.setEnabled(True)
        self.batchSizeSlider.setEnabled(True)

    def onTrainingFinished(self):
        self.enableControls()
        elapsed_time = time.time() - self.start_time
        self.statusLabel.setText(f"Training Complete. Elapsed Time: {elapsed_time:.2f}s")
        QMessageBox.information(self, "Training Complete", "Training has finished.")

    def onStopTrainingClicked(self):
        self.stop_event.set()

    def handleErrors(self, message):
        QMessageBox.critical(self, "Training Error", message)
        self.enableControls()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    trainApp = Train()
    trainApp.show()
    sys.exit(app.exec_())
