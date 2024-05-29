import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QProgressBar, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pyqtgraph as pg
from models import ModifiedAlexNet, ModifiedResNet, CustomCNN  # Assuming these are defined in models.py
from dataset import SignLanguageDataset
from training_thread import TrainingThread
from threading import Event

class Train(QWidget):
    def __init__(self, csv_file):
        super().__init__()
        self.csv_file = csv_file
        self.initUI()
        self.train_thread = None
        self.stop_event = Event()

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

        self.setLayout(self.layout)

    def addSlider(self, label, min_value, max_value, tick_interval):
        layout = QHBoxLayout()
        sliderLabel = QLabel(label)
        valueLabel = QLabel(f"{(min_value + max_value) // 2}%")  # Default label value set to midpoint
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
        model_name = self.modelSelector.currentText()
        validation_ratio = self.trainRatioSlider.value()
        train_ratio = 100 - validation_ratio  # Ensure that train + validation = 100%

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust if necessary
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = SignLanguageDataset(self.csv_file, transform=transform)
        train_size = int(train_ratio / 100 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        batch_size = int((self.batchSizeSlider.value() / 100) * len(dataset))
        epochs = self.epochsSlider.value()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = {'Modified AlexNet': ModifiedAlexNet, 'Modified ResNet': ModifiedResNet, 'Custom CNN': CustomCNN}[model_name]()

        self.train_thread = TrainingThread(model, train_loader, val_loader, epochs, batch_size, self.stop_event)
        self.train_thread.progress.connect(self.progressBar.setValue)
        self.train_thread.loss_signal.connect(
            lambda loss: self.lossCurve.setData(self.train_thread.epoch_list, self.train_thread.loss_list))
        self.train_thread.acc_signal.connect(
            lambda acc: self.accCurve.setData(self.train_thread.epoch_list, self.train_thread.acc_list))
        self.train_thread.start()

        self.trainButton.setEnabled(False)
        self.stopButton.setEnabled(True)

    def onStopTrainingClicked(self):
        self.stop_event.set()
        self.train_thread.wait()
        self.trainButton.setEnabled(True)
        self.stopButton.setEnabled(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    trainApp = Train('path_to_your_csv_file.csv')
    trainApp.show()
    sys.exit(app.exec_())
