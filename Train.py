import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QProgressBar, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from sympy import this
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pyqtgraph as pg
from models import SimpleDNN, CNNModel, ComplexDNN
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
        self.modelSelector.addItems(["Simple DNN", "CNN Model", "Complex DNN"])
        self.modelSelectorLabel = QLabel("Select Model:")
        modelSelectorLayout = QVBoxLayout()
        modelSelectorLayout.addWidget(self.modelSelectorLabel)
        modelSelectorLayout.addWidget(self.modelSelector)
        self.layout.addLayout(modelSelectorLayout)

        self.trainRatioSlider = self.addSlider("Train/Validation Ratio", 10, 90)
        self.batchSizeSlider = self.addSlider("Batch size", 16, 128)
        self.epochsSlider = self.addSlider("Epochs", 1, 50)

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

    def addSlider(self, label, min_value, max_value):
        sliderLayout = QVBoxLayout()
        sliderLabel = QLabel(label)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue((min_value + max_value) // 2)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval((max_value - min_value) // 10)
        sliderLayout.addWidget(sliderLabel)
        sliderLayout.addWidget(slider)
        self.layout.addLayout(sliderLayout)
        return slider

    def onTrainDataClicked(self):
        self.modelSelector.hide()
        self.modelSelectorLabel.hide()
        self.trainRatioSlider.parent().hide()
        self.batchSizeSlider.parent().hide()
        self.epochsSlider.parent().hide()
        self.trainButton.hide()

        self.layout.addWidget(self.lossPlotWidget)
        self.layout.addWidget(self.accPlotWidget)
        self.stopButton.setEnabled(True)

        model_name = self.modelSelector.currentText()
        train_ratio = self.trainRatioSlider.value() / 100
        batch_size = self.batchSizeSlider.value()
        epochs = self.epochsSlider.value()

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = SignLanguageDataset(self.csv_file, transform=transform)
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = {'Simple DNN': SimpleDNN, 'CNN Model': CNNModel, 'Complex DNN': ComplexDNN}[model_name]()

        self.train_thread = TrainingThread(model, train_loader, val_loader, epochs, batch_size, self.stop_event)
        self.train_thread.progress.connect(self.progressBar.setValue)
        self.train_thread.loss_signal.connect(lambda loss: self.lossCurve.setData(self.train_thread.epoch_list, self.train_thread.loss_list))
        self.train_thread.acc_signal.connect(lambda acc: self.accCurve.setData(self.train_thread.epoch_list, self.train_thread.acc_list))
        self.train_thread.start()

    def onStopTrainingClicked(self):
        self.stop_event.set()
        self.train_thread.wait()

    def closeTrainingView(self):
        self.modelSelector.show()
        self.modelSelectorLabel.show()
        self.trainRatioSlider.parent().show()
        self.batchSizeSlider.parent().show()
        self.epochsSlider.parent().show()
        this.trainButton.show()
        this.stopButton.setEnabled(False)
        this.progressBar.setValue(0)
        this.layout.removeWidget(this.lossPlotWidget)
        this.layout.removeWidget(this.accPlotWidget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    trainApp = Train('path_to_your_csv_file.csv')
    trainApp.show()
    sys.exit(app.exec_())
