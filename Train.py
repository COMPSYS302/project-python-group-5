from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox
from PyQt5.QtCore import Qt

class TrainingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)  # Adjust margins as needed (left, top, right, bottom)
        layout.setSpacing(10)  # Adjust spacing between widgets in the layout

        # Model selector dropdown with label above
        self.modelSelector = QComboBox()
        self.modelSelector.addItems(["Model 1", "Model 2", "Model 3", "Model 4"])  # Example model names
        modelSelectorLayout = QVBoxLayout()
        modelSelectorLayout.setSpacing(5)  # Reduced spacing between label and dropdown

        modelSelectorLabel = QLabel("Select Model:")
        modelSelectorLayout.addWidget(modelSelectorLabel)
        modelSelectorLayout.addWidget(self.modelSelector)
        layout.addLayout(modelSelectorLayout)

        # Add sliders with labels above them
        self.addSlider(layout, "Train/Validation Ratio", True)
        self.addSlider(layout, "Batch size", True)
        self.addSlider(layout, "Epochs", True)

        # Train data button
        self.trainButton = QPushButton("Train Data", self)
        self.trainButton.clicked.connect(self.onTrainDataClicked)
        layout.addWidget(self.trainButton)

        self.setLayout(layout)

    def addSlider(self, layout, label, isRangeSlider=False):
        # Creating a vertical layout for each slider and its label
        groupLayout = QVBoxLayout()
        groupLayout.setSpacing(5)  # Reduced spacing between label and slider

        labelWidget = QLabel(label)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(100)
        slider.setValue(50)  # Default position
        if isRangeSlider:
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)
        groupLayout.addWidget(labelWidget)  # Label is added first, so it appears above the slider
        groupLayout.addWidget(slider)
        layout.addLayout(groupLayout)  # Add the group layout to the main layout

    def onTrainDataClicked(self):
        # Placeholder function for the train data button
        selectedModel = self.modelSelector.currentText()
        print(f"Training started with {selectedModel}...")  # Use the selected model for training

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    window = TrainingPage()
    window.show()
    sys.exit(app.exec_())
