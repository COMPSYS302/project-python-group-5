from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QLabel, QProgressBar, QWidget
from PyQt5.QtCore import Qt

class BatchPreparationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Batch Preparation")
        self.setGeometry(300, 300, 300, 200)

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)

        self.batchesLabel = QLabel("Batches Needed: 0")
        layout.addWidget(self.batchesLabel)

        self.preparationBar = QProgressBar()
        layout.addWidget(self.preparationBar)

    def updatePreparationProgress(self, progress, num_batches):
        self.preparationBar.setValue(progress)
        self.batchesLabel.setText(f"Batches Needed: {num_batches}")
