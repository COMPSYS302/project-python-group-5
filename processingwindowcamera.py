from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit, QApplication
import pandas as pd
from numpy import empty


class ProcessingWindowCamera(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()
        self.data = None

    def setupUi(self):
        self.setWindowTitle("Data Processing Window")
        self.setGeometry(200, 200, 600, 400)
        layout = QVBoxLayout(self)

        loadButton = QPushButton("Load Data", self)
        loadButton.clicked.connect(self.loadData)
        layout.addWidget(loadButton)

        self.dataLabel = QLabel("Data not loaded.", self)
        layout.addWidget(self.dataLabel)

        self.dataTextEdit = QTextEdit(self)
        self.dataTextEdit.setReadOnly(True)
        layout.addWidget(self.dataTextEdit)

    def loadData(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open CSV file", "", "CSV Files (*.csv);;All Files (*)")
        if filePath:
            self.data = pd.read_csv(filePath)
            self.dataLabel.setText(f"Data loaded: {filePath}")
            self.showData()

    def showData(self):
        if self.data is not empty:
            self.dataTextEdit.setText(str(self.data.head()))  # Show first few rows of the data
        else:
            self.dataTextEdit.setText("No data to display.")

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = ProcessingWindow()
    window.show()
    sys.exit(app.exec_())
