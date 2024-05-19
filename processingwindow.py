# processingwindow.py
import csv
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QProgressBar, QLabel, QMessageBox

class ProcessingThread(QThread):
    progress = pyqtSignal(int)
    dataLoaded = pyqtSignal(list)
    errorOccurred = pyqtSignal(str)  # Signal to report errors

    def __init__(self, filePath):
        super().__init__()
        self.filePath = filePath

    def run(self):
        try:
            images = []
            with open(self.filePath, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                if not rows:
                    raise ValueError("CSV file is empty or the format is incorrect.")

                # Assume the first row might be headers or labels and check if it's numeric
                if not rows[0][0].isdigit():
                    rows = rows[1:]  # Skip the header row

                total_lines = len(rows)
                if total_lines == 0:
                    raise ValueError("CSV file contains no data or only headers.")

                for index, row in enumerate(rows):
                    if len(row) == 785:  # Assuming there's one label or identifier column
                        row = row[1:]  # Skip the label
                    pixel_data = np.array([int(x) for x in row], dtype=np.uint8).reshape((28, 28))
                    image = QImage(28, 28, QImage.Format_RGB888)
                    for y in range(28):
                        for x in range(28):
                            gray = pixel_data[y][x]
                            color = QColor(gray, gray, gray)
                            image.setPixelColor(x, y, color)
                    images.append(image)
                    progress_percent = int((index + 1) / total_lines * 100)
                    self.progress.emit(progress_percent)
                    if self.isInterruptionRequested():
                        return
            self.dataLoaded.emit(images)
        except Exception as e:
            self.errorOccurred.emit(str(e))

class ProcessingWindow(QMainWindow):
    def __init__(self, filePath, parent=None):
        super().__init__(parent)
        self.filePath = filePath
        self.initUI()
        self.thread = ProcessingThread(filePath)
        self.thread.progress.connect(self.updateProgressBar)
        self.thread.dataLoaded.connect(self.finishedProcessing)
        self.thread.errorOccurred.connect(self.handleError)
        self.thread.start()

    def initUI(self):
        self.setWindowTitle("Processing CSV File")
        self.setGeometry(300, 300, 300, 200)

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)

        self.stopButton = QPushButton("Stop")
        self.stopButton.clicked.connect(self.stopProcessing)
        layout.addWidget(self.stopButton)

        self.progressBar = QProgressBar(self)
        self.progressBar.setMaximum(100)
        layout.addWidget(self.progressBar)

    def updateProgressBar(self, value):
        self.progressBar.setValue(value)

    def stopProcessing(self):
        self.thread.requestInterruption()
        self.thread.wait()
        self.close()

    def handleError(self, error_message):
        QMessageBox.critical(self, "Error Processing CSV", error_message)
        self.close()

    def finishedProcessing(self, images):
        self.close()  # Optionally update window or close after processing
