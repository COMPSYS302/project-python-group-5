import csv
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QProgressBar, QLabel, QMessageBox
import time

class ProcessingThread(QThread):
    progress = pyqtSignal(int)
    timeLeft = pyqtSignal(str)
    dataLoaded = pyqtSignal(list, list)
    errorOccurred = pyqtSignal(str)

    def __init__(self, filePath):
        super().__init__()
        self.filePath = filePath

    def run(self):
        try:
            start_time = time.time()
            all_images = []
            unique_images = []
            unique_pixel_data = set()

            # Mapping for incorrect number labels
            number_mapping = {0: 9, 1: 0, 2: 7, 3: 6, 4: 1, 5: 8, 6: 4, 7: 3, 8: 2, 9: 5}

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
                    label_number = int(row[0])
                    if 0 <= label_number <= 25:
                        label = chr(label_number + 65)  # Convert to 'A'-'Z'
                    elif 26 <= label_number <= 35:
                        corrected_number = label_number - 26  # Normalize to 0-9 range
                        corrected_label = number_mapping[corrected_number]
                        label = chr(corrected_label + 48)  # Convert corrected number to '0'-'9'
                    else:
                        raise ValueError("Label number out of range (0-35).")

                    pixel_data = np.array([int(x) for x in row[1:]], dtype=np.uint8).reshape((28, 28))
                    image = QImage(28, 28, QImage.Format_RGB888)
                    for y in range(28):
                        for x in range(28):
                            gray = pixel_data[y][x]
                            color = QColor(gray, gray, gray)
                            image.setPixelColor(x, y, color)

                    all_images.append((label, image))

                    # Check for uniqueness
                    pixel_data_tuple = tuple(pixel_data.flatten())
                    if pixel_data_tuple not in unique_pixel_data:
                        unique_pixel_data.add(pixel_data_tuple)
                        unique_images.append((label, image))

                    elapsed_time = time.time() - start_time
                    progress_percent = int((index + 1) / total_lines * 100)
                    self.progress.emit(progress_percent)

                    # Time estimation
                    if elapsed_time > 0:
                        estimated_total_time = elapsed_time / ((index + 1) / total_lines)
                        estimated_time_left = estimated_total_time - elapsed_time
                        self.timeLeft.emit(f"{int(estimated_time_left // 60)}m {int(estimated_time_left % 60)}s")

                    if self.isInterruptionRequested():
                        return

            self.dataLoaded.emit(all_images, unique_images)
        except Exception as e:
            self.errorOccurred.emit(str(e))

class ProcessingWindow(QMainWindow):
    def __init__(self, filePath, parent=None):
        super().__init__(parent)
        self.filePath = filePath
        self.initUI()
        self.thread = ProcessingThread(filePath)
        self.thread.progress.connect(self.updateProgressBar)
        self.thread.timeLeft.connect(self.updateTimeLabel)
        self.thread.dataLoaded.connect(self.finishedProcessing)
        self.thread.errorOccurred.connect(self.handleError)
        self.thread.start()

    def initUI(self):
        self.setWindowTitle("Processing CSV File")
        self.setGeometry(300, 300, 300, 200)

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)

        self.progressBar = QProgressBar()
        layout.addWidget(self.progressBar)

        self.timeLabel = QLabel("Estimated Time Left: Calculating...")
        layout.addWidget(self.timeLabel)

        self.stopButton = QPushButton("Stop")
        self.stopButton.clicked.connect(self.stopProcessing)
        layout.addWidget(self.stopButton)

    def updateProgressBar(self, value):
        self.progressBar.setValue(value)

    def updateTimeLabel(self, time_left):
        self.timeLabel.setText(f"Estimated Time Left: {time_left}")

    def stopProcessing(self):
        self.thread.requestInterruption()
        self.thread.wait()
        self.close()

    def handleError(self, error_message):
        QMessageBox.critical(self, "Error Processing CSV", error_message)
        self.close()

    def finishedProcessing(self, all_images, unique_images):
        self.close()
        self.parent().storeImages(all_images, unique_images)
