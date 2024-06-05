import sys
import cv2
import os
import torch
import torchvision.transforms as transforms
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QApplication, QMessageBox, QComboBox, QPushButton
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

from models import ModifiedAlexNet
import mediapipe as mp

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

class CameraThread(QThread):
    frameCaptured = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.capture = cv2.VideoCapture(0)  # Initialize camera
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.frameCaptured.emit(frame)
            QThread.msleep(500)

    def stop(self):
        self.running = False
        self.capture.release()

class PredictionThread(QThread):
    predictionMade = pyqtSignal(dict)

    def __init__(self, model, frame, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.frame = frame
        self.device = device

    def run(self):
        try:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            image = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                self.model.eval()
                outputs = self.model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                prediction_scores = {chr(65+i): prob for i, prob in enumerate(probabilities.squeeze().tolist())}

            self.predictionMade.emit(prediction_scores)
        except Exception as e:
            print(f"Error in PredictionThread: {e}")

class CameraWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_files = self.getModelFiles()
        self.model = None
        self.initUI()
        self.loadSelectedModel()

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.cameraThread = CameraThread()
        self.cameraThread.frameCaptured.connect(self.updateFrame)
        self.cameraThread.start()

    def initUI(self):
        self.setWindowTitle("Camera Window")
        self.setGeometry(200, 200, 1200, 600)

        # Central widget
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        # Layout for central widget
        layout = QVBoxLayout(centralWidget)

        # Model selection combo box
        self.modelComboBox = QComboBox()
        self.modelComboBox.addItems([self.formatModelName(f) for f in self.model_files])
        self.modelComboBox.currentIndexChanged.connect(self.loadSelectedModel)
        layout.addWidget(self.modelComboBox)

        # Label to show the camera feed
        self.cameraLabel = QLabel()
        self.cameraLabel.setFixedSize(640, 480)
        layout.addWidget(self.cameraLabel)

        # Prediction label
        self.predictionLabel = QLabel("Prediction: ")
        layout.addWidget(self.predictionLabel)

        # Start capture button
        self.startCaptureButton = QPushButton("Start Capture")
        self.startCaptureButton.clicked.connect(self.startCapture)
        layout.addWidget(self.startCaptureButton)

        # Take picture button
        self.takePictureButton = QPushButton("Take Picture")
        self.takePictureButton.clicked.connect(self.takePicture)
        layout.addWidget(self.takePictureButton)

        # Matplotlib Figure
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def load_model(self, model_path):
        try:
            model = ModifiedAlexNet(num_classes=36)  # Adjust num_classes as needed
            if self.device == 'cuda':
                model.load_state_dict(torch.load(model_path))
            else:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.to(self.device)
            model.eval()  # Set the model to evaluation mode
            return model
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "No Trained Model Available")
            return None

    def getModelFiles(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            return [f for f in os.listdir(current_dir) if f.endswith('.pth')]
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "No Trained Model Available")
            return []

    def loadSelectedModel(self):
        model_name = self.modelComboBox.currentText().replace(' ', '_').lower() + '.pth'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_name)
        self.model = self.load_model(model_path)
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", "Model file does not exist: " + model_path)
            return
        self.model = self.load_model(model_path)
        print("Model loaded: " + model_path)

    def updateFrame(self, frame):
        try:
            frame = self.detectAndDrawHands(frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytesPerLine = ch * w
            qImg = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
            self.cameraLabel.setPixmap(QPixmap.fromImage(qImg))
        except Exception as e:
            print(f"Error in updateFrame: {e}")

    def formatModelName(self, filename):
        return filename.replace('_', ' ').replace('.pth', '').title()

    def startCapture(self):
        if self.model is not None:
            self.captureTimer = QTimer(self)
            self.captureTimer.timeout.connect(self.captureFrame)
            self.captureTimer.start(100)  # Capture every 100ms
        else:
            QMessageBox.warning(self, "Warning", "No model selected/trained")

    def captureFrame(self):
        ret, frame = self.cameraThread.capture.read()
        if ret:
            self.processFrame(frame)

    def takePicture(self):
        if self.model is not None:
            ret, frame = self.cameraThread.capture.read()
            if ret:
                self.processFrame(frame)
        else:
            QMessageBox.warning(self, "Warning", "No model selected/trained")

    def processFrame(self, frame):
        if self.model is not None:
            self.predictionThread = PredictionThread(self.model, frame, self.device)
            self.predictionThread.predictionMade.connect(self.updatePrediction)
            self.predictionThread.start()

    def updatePrediction(self, prediction_scores):
        try:
            self.predictionLabel.setText("Prediction: " + max(prediction_scores, key=prediction_scores.get))
            self.updateGraph(prediction_scores)
        except Exception as e:
            print(f"Error in updatePrediction: {e}")

    def updateGraph(self, prediction_scores):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        characters = prediction_scores.keys()
        scores = prediction_scores.values()
        ax.bar(characters, scores, color='blue')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Confidence')
        ax.set_title('Prediction Confidence')
        self.canvas.draw()

    def detectAndDrawHands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        return frame

    def closeEvent(self, event):
        self.cameraThread.stop()
        self.mp_hands.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraWindow()
    window.show()
    sys.exit(app.exec_())
