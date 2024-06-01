import sys
import cv2
import os
import torch
import torchvision.transforms as transforms
import warnings
import numpy as np

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QApplication, QMessageBox, QComboBox
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
            QThread.msleep(500)  # Capture frame every 100ms for stability

    def stop(self):
        self.running = False
        self.capture.release()

class PredictionThread(QThread):
    predictionMade = pyqtSignal(str)

    def __init__(self, model, frame, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.frame = frame
        self.device = device

    def run(self):
        try:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),  # Convert image to grayscale with 3 channels
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            image = transform(image).unsqueeze(0).to(self.device)  # Move to the same device as the model

            with torch.no_grad():
                self.model.eval()
                outputs = self.model(image)
                _, predicted = torch.max(outputs, 1)
                prediction = self.idx_to_class(predicted.item())

            self.predictionMade.emit(prediction)
        except Exception as e:
            print(f"Error in PredictionThread: {e}")

    def idx_to_class(self, idx):
        # Map the predicted index to the corresponding ASL character
        asl_classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if 0 <= idx < len(asl_classes):
            return asl_classes[idx]
        return "Unknown"

class CameraWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_files = self.getModelFiles()
        self.model = self.modelComboBox.currentText()
        self.initUI()

        if self.model is not None:
            # Initialize MediaPipe Hands
            self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.mp_drawing = mp.solutions.drawing_utils

            self.cameraThread = CameraThread()
            self.cameraThread.frameCaptured.connect(self.updateFrame)
            self.cameraThread.start()

    def initUI(self):
        self.setWindowTitle("Camera Window")
        self.setGeometry(200, 200, 800, 600)
        layout = QVBoxLayout(self)

        self.modelComboBox = QComboBox(self)
        self.modelComboBox.addItems(self.model_files)
        self.modelComboBox.currentIndexChanged.connect(self.model_selection_changed)
        layout.addWidget(self.modelComboBox)

        self.cameraLabel = QLabel(self)
        self.cameraLabel.setFixedSize(640, 480)
        layout.addWidget(self.cameraLabel)

        self.predictionLabel = QLabel("Prediction: ", self)
        layout.addWidget(self.predictionLabel)

        self.setLayout(layout)

    def load_model(self):
        try:
            model = ModifiedAlexNet(num_classes=36)  # Adjust num_classes as needed
            model.load_state_dict(torch.load(r'C:\project-python-group-5\trained_model.pth'))
            model.to(self.device)
            model.eval()  # Set the model to evaluation mode
            return model
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "No Trained Model Available")
            return None
    def getModelFiles(self):
        modeldirectory = r'C:\project-python-group-5'
        try:
            return [f for f in os.listdir(modeldirectory) if f.endswith('.pth')]
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "No Trained Model Available")
            return None

    def updateFrame(self, frame):
        try:
            frame = self.detectAndDrawHands(frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytesPerLine = ch * w
            qImg = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
            self.cameraLabel.setPixmap(QPixmap.fromImage(qImg))
            self.processFrame(frame)  # Process the frame for prediction
        except Exception as e:
            print(f"Error in updateFrame: {e}")

    def processFrame(self, frame):
        try:
            self.predictionThread = PredictionThread(self.model, frame, self.device)
            self.predictionThread.predictionMade.connect(self.updatePrediction)
            self.predictionThread.start()
        except Exception as e:
            print(f"Error in processFrame: {e}")

    def updatePrediction(self, prediction):
        try:
            self.predictionLabel.setText(f"Prediction: {prediction}")
        except Exception as e:
            print(f"Error in updatePrediction: {e}")

    def detectAndDrawHands(self, frame):
        try:
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            return frame
        except Exception as e:
            print(f"Error in detectAndDrawHands: {e}")
            return frame

    def closeEvent(self, event):
        self.cameraThread.stop()
        self.mp_hands.close()  # Close MediaPipe Hands
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraWindow()
    window.show()
    sys.exit(app.exec_())