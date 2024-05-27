import sys
import cv2
import mediapipe as mp
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QApplication
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

class CameraWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.camera = None
        self.setupUi()

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def setupUi(self):
        self.setWindowTitle("Camera View")
        self.setGeometry(100, 100, 640, 480)
        layout = QVBoxLayout(self)

        self.imageLabel = QLabel(self)
        layout.addWidget(self.imageLabel)

        self.startButton = QPushButton("Start Camera", self)
        self.startButton.clicked.connect(self.startCamera)
        layout.addWidget(self.startButton)

    def startCamera(self):
        self.camera = cv2.VideoCapture(0)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)

    def updateFrame(self):
        ret, frame = self.camera.read()
        if ret:
            frame = self.detectAndDrawHands(frame)
            qt_img = self.convert_cv_qt(frame)
            self.imageLabel.setPixmap(qt_img)

    def detectAndDrawHands(self, frame):
        # Convert the BGR image to RGB, flip the image around y-axis for correct handedness output
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.flip(rgb_frame, 1)
        results = self.mp_hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        return cv2.flip(rgb_frame, 1)  # Flip back for correct display

    def convert_cv_qt(self, cv_img):
        """Convert from an OpenCV image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        """Properly release the camera resource and close MediaPipe Hands."""
        if self.camera is not None:
            self.timer.stop()
            self.camera.release()
        self.mp_hands.close()  # Close MediaPipe Hands
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CameraWindow()
    win.show()
    sys.exit(app.exec_())
