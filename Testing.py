from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import pyqtSignal

class ClickableLabel(QLabel):
    clicked = pyqtSignal()  # Signal to be emitted on click

    def __init__(self, image, parent=None):
        super(ClickableLabel, self).__init__(parent)
        self.image = image

    def mousePressEvent(self, event):
        self.clicked.emit()  # Emit the clicked signal when the label is clicked
