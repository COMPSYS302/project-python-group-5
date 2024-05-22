# ui_components.py
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QLabel


class SearchBar(QWidget):
    textChanged = pyqtSignal(str)  # Define the signal

    def __init__(self, parent=None):
        super(SearchBar, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.searchBar = QLineEdit()
        self.searchBar.setPlaceholderText("Search...")
        self.layout.addWidget(self.searchBar)

        self.searchBar.textChanged.connect(
            self.textChanged.emit)  # Connect the QLineEdit text change to emit the signal

        self.imageArea = ImageArea()
        self.layout.addWidget(self.imageArea)
        self.setLayout(self.layout)

class ImageArea(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Image Area")
        self.setWordWrap(True)