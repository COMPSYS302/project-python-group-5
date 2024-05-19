# ui_components.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QLabel

class SearchBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.searchBar = QLineEdit()
        self.searchBar.setPlaceholderText("Search...")
        self.layout.addWidget(self.searchBar)

        self.imageArea = ImageArea()
        self.layout.addWidget(self.imageArea)
        self.setLayout(self.layout)

class ImageArea(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Image Area")
        self.setWordWrap(True)
