import sys
import pandas as pd
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QFrame
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# Assuming your CSV file path
csv_file_path = 'C:\\Users\\rohan\\Documents\\Downloads\\sign_mnist_alpha_digits_test_train (1).csv'
data = pd.read_csv(csv_file_path)

# Assuming pixel columns are named 'pixel1', 'pixel2', ..., 'pixelN'
pixel_columns = [col for col in data.columns if 'pixel' in col]

class SignLanguageSearch(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.images = {}
        self.loadImages()

    def initUI(self):
        # Main layout
        layout = QVBoxLayout()

        # Search bar
        self.lineEdit = QLineEdit(self)
        self.lineEdit.setPlaceholderText('Enter a letter or number...')
        self.lineEdit.textChanged.connect(self.onTextChanged)
        layout.addWidget(self.lineEdit)

        # Scroll area for images
        self.scrollArea = QScrollArea(self)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollLayout = QHBoxLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        layout.addWidget(self.scrollArea)

        self.setLayout(layout)

    def loadImages(self):
        for index, row in data.iterrows():
            image_array = np.array(row[pixel_columns], dtype=np.uint8).reshape((28, 28))  # Assuming image size is 28x28
            qimage = QImage(image_array, image_array.shape[1], image_array.shape[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            self.images[row['label']] = pixmap

    def get_label_from_input(self, input_char):
        if input_char.isdigit():  # Checks if the input is a digit
            return int(input_char) + 26  # Maps digits 0-9 to labels 26-35
        else:
            index = ord(input_char.upper()) - ord('A')
            if index == 9 or index == 25:  # Skip J and Z
                return None
            if index > 9:  # Adjust index to skip J
                index -= 1
            return index

    def onTextChanged(self, text):
        # Clear current images in layout
        for i in reversed(range(self.scrollLayout.count())):
            widget = self.scrollLayout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Get label index from input text
        if text:
            label_index = self.get_label_from_input(text)
            if label_index is not None:
                # Filter data for this specific label
                filtered_data = data[data['label'] == label_index]

                # Display images
                for index, row in filtered_data.iterrows():
                    lbl = QLabel()
                    lbl.setPixmap(self.images[row['label']].scaled(100, 100, Qt.KeepAspectRatio))
                    self.scrollLayout.addWidget(lbl)
            else:
                print("Invalid input or unsupported character")

def main():
    app = QApplication(sys.argv)
    ex = SignLanguageSearch()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
