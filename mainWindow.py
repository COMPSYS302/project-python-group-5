from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QLabel, QScrollArea, QVBoxLayout, QWidget, QApplication, QMessageBox, QGridLayout, QProgressBar, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from processingwindow import ProcessingWindow
from UIcomponents import SearchBar
from Train import Train

class ImageLoaderThread(QtCore.QThread):
    update_pixmap = QtCore.pyqtSignal(QPixmap, int, int, str)
    progress_updated = QtCore.pyqtSignal(int)

    def __init__(self, images, thumbnail_size):
        super().__init__()
        self.images = images
        self.thumbnail_size = thumbnail_size

    def run(self):
        total = len(self.images)
        for i, img in enumerate(self.images):
            label, qimage = img  # Unpack the tuple
            if not isinstance(qimage, QImage):
                print(f"Error: Expected QImage, got {type(qimage)}")
                continue
            pixmap = QPixmap.fromImage(qimage)
            pixmap = pixmap.scaled(self.thumbnail_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            col = i % 6
            row = i // 6
            self.update_pixmap.emit(pixmap, row, col, label)
            self.progress_updated.emit((i + 1) * 100 // total)
            self.msleep(10)

class ImageWindow(QMainWindow):
    def __init__(self, pixmap, label, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Prediction")
        self.setGeometry(100, 100, 400, 800)
        reference = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "R", "S", "T", "U",
                     "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        # Create main layout and central widget
        layout = QVBoxLayout()
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        centralWidget.setLayout(layout)

        # Label for displaying the image description or title
        self.titleLabel = QLabel(reference[int(label)], self)
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.titleLabel.setStyleSheet("font-size: 16px; font-weight: bold; color: black;")
        layout.addWidget(self.titleLabel)

        # Label for displaying the image
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.imageLabel)

        self.displayImage(pixmap)

    def displayImage(self, pixmap):
        self.imageLabel.setPixmap(pixmap)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.images = []
        self.labels = []
        self.filtered_images = []
        self.current_page = 0
        self.images_per_page = 500
        self.csv_file = ""
        self.search_cache = {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Viewer App")
        self.setGeometry(100, 100, 800, 600)
        self.menu = self.menuBar()
        self.setupMenus()
        self.setupUIComponents()

    def setupMenus(self):
        # File menu
        fileMenu = self.menu.addMenu("File")
        loadAction = QAction("Load Data", self)
        loadAction.triggered.connect(self.openFileDialog)
        fileMenu.addAction(loadAction)

        # Train Data action
        trainDataAction = QAction("Train Data", self)
        trainDataAction.triggered.connect(self.openTrainPage)
        fileMenu.addAction(trainDataAction)

        testDataAction = QAction("Test Data", self)
        testDataAction.triggered.connect(self.openTestPage)
        fileMenu.addAction(testDataAction)

        # View menu
        viewMenu = self.menu.addMenu("View")
        viewAction = QAction("View", self)
        viewAction.triggered.connect(self.onViewClicked)
        viewMenu.addAction(viewAction)

    def setupUIComponents(self):
        layout = QVBoxLayout()
        self.searchBarWidget = SearchBar(self)
        layout.addWidget(self.searchBarWidget)
        self.searchBarWidget.textChanged.connect(self.filterImages)

        self.scroll = QScrollArea(self)
        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.container.setLayout(self.grid)
        self.scroll.setWidget(self.container)
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)

        self.progressBar = QProgressBar(self)
        layout.addWidget(self.progressBar)

        # Pagination controls
        self.paginationLayout = QHBoxLayout()
        self.prevButton = QPushButton("Previous")
        self.prevButton.clicked.connect(self.prevPage)
        self.paginationLayout.addWidget(self.prevButton)

        self.pageLabel = QLabel()
        self.paginationLayout.addWidget(self.pageLabel)

        self.nextButton = QPushButton("Next")
        self.nextButton.clicked.connect(self.nextPage)
        self.paginationLayout.addWidget(self.nextButton)

        layout.addLayout(self.paginationLayout)

        centralWidget = QWidget(self)
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        self.setPaginationControlsVisible(False)  # Hide pagination controls initially

    def setPaginationControlsVisible(self, visible):
        self.prevButton.setVisible(visible)
        self.pageLabel.setVisible(visible)
        self.nextButton.setVisible(visible)

    def onViewClicked(self):
        self.displayImages()

    def openFileDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open CSV file", "", "CSV Files (*.csv)")
        if fileName:
            self.csv_file = fileName
            self.processWindow = ProcessingWindow(fileName, self)
            self.processWindow.thread.dataLoaded.connect(self.storeImages)
            self.processWindow.thread.errorOccurred.connect(self.handleError)
            self.processWindow.show()

    def storeImages(self, all_images, unique_images):
        print("Storing images...")
        # Convert image paths to QImage objects
        self.images = [(label, QImage(image_path)) for label, image_path in all_images]
        self.filtered_images = self.images  # Initialize filtered images to all images
        self.labels = [img[0] for img in all_images]  # Extract labels
        self.processWindow = None  # Clean up
        self.current_page = 0  # Reset to the first page
        self.updatePagination()
        print(f"Stored {len(self.images)} images.")

    def displayImages(self):
        print("Displaying images...")
        if not self.filtered_images:
            QMessageBox.information(self, "Error", "No images to display.")
            return

        # Clear previous grid
        for i in range(self.grid.count()):
            widget = self.grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        start_index = self.current_page * self.images_per_page
        end_index = min(start_index + self.images_per_page, len(self.filtered_images))
        for index in range(start_index, end_index):
            label, qimage = self.filtered_images[index]
            if isinstance(qimage, QImage):
                pixmap = QPixmap.fromImage(qimage)
                pixmap = pixmap.scaled(100, 100, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                row, col = divmod(index - start_index, 6)
                self.addImageToGrid(pixmap, row, col, label)
            else:
                print(f"Skipping non-QImage object at index {index}")

        self.updatePagination()

    def updatePagination(self):
        total_pages = (len(self.filtered_images) + self.images_per_page - 1) // self.images_per_page
        self.pageLabel.setText(f"Page {self.current_page + 1} of {total_pages}")
        self.prevButton.setEnabled(self.current_page > 0)
        self.nextButton.setEnabled(self.current_page < total_pages - 1)
        self.setPaginationControlsVisible(total_pages > 1)  # Show pagination controls only if there are multiple pages

    def filterImages(self, searchText):
        searchText = searchText.lower()
        print(f"Filtering images with search text: '{searchText}'")
        if searchText == "":
            self.filtered_images = self.images
        elif searchText in self.search_cache:
            self.filtered_images = self.search_cache[searchText]
        else:
            self.filtered_images = [(label, qimage) for label, qimage in self.images if searchText in label.lower()]
            self.search_cache[searchText] = self.filtered_images
        print(f"Filtered down to {len(self.filtered_images)} images.")
        self.current_page = 0  # Reset to the first page
        self.displayImages()

    def addImageToGrid(self, pixmap, row, col, label):
        label_widget = QLabel(label)
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        image_label.setAlignment(QtCore.Qt.AlignCenter)
        image_label.mousePressEvent = lambda event, pix=pixmap, lbl=label: self.onImageClick(pix, lbl)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(image_label)
        layout.addWidget(label_widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        self.grid.addWidget(container, row, col)
        print(f"Added image to grid at row {row}, col {col}.")

    def onImageClick(self, pixmap, label):
        self.imageWindow = ImageWindow(pixmap, label)
        self.imageWindow.show()

    def prevPage(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.displayImages()

    def nextPage(self):
        total_pages = (len(self.filtered_images) + self.images_per_page - 1) // self.images_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.displayImages()

    def handleError(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.processWindow = None

    def openTrainPage(self):
        if self.images and self.labels:
            self.trainPage = Train(self.images, self.labels)
            self.trainPage.show()
            QMessageBox.information(self, "Training Initiated", "Training module has been opened with the selected images and labels.")
        else:
            QMessageBox.information(self, "Error", "No images loaded. Please load images first.")

    def openTestPage(self):
        print("Attempting to open test page...")
        if hasattr(self, 'cameraWindow') and self.cameraWindow.isVisible():
            print("Camera window is already visible.")
            self.cameraWindow.raise_()
        else:
            print("Creating new camera window...")
            from camerawindow import CameraWindow
            self.cameraWindow = CameraWindow(self)
            self.cameraWindow.show()
            print("Camera window should be open now.")

def main():
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
