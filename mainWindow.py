from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QLabel, QScrollArea, QVBoxLayout, QWidget, QApplication, QMessageBox, QGridLayout, QProgressBar
from PyQt5.QtGui import QPixmap
from processingwindow import ProcessingWindow
from UIcomponents import SearchBar
from Train import Train

class ImageLoaderThread(QtCore.QThread):
    update_pixmap = QtCore.pyqtSignal(QPixmap, int, int)
    progress_updated = QtCore.pyqtSignal(int)

    def __init__(self, images, thumbnail_size):
        super().__init__()
        self.images = images
        self.thumbnail_size = thumbnail_size

    def run(self):
        total = len(self.images)
        for i, img in enumerate(self.images):
            pixmap = QPixmap.fromImage(img)
            pixmap = pixmap.scaled(self.thumbnail_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            col = i % 6
            row = i // 6
            self.update_pixmap.emit(pixmap, row, col)
            self.progress_updated.emit((i + 1) * 100 // total)
            self.msleep(10)

class ImageWindow(QMainWindow):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Prediction")

        self.setGeometry(100, 100, 400, 800)
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.setCentralWidget(self.imageLabel)
        self.displayImage(image)

    def displayImage(self, pixmap):
        self.imageLabel.setPixmap(pixmap)
    def plotProbabilities(self, probabilities):
        ax = self.figure.add_subplot(111)
        ax.bar(range(len(probabilities)), probabilities, tick_label=[str(i) for i in range(len(probabilities))])
        ax.set_title("Output Probabilities")
        ax.set_ylabel("Probability")
        self.canvas.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.images = []
        self.unique_images = []
        self.csv_file = ""
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Viewer App")
        self.setGeometry(100, 100, 800, 600)
        self.menu = self.menuBar()
        self.setupMenus()

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
        viewAction.triggered.connect(self.displayImages)
        viewMenu.addAction(viewAction)

    def openFileDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open CSV file", "", "CSV Files (*.csv)")
        if fileName:
            self.csv_file = fileName
            self.processWindow = ProcessingWindow(fileName, self)
            self.processWindow.thread.dataLoaded.connect(self.storeImages)
            self.processWindow.thread.errorOccurred.connect(self.handleError)
            self.processWindow.show()

    def storeImages(self, all_images, unique_images):
        self.images = all_images
        self.unique_images = unique_images
        self.processWindow = None  # Clean up

    def displayImages(self):
        if not self.unique_images:
            QMessageBox.information(self, "Error", "No unique images to display.")
            return

        layout = QVBoxLayout()
        self.searchBarWidget = SearchBar(self)
        layout.addWidget(self.searchBarWidget)

        self.scroll = QScrollArea(self)
        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.container.setLayout(self.grid)
        self.scroll.setWidget(self.container)
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)

        self.progressBar = QProgressBar(self)
        layout.addWidget(self.progressBar)

        centralWidget = QWidget(self)
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        self.loader_thread = ImageLoaderThread(self.unique_images, QtCore.QSize(75, 75))
        self.loader_thread.update_pixmap.connect(self.addImageToGrid, QtCore.Qt.QueuedConnection)
        self.loader_thread.progress_updated.connect(self.updateProgressBar)
        self.loader_thread.start()

    def addImageToGrid(self, pixmap, row, col):
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.mousePressEvent = lambda event, pix=pixmap: self.onImageClick(pix)
        self.grid.addWidget(label, row, col)

    def onImageClick(self, pixmap):
        self.imageWindow = ImageWindow(pixmap)
        self.imageWindow.show()

    def updateProgressBar(self, value):
        self.progressBar.setValue(value)

    def handleError(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.processWindow = None

    def openTrainPage(self):
        if self.csv_file:
            self.trainPage = Train(self.csv_file)
            self.trainPage.show()

        else:
            QMessageBox.information(self, "Error", "No CSV file loaded. Please load a CSV file first.")

    def openTestPage(self):
        if hasattr(self, 'trainPage') and self.trainPage.isVisible():
            self.trainPage.show()
        else:
            QMessageBox.information(self, "Error", "No train file loaded. Please train a CSV file first.")

def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()

if __name__ == '__main__':
    main()
