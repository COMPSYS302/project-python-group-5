# mainwindow.py
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QLabel, QScrollArea, QVBoxLayout, QWidget, QApplication, QMessageBox, QGridLayout
from PyQt5.QtGui import QPixmap
from processingwindow import ProcessingWindow
from UIcomponents import SearchBar

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.images = []
        self.searchBarWidget = None

    def initUI(self):
        self.setWindowTitle("Image Viewer App")
        self.setGeometry(100, 100, 800, 600)
        self.menu = self.menuBar()
        self.setupMenus()

    def setupMenus(self):
        # File menu
        fileMenu = self.menu.addMenu("File")

        # Load Data action
        loadAction = QAction("Load Data", self)
        loadAction.triggered.connect(self.openFileDialog)
        fileMenu.addAction(loadAction)

        # Train Data action - No method connected
        trainDataAction = QAction("Train Data", self)
        trainDataAction.setEnabled(False)  # Makes the action disabled, i.e., it appears but cannot be clicked
        fileMenu.addAction(trainDataAction)

        # View menu
        viewMenu = self.menu.addMenu("View")
        viewAction = QAction("View", self)
        viewAction.triggered.connect(self.displayImages)
        viewMenu.addAction(viewAction)


    def openFileDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open CSV file", "", "CSV Files (*.csv)")
        if fileName:
            self.processWindow = ProcessingWindow(fileName, self)
            self.processWindow.thread.dataLoaded.connect(self.storeImages)
            self.processWindow.thread.errorOccurred.connect(self.handleError)
            self.processWindow.show()

    def storeImages(self, images):
        self.images = images
        self.processWindow = None  # Ensure to clean up

    def displayImages(self):
        if not self.images:
            QMessageBox.information(self, "Error", "No images to display.")
            return

        if not self.searchBarWidget:
            self.searchBarWidget = SearchBar(self)
            self.setCentralWidget(self.searchBarWidget)

        # Set up a scrollable area with a grid layout for images
        scroll = QScrollArea(self.searchBarWidget)
        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(10)

        row = 0
        col = 0
        for index, img in enumerate(self.images):
            label = QLabel()
            pixmap = QPixmap.fromImage(img)
            label.setPixmap(pixmap)
            grid.addWidget(label, row, col)
            col += 1
            if col == 6:
                col = 0
                row += 1

        container.setLayout(grid)
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        self.searchBarWidget.layout.addWidget(scroll)

    def handleError(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.processWindow = None  # Clean up after error

def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()

if __name__ == '__main__':
    main()
