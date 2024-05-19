# mainwindow.py
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog, QScrollArea, QVBoxLayout, QWidget, QApplication, QMessageBox, QGridLayout
from PyQt5.QtGui import QPixmap
from processingwindow import ProcessingWindow
from ui_components import SearchBar

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.images = []

    def initUI(self):
        self.setWindowTitle("Image Viewer App")
        self.setGeometry(100, 100, 800, 600)
        self.menu = self.menuBar()
        self.setupMenus()
        self.searchBarWidget = None

    def setupMenus(self):
        fileMenu = self.menu.addMenu("File")
        loadAction = QAction("Load Data", self)
        loadAction.triggered.connect(self.openFileDialog)
        fileMenu.addAction(loadAction)

        viewMenu = self.menu.addMenu("View")
        viewAction = QAction("View", self)
        viewAction.triggered.connect(self.displayImages)
        viewMenu.addAction(viewAction)

    def openFileDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open CSV file", "", "CSV Files (*.csv)")
        if fileName:
            self.processWindow = ProcessingWindow(fileName, self)
            self.processWindow.thread.dataLoaded.connect(self.storeImages)
            self.processWindow.show()

    def storeImages(self, images):
        self.images = images

    def displayImages(self):
        if not self.images:
            QMessageBox.information(self, "Error", "No images to display.")
            return

        if not self.searchBarWidget:
            self.searchBarWidget = SearchBar(self)
            self.setCentralWidget(self.searchBarWidget)
            self.searchBarWidget.show()

        # Setup the scroll area and grid layout for images below the search bar
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
            if col >= 6:  # Ensure only four images per row
                col = 0
                row += 1

        container.setLayout(grid)
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)

        # Adding the scroll area to the layout of the search bar widget
        self.searchBarWidget.layout.addWidget(scroll)

def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()

if __name__ == '__main__':
    main()
