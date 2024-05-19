from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QAction, QGridLayout, QWidget, QLabel, QVBoxLayout, QLineEdit

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("My App")
        self.setGeometry(100, 100, 728, 592)  # Size adjusted to mimic your screenshot


        # Create Menu
        self.menu = self.menuBar()
        loadMenu = self.menu.addMenu("Load Data")
        viewMenu = self.menu.addMenu("View")
        trainMenu = self.menu.addMenu("Train")
        testMenu = self.menu.addMenu("Test")

        # Dummy Actions
        loadAction = QAction("Load", self)
        viewAction = QAction("View", self)
        trainAction = QAction("Train", self)
        testAction = QAction("Test", self)

        loadMenu.addAction(loadAction)
        viewMenu.addAction(viewAction)
        trainMenu.addAction(trainAction)
        testMenu.addAction(testAction)

        # Main Widget
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)

        # Layout for central widget
        layout = QVBoxLayout(mainWidget)

        # Search Bar
        searchBar = QLineEdit()
        searchBar.setPlaceholderText("Search...")
        layout.addWidget(searchBar)

        # Placeholder for image area
        imageArea = QLabel("Image Area")
        imageArea.setWordWrap(True)
        layout.addWidget(imageArea)

        # Set layout to the main widget
        mainWidget.setLayout(layout)

def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()

if __name__ == '__main__':
    main()
