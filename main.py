from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QWidget, QVBoxLayout, QLineEdit, QLabel, QFileDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("My App")
        self.setGeometry(100, 100, 728, 592)
        # self.setWindowIcon()  # Consider adding an icon if needed.

        # Create the main menu bar
        self.menu = self.menuBar()

        # 'File' menu with 'Load Data' and 'Test' submenu
        fileMenu = self.menu.addMenu("File")

        # 'Load Data' action that opens a file dialog for CSV files
        loadAction = QAction("Load Data", self)
        loadAction.triggered.connect(self.openFileDialog)
        fileMenu.addAction(loadAction)

        # 'Train' menu item as a non-interactive label
        trainAction = QAction("Train", self)
        trainAction.setEnabled(False)  # Make it non-interactive
        fileMenu.addAction(trainAction)

        # 'Test' menu item
        testAction = QAction("Test", self)
        fileMenu.addAction(testAction)

        # 'View' menu
        viewMenu = self.menu.addMenu("View")
        viewAction = QAction("View", self)
        viewMenu.addAction(viewAction)

        # Main Widget Setup
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
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

    def openFileDialog(self):
        # Function to open file dialog and filter for CSV files
        fileName, _ = QFileDialog.getOpenFileName(self, "Open CSV file", "", "CSV Files (*.csv)")
        if fileName:
            print(f"File selected: {fileName}")  # For debug purposes or further processing

def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()

if __name__ == '__main__':
    main()
