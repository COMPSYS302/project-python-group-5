# mainwindow.py
from PyQt5.QtWidgets import QMainWindow, QAction, QFileDialog
from UIcomponents import ImageArea, SearchBar

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("My App")
        self.setGeometry(100, 100, 728, 592)

        # Create the main menu bar
        self.menu = self.menuBar()
        self.setupMenus()

        # Main Widget Setup
        self.setupCentralWidget()

    def setupMenus(self):
        # File menu
        fileMenu = self.menu.addMenu("File")

        # Load Data action
        loadAction = QAction("Load Data", self)
        loadAction.triggered.connect(self.openFileDialog)
        fileMenu.addAction(loadAction)

        # Train action (non-interactive label)
        trainAction = QAction("Train", self)
        trainAction.setEnabled(False)
        fileMenu.addAction(trainAction)

        # Test action
        testAction = QAction("Test", self)
        fileMenu.addAction(testAction)

        # View menu
        viewMenu = self.menu.addMenu("View")
        viewAction = QAction("View", self)
        viewMenu.addAction(viewAction)

    def setupCentralWidget(self):
        mainWidget = SearchBar(self)
        self.setCentralWidget(mainWidget)

    def openFileDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open CSV file", "", "CSV Files (*.csv)")
        if fileName:
            print(f"File selected: {fileName}")

