from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont

def main():
    app = QApplication([])
    window = QWidget()
    window.setGeometry(500,500,500,500)
    window.setWindowTitle("My App")

    b1 = QPushButton(window)
    b1.setText("Click Me")
    b1.clicked.connect(QCoreApplication.instance().quit)

    window.show()
    app.exec_()


if __name__ == '__main__':
    main()