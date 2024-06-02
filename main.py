# main.py
from PyQt5.QtWidgets import QApplication
from mainWindow import MainWindow


def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()

if __name__ == '__main__':
    main()
