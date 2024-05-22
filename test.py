# Assume this is in a new Python file, e.g., main.py
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QProgressBar, \
    QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from sympy import this
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pyqtgraph as pg
from models import SimpleDNN, CNNModel, ComplexDNN
from dataset import SignLanguageDataset
from training_thread import TrainingThread
from threading import Event
# Import the necessary classes and functions
from dataset import SignLanguageDataset  # Modify this line according to your file organization
from torchvision import transforms

# Define any transforms if necessary (optional)
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create an instance of the dastaset
dataset = SignLanguageDataset("C:\\Users\\rohan\\Documents\\Downloads\\sign_mnist_alpha_digits_test_train (1).csv",
                              transform=transformations)

# Access a specific item, for example, the 10th item
image, label = dataset[0]

# Now you can use 'image' and 'label' as needed, for example:
print(label)
image.show()
