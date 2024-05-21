


import sys
import pandas as pd
from PyQt5 import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from app import code
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV to Image Viewer")
        self.filename = None
        # Central Widget Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Browse Button
        self.browse_button = QPushButton("Browse CSV")
        self.browse_button.clicked.connect(self.browse_csv_file)
        layout.addWidget(self.browse_button)

        # Label to Display Image
        self.image_label = QLabel()
        layout.addWidget(self.image_label)




    def display_image_based_on_csv(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))

    def browse_csv_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            result  = code.identify(self.filename)
            if result :
                self.display_image_based_on_csv(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
