import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'

import cv2
import numpy as np
import fitz
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    hand_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.last_x = None

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                # Convert to grayscale and blur
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (35, 35), 0)

                # Threshold the image
                ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Find the largest contour (assuming it's the hand)
                    max_contour = max(contours, key=cv2.contourArea)

                    # Get the rightmost point of the contour (hand edge)
                    rightmost = tuple(max_contour[max_contour[:, :, 0].argmax()][0])
                    cv2.circle(frame, rightmost, 5, (0, 0, 255), -1)

                    # Detect significant horizontal movement
                    if self.last_x is not None:
                        if rightmost[0] > self.last_x + 100:  # Increased threshold for whole hand movement
                            self.hand_signal.emit("right")
                        elif rightmost[0] < self.last_x - 100:
                            self.hand_signal.emit("left")

                    self.last_x = rightmost[0]

                    # Draw the largest contour
                    cv2.drawContours(frame, [max_contour], 0, (0, 255, 0), 2)

                # Convert the frame to RGB for Qt
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Reader with Hand Gesture Control")
        self.image_label = QLabel(self)
        self.pdf_label = QLabel(self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.pdf_label)
        self.setLayout(vbox)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.hand_signal.connect(self.handle_hand_movement)
        self.thread.start()

        self.pdf_document = fitz.open("pdf_files/book.pdf")
        self.current_page = 0
        self.update_pdf_display()

    def update_image(self, image):
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def handle_hand_movement(self, direction):
        if direction == "right" and self.current_page < len(self.pdf_document) - 1:
            self.current_page += 1
            self.update_pdf_display()
        elif direction == "left" and self.current_page > 0:
            self.current_page -= 1
            self.update_pdf_display()

    def update_pdf_display(self):
        logger.info(f"Displaying PDF page {self.current_page}...")
        page = self.pdf_document.load_page(self.current_page)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase scale for better resolution
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.pdf_label.setPixmap(pixmap)
        logger.info("PDF display updated.")

def main():
    logger.info("Starting the application...")
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()