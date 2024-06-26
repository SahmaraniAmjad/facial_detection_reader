import cv2
import dlib
import fitz
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QScrollArea
from PyQt5.QtCore import QThread, Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    scroll_signal = pyqtSignal(bool)

    def __init__(self, detector, predictor):
        super().__init__()
        self.detector = detector
        self.predictor = predictor
        self.last_y = None
        self.scroll_threshold = 10  # Adjust this value to change sensitivity

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                for face in faces:
                    landmarks = self.predictor(gray, face)
                    
                    # Get the y-coordinate of the nose tip (landmark point 30)
                    nose_y = landmarks.part(30).y

                    # Draw green rectangle around the face
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Check for face movement
                    if self.last_y is not None:
                        delta_y = nose_y - self.last_y
                        if delta_y > self.scroll_threshold:
                            self.scroll_signal.emit(True)  # Start scrolling
                        else:
                            self.scroll_signal.emit(False)  # Stop scrolling
                    
                    self.last_y = nose_y

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)

# Function to initialize facial landmark detector
def initialize_face_detector():
    logger.info("Initializing face detector...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    logger.info("Face detector initialized.")
    return detector, predictor

class App(QWidget):
    def __init__(self, detector, predictor):
        super().__init__()
        self.setWindowTitle("PDF Reader with Gesture Control")
        self.image_label = QLabel(self)
        self.pdf_scroll_area = QScrollArea(self)
        self.pdf_content = QLabel(self)
        self.pdf_scroll_area.setWidget(self.pdf_content)
        self.pdf_scroll_area.setWidgetResizable(True)
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.pdf_scroll_area)
        self.setLayout(vbox)

        self.thread = VideoThread(detector, predictor)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.scroll_signal.connect(self.handle_scroll_signal)
        self.thread.start()

        self.pdf_document = fitz.open("pdf_files/book.pdf")
        self.current_page = 0
        self.update_pdf_display()

        self.scroll_speed = 0
        self.scroll_timer = QTimer(self)
        self.scroll_timer.timeout.connect(self.scroll_pdf)
        self.scroll_timer.start(50)  # Adjust this value to change scroll speed (in milliseconds)

    def update_image(self, image):
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def handle_scroll_signal(self, should_scroll):
        if should_scroll:
            self.scroll_speed = 30  # Adjust this value to change scroll speed
        else:
            self.scroll_speed = 0

    def scroll_pdf(self):
        v_bar = self.pdf_scroll_area.verticalScrollBar()
        new_value = int(v_bar.value() + self.scroll_speed)

        if new_value >= v_bar.maximum() and self.current_page < len(self.pdf_document) - 1:
            self.current_page += 1
            self.update_pdf_display()
            v_bar.setValue(0)
        elif new_value <= 0 and self.current_page > 0:
            self.current_page -= 1
            self.update_pdf_display()
            v_bar.setValue(v_bar.maximum())
        else:
            v_bar.setValue(new_value)

    def update_pdf_display(self):
        logger.info(f"Displaying PDF page {self.current_page}...")
        page = self.pdf_document.load_page(self.current_page)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase scale for better resolution
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.pdf_content.setPixmap(pixmap)
        logger.info("PDF display updated.")

# Main function to create the GUI and start the application
def main():
    logger.info("Initializing face detector...")
    detector, predictor = initialize_face_detector()

    logger.info("Initializing GUI...")
    app = QApplication(sys.argv)
    ex = App(detector, predictor)
    ex.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    logger.info("Starting the application...")
    main()
    logger.info("Application ended.")