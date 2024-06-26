import cv2
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

    def __init__(self):
        super().__init__()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.scroll_threshold = 0.7  # Adjust this value to change sensitivity

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Calculate the position of the bottom of the face
                    face_bottom = y + h
                    
                    # Check if the bottom of the face is near the bottom of the frame
                    if face_bottom > frame.shape[0] * self.scroll_threshold:
                        self.scroll_signal.emit(True)  # Start scrolling
                        cv2.putText(frame, "Scrolling", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        self.scroll_signal.emit(False)  # Stop scrolling

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Reader with Face Position Tracking")
        self.image_label = QLabel(self)
        self.pdf_scroll_area = QScrollArea(self)
        self.pdf_content = QLabel(self)
        self.pdf_scroll_area.setWidget(self.pdf_content)
        self.pdf_scroll_area.setWidgetResizable(True)
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.pdf_scroll_area)
        self.setLayout(vbox)

        self.thread = VideoThread()
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
            self.scroll_speed = 20  # Adjust this value to change scroll speed
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
    logger.info("Starting the application...")
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()