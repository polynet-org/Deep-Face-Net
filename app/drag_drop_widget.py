from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
import os

class DragDropWidget(QLabel):
    """
    A widget that accepts file drag and drop events.
    Emits file_dropped signal with the file path when a valid file is dropped.
    """
    file_dropped = pyqtSignal(str)

    def __init__(self, text="Drag & Drop Video/Image Here", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #666;
                border-radius: 10px;
                background-color: #252525;
                color: #aaa;
                font-size: 14px;
                padding: 20px;
            }
            QLabel:hover {
                border-color: #4CAF50;
                background-color: #2d2d2d;
                color: #fff;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            # Check if at least one file is supported (optional check, can be done later)
            event.accept()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #4CAF50;
                    border-radius: 10px;
                    background-color: #2d3d2d;
                    color: #fff;
                    font-size: 14px;
                    padding: 20px;
                }
            """)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Reset style when drag leaves"""
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #666;
                border-radius: 10px;
                background-color: #252525;
                color: #aaa;
                font-size: 14px;
                padding: 20px;
            }
            QLabel:hover {
                border-color: #4CAF50;
                background-color: #2d2d2d;
                color: #fff;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        """Handle drop event"""
        urls = event.mimeData().urls()
        if urls:
            # Take the first file
            file_path = urls[0].toLocalFile()
            if os.path.exists(file_path):
                self.file_dropped.emit(file_path)
                self.setText(f"Selected: {os.path.basename(file_path)}")
                self.setStyleSheet("""
                    QLabel {
                        border: 2px solid #4CAF50;
                        border-radius: 10px;
                        background-color: #252525;
                        color: #4CAF50;
                        font-weight: bold;
                        font-size: 14px;
                        padding: 20px;
                    }
                """)
