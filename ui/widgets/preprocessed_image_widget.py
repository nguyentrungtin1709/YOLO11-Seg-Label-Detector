"""
Preprocessed Image Widget

Widget for displaying preprocessed (cropped, rotated, orientation-fixed) label images.
"""

import logging
from typing import Optional
import numpy as np
import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QGroupBox


logger = logging.getLogger(__name__)


class PreprocessedImageWidget(QWidget):
    """
    Widget for displaying OCR input image (merged components).
    
    Shows the result of component extraction (S6):
    - Merged text components from the label
    - This is the exact image that will be sent to OCR (S7)
    - Includes preprocessing (S3) and enhancement (S4) if enabled
    
    Features:
    - Auto-scaling to fit widget while maintaining aspect ratio
    - Placeholder text when no image is available
    - Dark theme styling consistent with app
    """
    
    def __init__(
        self,
        parent=None,
        displayWidth: int = 230,
        displayHeight: int = 100
    ):
        """
        Initialize PreprocessedImageWidget.
        
        Args:
            parent: Parent widget.
            displayWidth: Maximum display width for the image.
            displayHeight: Maximum display height for the image.
        """
        super().__init__(parent)
        
        self._displayWidth = displayWidth
        self._displayHeight = displayHeight
        self._currentImage: Optional[np.ndarray] = None
        
        self._setupUI()
    
    def _setupUI(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Group box with title
        groupBox = QGroupBox("OCR Input (Merged Components)")
        groupLayout = QVBoxLayout(groupBox)
        groupLayout.setContentsMargins(5, 10, 5, 5)
        
        # Image display label
        self._imageLabel = QLabel()
        self._imageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._imageLabel.setMinimumSize(self._displayWidth, self._displayHeight)
        self._imageLabel.setMaximumHeight(self._displayHeight + 20)
        self._imageLabel.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 1px solid #444444;
                border-radius: 5px;
                color: #666666;
                font-size: 11px;
            }
        """)
        
        # Show placeholder
        self._showPlaceholder()
        
        groupLayout.addWidget(self._imageLabel)
        layout.addWidget(groupBox)
    
    def _showPlaceholder(self):
        """Show placeholder text when no image is available."""
        self._imageLabel.setText("No label detected")
        self._imageLabel.setPixmap(QPixmap())  # Clear any existing pixmap
    
    def updateImage(self, image: np.ndarray) -> None:
        """
        Update the displayed image.
        
        Args:
            image: Merged component image for OCR (BGR format from OpenCV).
        """
        if image is None:
            self.clear()
            return
        
        try:
            self._currentImage = image.copy()
            
            # Get image dimensions
            h, w = image.shape[:2]
            
            # Calculate scaling to fit within display bounds while maintaining aspect ratio
            scaleW = self._displayWidth / w
            scaleH = self._displayHeight / h
            scale = min(scaleW, scaleH, 1.0)  # Don't upscale
            
            # Resize if needed
            if scale < 1.0:
                newW = int(w * scale)
                newH = int(h * scale)
                displayImage = cv2.resize(image, (newW, newH), interpolation=cv2.INTER_AREA)
            else:
                displayImage = image
            
            # Convert BGR to RGB for Qt
            rgbImage = cv2.cvtColor(displayImage, cv2.COLOR_BGR2RGB)
            
            # Create QImage
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            qImage = QImage(
                rgbImage.data,
                w,
                h,
                bytesPerLine,
                QImage.Format.Format_RGB888
            )
            
            # Convert to QPixmap and display
            pixmap = QPixmap.fromImage(qImage)
            self._imageLabel.setPixmap(pixmap)
            self._imageLabel.setText("")  # Clear placeholder text
            
        except Exception as e:
            logger.error(f"Error updating preprocessed image: {e}")
            self.clear()
    
    def clear(self) -> None:
        """Clear the displayed image and show placeholder."""
        self._currentImage = None
        self._showPlaceholder()
    
    def getCurrentImage(self) -> Optional[np.ndarray]:
        """
        Get the currently displayed image.
        
        Returns:
            Current image as numpy array (BGR), or None if no image.
        """
        return self._currentImage.copy() if self._currentImage is not None else None
    
    def setDisplaySize(self, width: int, height: int) -> None:
        """
        Set the display size for the image.
        
        Args:
            width: Maximum display width.
            height: Maximum display height.
        """
        self._displayWidth = width
        self._displayHeight = height
        self._imageLabel.setMinimumSize(width, height)
        self._imageLabel.setMaximumHeight(height + 20)
        
        # Re-display current image with new size
        if self._currentImage is not None:
            self.updateImage(self._currentImage)
