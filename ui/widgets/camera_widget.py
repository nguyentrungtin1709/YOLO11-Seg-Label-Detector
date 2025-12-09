"""
Camera Widget

Widget for displaying camera feed with detection overlays.
"""

import logging
from typing import Optional
import numpy as np
import cv2

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout

from core.interfaces.detector_interface import Detection


logger = logging.getLogger(__name__)


class CameraWidget(QWidget):
    """
    Widget for displaying camera feed.
    
    Displays video frames from camera and overlays detection results.
    Supports segmentation mask rendering with configurable opacity.
    Emits frameReady signal for each processed frame.
    """
    
    frameReady = Signal(np.ndarray)  # Emitted when a new frame is ready
    
    def __init__(
        self,
        parent=None,
        boxColor: tuple[int, int, int] = (0, 255, 0),
        textColor: tuple[int, int, int] = (255, 255, 255),
        lineThickness: int = 2,
        fontSize: float = 0.6,
        maskOpacity: float = 0.4,
        maskColors: list[tuple[int, int, int]] = None
    ):
        """
        Initialize CameraWidget.
        
        Args:
            parent: Parent widget.
            boxColor: Color for detection boxes (BGR).
            textColor: Color for text labels (BGR).
            lineThickness: Thickness of detection boxes.
            fontSize: Font scale for labels.
            maskOpacity: Opacity for segmentation masks (0.0-1.0).
            maskColors: List of colors for masks (BGR format).
        """
        super().__init__(parent)
        
        self._boxColor = boxColor
        self._textColor = textColor
        self._lineThickness = lineThickness
        self._fontSize = fontSize
        self._maskOpacity = max(0.0, min(1.0, maskOpacity))
        
        self._debugMode = False
        self._currentFrame: Optional[np.ndarray] = None
        self._detections: list[Detection] = []
        
        # Use provided mask colors or default palette (BGR format)
        if maskColors:
            self._maskColors = [tuple(c) for c in maskColors]
        else:
            self._maskColors = [
                (128, 0, 128),   # Purple
                (0, 128, 255),   # Orange
                (255, 0, 0),     # Blue
                (0, 0, 255),     # Red
                (255, 255, 0),   # Cyan
                (255, 0, 255),   # Magenta
                (0, 255, 255),   # Yellow
                (128, 255, 0),   # Light green
                (255, 128, 0),   # Light blue
                (0, 255, 0),     # Green
            ]
        
        self._setupUI()
    
    def _setupUI(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Video display label
        self._videoLabel = QLabel()
        self._videoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._videoLabel.setStyleSheet("background-color: #1a1a1a;")
        self._videoLabel.setMinimumSize(640, 640)
        layout.addWidget(self._videoLabel)
        
        # Placeholder text
        self._videoLabel.setText("Camera not connected")
        self._videoLabel.setStyleSheet(
            "background-color: #1a1a1a; color: #666666; font-size: 16px;"
        )
    
    def updateFrame(self, frame: np.ndarray, detections: list[Detection] = None):
        """
        Update displayed frame with optional detections.
        
        Args:
            frame: Camera frame (BGR format).
            detections: List of detections to overlay.
        """
        self._currentFrame = frame.copy()
        self._detections = detections or []
        
        # Draw detections on frame
        displayFrame = self._drawOverlay(frame.copy(), self._detections)
        
        # Convert BGR to RGB for Qt
        rgbFrame = cv2.cvtColor(displayFrame, cv2.COLOR_BGR2RGB)
        
        # Create QImage
        height, width, channels = rgbFrame.shape
        bytesPerLine = channels * width
        qImage = QImage(
            rgbFrame.data,
            width,
            height,
            bytesPerLine,
            QImage.Format.Format_RGB888
        )
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qImage)
        scaledPixmap = pixmap.scaled(
            self._videoLabel.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self._videoLabel.setPixmap(scaledPixmap)
        self.frameReady.emit(self._currentFrame)
    
    def _drawOverlay(
        self,
        frame: np.ndarray,
        detections: list[Detection]
    ) -> np.ndarray:
        """
        Draw detection overlays on frame.
        Draws masks first (if available), then bounding boxes and labels on top.
        
        Args:
            frame: Frame to draw on.
            detections: Detections to draw.
            
        Returns:
            Frame with overlays.
        """
        # First pass: Draw all masks (so they appear behind boxes and labels)
        for idx, det in enumerate(detections):
            if det.mask is not None:
                frame = self._drawMask(frame, det.mask, idx)
        
        # Second pass: Draw bounding boxes and labels
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                self._boxColor,
                self._lineThickness
            )
            
            # Prepare label text
            label = f"{det.className}: {det.confidence:.2f}"
            
            # Calculate text size for background
            (textWidth, textHeight), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self._fontSize,
                1
            )
            
            # Draw text background
            cv2.rectangle(
                frame,
                (x1, y1 - textHeight - 10),
                (x1 + textWidth + 10, y1),
                self._boxColor,
                -1
            )
            
            # Draw text label
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._fontSize,
                self._textColor,
                1,
                cv2.LINE_AA
            )
        
        # Draw debug info if enabled
        if self._debugMode:
            frame = self._drawDebugInfo(frame, detections)
        
        return frame
    
    def _drawMask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        colorIndex: int = 0
    ) -> np.ndarray:
        """
        Draw a segmentation mask on frame with transparency.
        
        Args:
            frame: Frame to draw on (will be modified).
            mask: Binary mask (H x W) with values 0 or 255.
            colorIndex: Index into mask colors palette.
            
        Returns:
            Frame with mask overlay.
        """
        if mask is None:
            return frame
        
        try:
            # Get color for this mask
            color = self._maskColors[colorIndex % len(self._maskColors)]
            
            # Create colored mask overlay (full color where mask is active)
            coloredMask = np.zeros_like(frame, dtype=np.uint8)
            
            # Apply color where mask is active
            maskBool = mask > 127  # Binary threshold
            coloredMask[maskBool] = color
            
            # Proper alpha blending: result = frame * (1 - alpha) + coloredMask * alpha
            # This gives true transparency effect
            blended = cv2.addWeighted(
                frame, 
                1.0 - self._maskOpacity,  # Background weight
                coloredMask, 
                self._maskOpacity,         # Mask weight
                0
            )
            
            # Only apply blending where mask is active
            frame[maskBool] = blended[maskBool]
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to draw mask: {e}")
            return frame
    
    def _drawDebugInfo(
        self,
        frame: np.ndarray,
        detections: list[Detection]
    ) -> np.ndarray:
        """
        Draw debug information on frame.
        
        Args:
            frame: Frame to draw on.
            detections: Current detections.
            
        Returns:
            Frame with debug info.
        """
        height, width = frame.shape[:2]
        
        # Debug panel background
        panelHeight = 80
        cv2.rectangle(frame, (0, 0), (width, panelHeight), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (width, panelHeight), (50, 50, 50), 1)
        
        # Debug text
        debugLines = [
            f"Frame Size: {width}x{height}",
            f"Detections: {len(detections)}",
            f"Mode: Debug"
        ]
        
        y = 25
        for line in debugLines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )
            y += 25
        
        return frame
    
    def setDebugMode(self, enabled: bool):
        """
        Enable or disable debug mode.
        
        Args:
            enabled: True to enable debug overlay.
        """
        self._debugMode = enabled
        logger.debug(f"Debug mode {'enabled' if enabled else 'disabled'}")
    
    def isDebugMode(self) -> bool:
        """Get debug mode state."""
        return self._debugMode
    
    def getCurrentFrame(self) -> Optional[np.ndarray]:
        """Get the current frame (without overlays)."""
        return self._currentFrame
    
    def getCurrentDetections(self) -> list[Detection]:
        """Get current detections."""
        return self._detections
    
    def showPlaceholder(self, message: str = "Camera not connected"):
        """
        Show placeholder text instead of video.
        
        Args:
            message: Placeholder message to display.
        """
        self._videoLabel.clear()
        self._videoLabel.setText(message)
        self._videoLabel.setStyleSheet(
            "background-color: #1a1a1a; color: #666666; font-size: 16px;"
        )
        self._currentFrame = None
        self._detections = []
