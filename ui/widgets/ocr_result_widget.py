"""
OCR Result Widget.

This widget displays the final OCR extraction results in a structured format.
Shows QR code data, extracted text fields, and validation status.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGroupBox, QGridLayout, QFrame
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont

from core.interfaces.text_processor_interface import LabelData


class OcrResultWidget(QWidget):
    """
    Widget to display OCR extraction results.
    
    Shows:
    - QR Code information (order code, facility, position)
    - OCR extracted fields (product, size, color, position/quantity)
    - Validation status
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._initUi()
    
    def _initUi(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title
        titleLabel = QLabel("OCR Results")
        titleLabel.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        titleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(titleLabel)
        
        # QR Code Group
        qrGroup = QGroupBox("QR Code")
        qrLayout = QGridLayout(qrGroup)
        qrLayout.setSpacing(4)
        
        self._qrOrderLabel = self._createValueLabel()
        self._qrFacilityLabel = self._createValueLabel()
        self._qrPositionLabel = self._createValueLabel()
        
        qrLayout.addWidget(QLabel("Order:"), 0, 0)
        qrLayout.addWidget(self._qrOrderLabel, 0, 1)
        qrLayout.addWidget(QLabel("Facility:"), 1, 0)
        qrLayout.addWidget(self._qrFacilityLabel, 1, 1)
        qrLayout.addWidget(QLabel("Position:"), 2, 0)
        qrLayout.addWidget(self._qrPositionLabel, 2, 1)
        
        layout.addWidget(qrGroup)
        
        # OCR Fields Group
        ocrGroup = QGroupBox("Extracted Fields")
        ocrLayout = QGridLayout(ocrGroup)
        ocrLayout.setSpacing(4)
        
        self._productLabel = self._createValueLabel()
        self._sizeLabel = self._createValueLabel()
        self._colorLabel = self._createValueLabel()
        self._posQtyLabel = self._createValueLabel()
        
        ocrLayout.addWidget(QLabel("Product:"), 0, 0)
        ocrLayout.addWidget(self._productLabel, 0, 1)
        ocrLayout.addWidget(QLabel("Size:"), 1, 0)
        ocrLayout.addWidget(self._sizeLabel, 1, 1)
        ocrLayout.addWidget(QLabel("Color:"), 2, 0)
        ocrLayout.addWidget(self._colorLabel, 2, 1)
        ocrLayout.addWidget(QLabel("Pos/Qty:"), 3, 0)
        ocrLayout.addWidget(self._posQtyLabel, 3, 1)
        
        layout.addWidget(ocrGroup)
        
        # Validation Status
        statusFrame = QFrame()
        statusFrame.setFrameStyle(QFrame.Shape.StyledPanel)
        statusLayout = QHBoxLayout(statusFrame)
        statusLayout.setContentsMargins(5, 5, 5, 5)
        
        statusLayout.addWidget(QLabel("Status:"))
        self._statusLabel = QLabel("--")
        self._statusLabel.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        statusLayout.addWidget(self._statusLabel)
        statusLayout.addStretch()
        
        layout.addWidget(statusFrame)
        
        # Processing time
        self._timeLabel = QLabel("Time: --")
        self._timeLabel.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._timeLabel.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self._timeLabel)
        
        layout.addStretch()
        
        # Initial state
        self.clear()
    
    def _createValueLabel(self) -> QLabel:
        """Create a styled value label."""
        label = QLabel("--")
        label.setFont(QFont("Consolas", 10))
        label.setStyleSheet("color: #333; background-color: #f5f5f5; padding: 2px 5px;")
        label.setWordWrap(True)
        return label
    
    @Slot()
    def clear(self) -> None:
        """Clear all displayed values."""
        self._qrOrderLabel.setText("--")
        self._qrFacilityLabel.setText("--")
        self._qrPositionLabel.setText("--")
        self._productLabel.setText("--")
        self._sizeLabel.setText("--")
        self._colorLabel.setText("--")
        self._posQtyLabel.setText("--")
        self._statusLabel.setText("--")
        self._statusLabel.setStyleSheet("color: gray;")
        self._timeLabel.setText("Time: --")
    
    @Slot(object, float)
    def updateResult(self, labelData: LabelData, processingTimeMs: float) -> None:
        """
        Update the widget with new OCR results.
        
        Args:
            labelData: Structured label data from OCR pipeline
            processingTimeMs: Processing time in milliseconds
        """
        if labelData is None:
            self.clear()
            return
        
        # Update QR Code fields
        self._qrOrderLabel.setText(
            f"{labelData.orderType}-{labelData.orderNumber}" 
            if labelData.orderNumber else "--"
        )
        self._qrFacilityLabel.setText(labelData.facility or "--")
        self._qrPositionLabel.setText(str(labelData.qrPosition) if labelData.qrPosition else "--")
        
        # Update OCR fields
        self._productLabel.setText(labelData.productCode or "--")
        self._sizeLabel.setText(labelData.size or "--")
        self._colorLabel.setText(labelData.color or "--")
        self._posQtyLabel.setText(labelData.positionQuantity or "--")
        
        # Update validation status
        if labelData.isValid:
            self._statusLabel.setText("✓ VALID")
            self._statusLabel.setStyleSheet(
                "color: white; background-color: #28a745; padding: 2px 8px; border-radius: 3px;"
            )
        else:
            self._statusLabel.setText("✗ INVALID")
            self._statusLabel.setStyleSheet(
                "color: white; background-color: #dc3545; padding: 2px 8px; border-radius: 3px;"
            )
        
        # Update processing time
        self._timeLabel.setText(f"Time: {processingTimeMs:.1f}ms")
    
    @Slot(str)
    def showError(self, message: str) -> None:
        """
        Display an error message.
        
        Args:
            message: Error message to display
        """
        self.clear()
        self._statusLabel.setText(f"⚠ {message}")
        self._statusLabel.setStyleSheet(
            "color: white; background-color: #ffc107; padding: 2px 8px; border-radius: 3px;"
        )
