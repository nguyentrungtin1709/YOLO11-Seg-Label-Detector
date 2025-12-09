"""
Toggle Switch Widget

A custom toggle switch widget for PySide6.
Provides iOS-style on/off toggle.
"""

from PySide6.QtCore import Qt, Property, QPropertyAnimation, QEasingCurve, Signal
from PySide6.QtGui import QPainter, QColor, QPainterPath
from PySide6.QtWidgets import QWidget


class ToggleSwitch(QWidget):
    """
    iOS-style toggle switch widget.
    
    Emits toggled signal when state changes.
    """
    
    toggled = Signal(bool)
    
    def __init__(
        self,
        parent=None,
        width: int = 50,
        height: int = 26,
        bgColorOff: str = "#cccccc",
        bgColorOn: str = "#4CAF50",
        handleColor: str = "#ffffff"
    ):
        """
        Initialize ToggleSwitch.
        
        Args:
            parent: Parent widget.
            width: Switch width.
            height: Switch height.
            bgColorOff: Background color when off.
            bgColorOn: Background color when on.
            handleColor: Handle (circle) color.
        """
        super().__init__(parent)
        
        self.setFixedSize(width, height)
        self.setCursor(Qt.PointingHandCursor)
        
        self._isChecked = False
        self._handlePosition = 3  # Left margin when off
        
        # Colors
        self._bgColorOff = QColor(bgColorOff)
        self._bgColorOn = QColor(bgColorOn)
        self._handleColor = QColor(handleColor)
        
        # Animation
        self._animation = QPropertyAnimation(self, b"handlePosition", self)
        self._animation.setDuration(150)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    
    def paintEvent(self, event):
        """Paint the toggle switch."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        bgColor = self._bgColorOn if self._isChecked else self._bgColorOff
        painter.setBrush(bgColor)
        painter.setPen(Qt.PenStyle.NoPen)
        
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 
                           self.height() / 2, self.height() / 2)
        painter.drawPath(path)
        
        # Draw handle
        handleDiameter = self.height() - 6
        painter.setBrush(self._handleColor)
        painter.drawEllipse(
            int(self._handlePosition),
            3,
            handleDiameter,
            handleDiameter
        )
    
    def mousePressEvent(self, event):
        """Handle mouse press to toggle."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle()
    
    def toggle(self):
        """Toggle the switch state."""
        self.setChecked(not self._isChecked)
    
    def isChecked(self) -> bool:
        """Get checked state."""
        return self._isChecked
    
    def setChecked(self, checked: bool):
        """
        Set checked state.
        
        Args:
            checked: New checked state.
        """
        if self._isChecked == checked:
            return
        
        self._isChecked = checked
        
        # Animate handle position
        handleDiameter = self.height() - 6
        startPos = 3 if not checked else self.width() - handleDiameter - 3
        endPos = self.width() - handleDiameter - 3 if checked else 3
        
        self._animation.stop()
        self._animation.setStartValue(startPos)
        self._animation.setEndValue(endPos)
        self._animation.start()
        
        self.toggled.emit(self._isChecked)
    
    @Property(float)
    def handlePosition(self) -> float:
        """Get handle position for animation."""
        return self._handlePosition
    
    @handlePosition.setter
    def handlePosition(self, pos: float):
        """Set handle position for animation."""
        self._handlePosition = pos
        self.update()
