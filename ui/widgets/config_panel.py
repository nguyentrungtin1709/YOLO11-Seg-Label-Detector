"""
Config Panel Widget

Panel containing camera selection and detection controls.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QDoubleSpinBox, QPushButton, QGroupBox
)

from ui.widgets.toggle_switch import ToggleSwitch
from core.interfaces.camera_interface import CameraInfo


class ConfigPanel(QWidget):
    """
    Configuration panel widget.
    
    Contains controls for camera selection, detection toggle,
    debug mode, confidence threshold, and capture button.
    """
    
    # Signals
    cameraChanged = Signal(int)
    cameraToggled = Signal(bool)  # New signal for camera on/off
    detectionToggled = Signal(bool)
    debugToggled = Signal(bool)
    confidenceChanged = Signal(float)
    captureRequested = Signal()
    closeRequested = Signal()  # New signal for close button
    
    def __init__(self, parent=None):
        """Initialize ConfigPanel."""
        super().__init__(parent)
        self._setupUI()
    
    def _setupUI(self):
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Camera Selection Group
        cameraGroup = QGroupBox("Camera")
        cameraLayout = QVBoxLayout(cameraGroup)
        
        self._cameraCombo = QComboBox()
        self._cameraCombo.currentIndexChanged.connect(self._onCameraChanged)
        cameraLayout.addWidget(self._cameraCombo)
        
        # Camera On/Off Toggle
        cameraToggleRow = QHBoxLayout()
        cameraToggleLabel = QLabel("Camera Power")
        self._cameraToggle = ToggleSwitch(bgColorOn="#FF5722")
        self._cameraToggle.toggled.connect(self._onCameraToggled)
        cameraToggleRow.addWidget(cameraToggleLabel)
        cameraToggleRow.addStretch()
        cameraToggleRow.addWidget(self._cameraToggle)
        cameraLayout.addLayout(cameraToggleRow)
        
        layout.addWidget(cameraGroup)
        
        # Detection Controls Group
        detectionGroup = QGroupBox("Detection")
        detectionLayout = QVBoxLayout(detectionGroup)
        
        # Detection Toggle
        toggleRow = QHBoxLayout()
        toggleLabel = QLabel("Enable Detection")
        self._detectionToggle = ToggleSwitch()
        self._detectionToggle.toggled.connect(self.detectionToggled)
        toggleRow.addWidget(toggleLabel)
        toggleRow.addStretch()
        toggleRow.addWidget(self._detectionToggle)
        detectionLayout.addLayout(toggleRow)
        
        # Debug Mode Toggle
        debugRow = QHBoxLayout()
        debugLabel = QLabel("Debug Mode")
        self._debugToggle = ToggleSwitch(bgColorOn="#2196F3")
        self._debugToggle.toggled.connect(self.debugToggled)
        debugRow.addWidget(debugLabel)
        debugRow.addStretch()
        debugRow.addWidget(self._debugToggle)
        detectionLayout.addLayout(debugRow)
        
        # Confidence Threshold
        confidenceRow = QHBoxLayout()
        confidenceLabel = QLabel("Confidence")
        self._confidenceSpinner = QDoubleSpinBox()
        self._confidenceSpinner.setRange(0.0, 1.0)
        self._confidenceSpinner.setSingleStep(0.05)
        self._confidenceSpinner.setDecimals(2)
        self._confidenceSpinner.setValue(0.5)
        self._confidenceSpinner.valueChanged.connect(self.confidenceChanged)
        confidenceRow.addWidget(confidenceLabel)
        confidenceRow.addStretch()
        confidenceRow.addWidget(self._confidenceSpinner)
        detectionLayout.addLayout(confidenceRow)
        
        layout.addWidget(detectionGroup)
        
        # Capture Button
        self._captureButton = QPushButton("📷 Capture Image")
        self._captureButton.setMinimumHeight(40)
        self._captureButton.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self._captureButton.clicked.connect(self.captureRequested)
        layout.addWidget(self._captureButton)
        
        # Close Button
        self._closeButton = QPushButton("✕ Close")
        self._closeButton.setMinimumHeight(40)
        self._closeButton.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        self._closeButton.clicked.connect(self.closeRequested)
        layout.addWidget(self._closeButton)
        
        # Spacer
        layout.addStretch()
    
    def _onCameraChanged(self, index: int):
        """Handle camera selection change."""
        if index >= 0:
            data = self._cameraCombo.currentData()
            if data is not None:
                self.cameraChanged.emit(data)
    
    def _onCameraToggled(self, enabled: bool):
        """Handle camera toggle."""
        self.cameraToggled.emit(enabled)
        # Enable/disable capture button based on camera state
        self._captureButton.setEnabled(enabled)
    
    def updateCameraList(self, cameras: list[CameraInfo]):
        """
        Update camera dropdown with available cameras.
        
        Args:
            cameras: List of available cameras.
        """
        self._cameraCombo.blockSignals(True)
        self._cameraCombo.clear()
        
        if not cameras:
            self._cameraCombo.addItem("No cameras found", -1)
        else:
            for camera in cameras:
                self._cameraCombo.addItem(camera.name, camera.index)
        
        self._cameraCombo.blockSignals(False)
    
    def setDetectionEnabled(self, enabled: bool):
        """Set detection toggle state."""
        self._detectionToggle.setChecked(enabled)
    
    def setDebugEnabled(self, enabled: bool):
        """Set debug toggle state."""
        self._debugToggle.setChecked(enabled)
    
    def setConfidenceThreshold(self, value: float):
        """Set confidence spinner value."""
        self._confidenceSpinner.setValue(value)
    
    def setCaptureEnabled(self, enabled: bool):
        """Enable or disable capture button."""
        self._captureButton.setEnabled(enabled)
    
    def getSelectedCameraIndex(self) -> int:
        """Get currently selected camera index."""
        data = self._cameraCombo.currentData()
        return data if data is not None else -1
    
    def isDetectionEnabled(self) -> bool:
        """Get detection toggle state."""
        return self._detectionToggle.isChecked()
    
    def isDebugEnabled(self) -> bool:
        """Get debug toggle state."""
        return self._debugToggle.isChecked()
    
    def getConfidenceThreshold(self) -> float:
        """Get confidence spinner value."""
        return self._confidenceSpinner.value()
    
    def setCameraEnabled(self, enabled: bool):
        """Set camera toggle state."""
        self._cameraToggle.setChecked(enabled)
    
    def isCameraEnabled(self) -> bool:
        """Get camera toggle state."""
        return self._cameraToggle.isChecked()
