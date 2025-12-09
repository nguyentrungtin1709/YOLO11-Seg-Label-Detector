"""
Main Window

Main application window containing all UI components.
Orchestrates the interaction between widgets and services.
"""

import logging
import time
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStatusBar, QMessageBox
)

from ui.widgets.camera_widget import CameraWidget
from ui.widgets.config_panel import ConfigPanel
from services.camera_service import CameraService
from services.detection_service import DetectionService
from services.image_saver_service import ImageSaverService


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window.
    
    Contains camera view and configuration panel.
    Manages the main application loop and service interactions.
    """
    
    def __init__(
        self,
        cameraService: CameraService,
        detectionService: DetectionService,
        imageSaverService: ImageSaverService,
        config: dict
    ):
        """
        Initialize MainWindow.
        
        Args:
            cameraService: Camera service instance.
            detectionService: Detection service instance.
            imageSaverService: Image saver service instance.
            config: Application configuration dictionary.
        """
        super().__init__()
        
        self._cameraService = cameraService
        self._detectionService = detectionService
        self._imageSaverService = imageSaverService
        self._config = config
        
        # Frame update timer
        self._frameTimer = QTimer(self)
        self._frameTimer.timeout.connect(self._updateFrame)
        
        # FPS display timer (update every 500ms)
        self._fpsTimer = QTimer(self)
        self._fpsTimer.timeout.connect(self._updateFpsDisplay)
        
        # Performance logging settings
        perfConfig = config.get("performanceLogging", {})
        self._showFpsInStatusBar = perfConfig.get("showInStatusBar", False)
        
        # Debug save cooldown (1 second)
        self._lastDebugSave = 0.0
        self._debugSaveCooldown = 2.0  # seconds
        
        self._setupUI()
        self._setupConnections()
        self._loadInitialState()
    
    def _setupUI(self):
        """Setup the main window UI."""
        self.setWindowTitle("Label Detector")
        self.setMinimumSize(900, 600)
        
        # Central widget
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        
        # Main layout
        mainLayout = QHBoxLayout(centralWidget)
        mainLayout.setContentsMargins(10, 10, 10, 10)
        mainLayout.setSpacing(10)
        
        # Camera widget (left side, stretches)
        boxColor = tuple(self._config.get("boxColor", [0, 255, 0]))
        textColor = tuple(self._config.get("textColor", [255, 255, 255]))
        
        self._cameraWidget = CameraWidget(
            boxColor=boxColor,
            textColor=textColor,
            lineThickness=self._config.get("lineThickness", 2),
            fontSize=self._config.get("fontSize", 0.6),
            maskOpacity=self._config.get("maskOpacity", 0.4),
            maskColors=self._config.get("maskColors", None)
        )
        mainLayout.addWidget(self._cameraWidget, stretch=3)
        
        # Config panel (right side, fixed width)
        self._configPanel = ConfigPanel()
        self._configPanel.setFixedWidth(250)
        mainLayout.addWidget(self._configPanel, stretch=0)
        
        # Status bar
        self._statusBar = QStatusBar()
        self.setStatusBar(self._statusBar)
        self._statusBar.showMessage("Ready")
        
        # FPS label for status bar (persistent)
        from PySide6.QtWidgets import QLabel
        self._fpsLabel = QLabel("")
        self._fpsLabel.setStyleSheet("color: #888888; padding-right: 10px;")
        self._statusBar.addPermanentWidget(self._fpsLabel)
        
        # Apply dark theme
        self._applyTheme()
    
    def _applyTheme(self):
        """Apply dark theme to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                selection-background-color: #555555;
            }
            QDoubleSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
            }
            QStatusBar {
                background-color: #1e1e1e;
                color: #888888;
            }
            QLabel {
                color: #ffffff;
            }
        """)
    
    def _setupConnections(self):
        """Setup signal/slot connections."""
        # Config panel signals
        self._configPanel.cameraChanged.connect(self._onCameraChanged)
        self._configPanel.cameraToggled.connect(self._onCameraToggled)
        self._configPanel.detectionToggled.connect(self._onDetectionToggled)
        self._configPanel.debugToggled.connect(self._onDebugToggled)
        self._configPanel.confidenceChanged.connect(self._onConfidenceChanged)
        self._configPanel.captureRequested.connect(self._onCaptureRequested)
        self._configPanel.closeRequested.connect(self.close)
    
    def _loadInitialState(self):
        """Load initial application state."""
        # Load model
        modelPath = self._config.get("modelPath", "models/yolo11n_best.onnx")
        if not self._detectionService.loadModel(modelPath):
            QMessageBox.warning(
                self,
                "Model Error",
                f"Failed to load model: {modelPath}"
            )
        
        # Set initial confidence
        confidence = self._config.get("confidenceThreshold", 0.5)
        self._configPanel.setConfidenceThreshold(confidence)
        self._detectionService.setConfidenceThreshold(confidence)
        
        # Refresh camera list
        self._refreshCameras()
        
        # Disable capture until camera is turned on
        self._configPanel.setCaptureEnabled(False)
    
    def _refreshCameras(self):
        """Refresh the list of available cameras."""
        cameras = self._cameraService.getAvailableCameras()
        self._configPanel.updateCameraList(cameras)
        
        # Don't auto-start camera - wait for user to toggle it on
    
    def _onCameraToggled(self, enabled: bool):
        """
        Handle camera power toggle.
        
        Args:
            enabled: True if camera should be turned on.
        """
        if enabled:
            # Turn on camera
            index = self._configPanel.getSelectedCameraIndex()
            if index >= 0:
                self._startCamera(index)
            else:
                self._statusBar.showMessage("No camera selected")
                self._configPanel.setCameraEnabled(False)
        else:
            # Turn off camera
            self._stopCamera()
    
    def _startCamera(self, index: int):
        """Start camera capture."""
        frameWidth = self._config.get("frameWidth", 640)
        frameHeight = self._config.get("frameHeight", 640)
        
        if self._cameraService.openCamera(index, frameWidth, frameHeight):
            self._statusBar.showMessage(f"Camera {index} connected")
            self._configPanel.setCaptureEnabled(True)
            
            # Start frame capture timer
            fps = self._config.get("fps", 30)
            self._frameTimer.start(int(1000 / fps))
            
            # Start FPS display timer if enabled
            if self._showFpsInStatusBar:
                self._fpsTimer.start(500)  # Update every 500ms
        else:
            self._statusBar.showMessage(f"Failed to open camera {index}")
            self._configPanel.setCaptureEnabled(False)
            self._configPanel.setCameraEnabled(False)
            self._cameraWidget.showPlaceholder("Failed to open camera")
    
    def _stopCamera(self):
        """Stop camera capture."""
        self._frameTimer.stop()
        self._fpsTimer.stop()
        self._fpsLabel.setText("")
        self._cameraService.closeCamera()
        self._configPanel.setCaptureEnabled(False)
        self._cameraWidget.showPlaceholder("Camera not connected")
        self._statusBar.showMessage("Camera disconnected")
    
    def _onCameraChanged(self, index: int):
        """
        Handle camera selection change.
        
        Args:
            index: Selected camera index.
        """
        if index < 0:
            return
        
        # If camera is currently on, restart with new camera
        if self._configPanel.isCameraEnabled():
            self._stopCamera()
            self._startCamera(index)
    
    def _onDetectionToggled(self, enabled: bool):
        """
        Handle detection toggle.
        
        Args:
            enabled: True if detection enabled.
        """
        self._detectionService.setEnabled(enabled)
        status = "Detection enabled" if enabled else "Detection disabled"
        self._statusBar.showMessage(status)
    
    def _onDebugToggled(self, enabled: bool):
        """
        Handle debug mode toggle.
        
        Args:
            enabled: True if debug enabled.
        """
        self._cameraWidget.setDebugMode(enabled)
        status = "Debug mode enabled" if enabled else "Debug mode disabled"
        self._statusBar.showMessage(status)
    
    def _onConfidenceChanged(self, value: float):
        """
        Handle confidence threshold change.
        
        Args:
            value: New confidence threshold.
        """
        self._detectionService.setConfidenceThreshold(value)
        self._statusBar.showMessage(f"Confidence threshold: {value:.2f}")
    
    def _onCaptureRequested(self):
        """Handle capture button click - save raw image without annotations."""
        frame = self._cameraWidget.getCurrentFrame()
        if frame is None:
            self._statusBar.showMessage("No frame to capture")
            return
        
        # Save raw frame without any detection annotations
        filepath = self._imageSaverService.saveRawCapture(frame)
        
        if filepath:
            self._statusBar.showMessage(f"Raw image saved: {filepath}")
        else:
            self._statusBar.showMessage("Failed to save image")
    
    def _updateFrame(self):
        """Update frame from camera and run detection."""
        frame = self._cameraService.readFrame()
        if frame is None:
            return
        
        # Run detection
        detections = self._detectionService.detect(frame)
        
        # Update camera widget
        self._cameraWidget.updateFrame(frame, detections)
        
        # Auto-save in debug mode when detections found (with 1 second cooldown)
        if self._cameraWidget.isDebugMode() and detections:
            currentTime = time.time()
            if currentTime - self._lastDebugSave >= self._debugSaveCooldown:
                self._imageSaverService.saveDebugFrame(frame, detections)
                self._lastDebugSave = currentTime
                self._statusBar.showMessage(f"Debug: saved frame with {len(detections)} detection(s)")
            else:
                self._statusBar.showMessage(f"Detected: {len(detections)} label(s)")
        elif self._detectionService.isEnabled() and detections:
            self._statusBar.showMessage(f"Detected: {len(detections)} label(s)")
    
    def _updateFpsDisplay(self):
        """Update FPS display in status bar."""
        performanceLogger = self._detectionService.performanceLogger
        if performanceLogger is None:
            return
        
        fps = performanceLogger.getAverageFPS()
        totalTime = performanceLogger.getTotalTime()
        
        if fps > 0:
            self._fpsLabel.setText(f"FPS: {fps:.1f} | Total: {totalTime:.1f}ms")
        else:
            self._fpsLabel.setText("FPS: --")
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop timers
        self._frameTimer.stop()
        self._fpsTimer.stop()
        
        # Release camera
        self._cameraService.closeCamera()
        
        logger.info("Application closed")
        event.accept()
