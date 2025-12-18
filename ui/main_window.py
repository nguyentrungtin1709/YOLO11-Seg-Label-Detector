"""
Main Window

Main application window containing all UI components.
Orchestrates the interaction between widgets and services via PipelineOrchestrator.
"""

import logging
import time
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStatusBar, QMessageBox, QLabel
)

from ui.widgets.camera_widget import CameraWidget
from ui.widgets.config_panel import ConfigPanel
from ui.widgets.ocr_result_widget import OcrResultWidget

if TYPE_CHECKING:
    from ui.pipeline_orchestrator import PipelineOrchestrator


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window.
    
    Contains camera view and configuration panel.
    Manages the main application loop and service interactions.
    Receives PipelineOrchestrator to access all services.
    """
    
    def __init__(self, orchestrator: "PipelineOrchestrator"):
        """
        Initialize MainWindow.
        
        Args:
            orchestrator: Pipeline orchestrator containing all services.
        """
        super().__init__()
        
        # Store orchestrator and extract services
        self._orchestrator = orchestrator
        self._configService = orchestrator.configService
        
        # Get services from orchestrator
        self._cameraService = orchestrator.cameraService
        self._detectionService = orchestrator.detectionService
        self._preprocessingService = orchestrator.preprocessingService
        self._enhancementService = orchestrator.enhancementService
        self._qrDetectionService = orchestrator.qrDetectionService
        self._componentExtractionService = orchestrator.componentExtractionService
        self._ocrService = orchestrator.ocrService
        self._postprocessingService = orchestrator.postprocessingService
        
        # Frame update timer
        self._frameTimer = QTimer(self)
        self._frameTimer.timeout.connect(self._updateFrame)
        
        # FPS display timer (update every 500ms)
        self._fpsTimer = QTimer(self)
        self._fpsTimer.timeout.connect(self._updateFpsDisplay)
        
        # Performance logging settings
        self._showFpsInStatusBar = self._configService.isShowFpsInStatusBar()
        
        # Debug save cooldown
        self._lastDebugSave = 0.0
        self._debugSaveCooldown = self._configService.getDebugSaveCooldown()
        
        # Frame counter for unique IDs
        self._frameCounter = 0
        
        self._setupUI()
        self._setupConnections()
        self._loadInitialState()
    
    def _setupUI(self):
        """Setup the main window UI."""
        self.setWindowTitle("Label Detector")
        self.setMinimumSize(
            self._configService.getWindowMinWidth(),
            self._configService.getWindowMinHeight()
        )
        
        # Central widget
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        
        # Main layout
        mainLayout = QHBoxLayout(centralWidget)
        mainLayout.setContentsMargins(10, 10, 10, 10)
        mainLayout.setSpacing(10)
        
        # Get visualization config
        vizConfig = self._configService.getVisualizationConfig()
        boxColor = tuple(vizConfig.get("boxColor", [0, 255, 0]))
        textColor = tuple(vizConfig.get("textColor", [0, 0, 0]))
        
        self._cameraWidget = CameraWidget(
            boxColor=boxColor,
            textColor=textColor,
            lineThickness=vizConfig.get("lineThickness", 2),
            fontSize=vizConfig.get("fontSize", 0.8),
            maskOpacity=vizConfig.get("maskOpacity", 0.4),
            maskColors=vizConfig.get("maskColors", None)
        )
        mainLayout.addWidget(self._cameraWidget, stretch=3)
        
        # Config panel (right side, fixed width)
        self._configPanel = ConfigPanel()
        self._configPanel.setFixedWidth(250)
        mainLayout.addWidget(self._configPanel, stretch=0)
        
        # OCR Result widget (right side of config panel)
        self._ocrResultWidget = OcrResultWidget()
        self._ocrResultWidget.setFixedWidth(220)
        mainLayout.addWidget(self._ocrResultWidget, stretch=0)
        
        # Status bar
        self._statusBar = QStatusBar()
        self.setStatusBar(self._statusBar)
        self._statusBar.showMessage("Ready")
        
        # FPS label for status bar (persistent)
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
        # Load model via detection service
        modelPath = self._configService.getModelPath()
        if not self._detectionService.loadModel(modelPath):
            QMessageBox.warning(
                self,
                "Model Error",
                f"Failed to load model: {modelPath}"
            )
        
        # Set initial confidence
        confidence = self._configService.getConfidenceThreshold()
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
        frameWidth = self._configService.getFrameWidth()
        frameHeight = self._configService.getFrameHeight()
        
        if self._cameraService.openCamera(index, frameWidth, frameHeight):
            self._statusBar.showMessage(f"Camera {index} connected")
            self._configPanel.setCaptureEnabled(True)
            
            # Start frame capture timer
            fps = self._configService.getFps()
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
        Controls debug output with cooldown mechanism.
        
        Debug is controlled per-frame based on cooldown in _updateFrame.
        This just sets the overall debug mode flag.
        
        Args:
            enabled: True if debug enabled.
        """
        self._cameraWidget.setDebugMode(enabled)
        
        # Reset cooldown timer when debug is toggled on
        if enabled:
            self._lastDebugSave = 0.0  # Force immediate save on next frame
        else:
            # Disable debug for all services immediately
            self._orchestrator.setDebugEnabled(False)
        
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
        
        # Save raw frame to captures folder
        import os
        from datetime import datetime
        
        capturesDir = "output/captures"
        os.makedirs(capturesDir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"capture_{timestamp}.png"
        filepath = os.path.join(capturesDir, filename)
        
        try:
            import cv2
            cv2.imwrite(filepath, frame)
            self._statusBar.showMessage(f"Captured: {filename}")
            logger.info(f"Frame captured: {filepath}")
        except Exception as e:
            self._statusBar.showMessage(f"Capture failed: {str(e)}")
            logger.error(f"Failed to capture frame: {e}")
    
    def _updateFrame(self):
        """
        Update frame from camera and run the full 8-step pipeline.
        
        Pipeline Steps:
        S1: Camera capture (frame already captured)
        S2: Detection
        S3: Preprocessing (crop, rotate, orientation)
        S4: Enhancement (brightness, sharpness)
        S5: QR Detection
        S6: Component Extraction
        S7: OCR
        S8: Postprocessing
        """
        import time
        pipelineStartTime = time.time()
        pipelineTiming = {}
        
        # Check debug cooldown - only save debug if cooldown elapsed
        currentTime = time.time()
        shouldSaveDebug = (
            self._cameraWidget.isDebugMode() and 
            (currentTime - self._lastDebugSave) >= self._debugSaveCooldown
        )
        
        # Temporarily enable/disable debug for all services based on cooldown
        if self._cameraWidget.isDebugMode():
            self._orchestrator.setDebugEnabled(shouldSaveDebug)
        
        # S1: Get frame from camera service
        frameResult = self._cameraService.captureFrame()
        pipelineTiming["s1_camera"] = frameResult.processingTimeMs
        
        if not frameResult.success or frameResult.image is None:
            return
        
        frame = frameResult.image
        frameId = frameResult.frameId
        
        # S2: Run detection
        detectionResult = self._detectionService.detect(frame, frameId)
        pipelineTiming["s2_detection"] = detectionResult.processingTimeMs
        
        # Update camera widget with frame and detections
        detections = detectionResult.detections if detectionResult.success else []
        self._cameraWidget.updateFrame(frame, detections)
        
        # If no detections, clear OCR results and return
        if not detections:
            self._configPanel.clearPreprocessedImage()
            self._ocrResultWidget.clear()
            return
        
        # Get first detection for processing
        firstDetection = detections[0]
        
        # S3: Preprocessing (crop, rotate, fix orientation)
        if self._preprocessingService.isEnabled():
            preprocessResult = self._preprocessingService.preprocess(
                frame, firstDetection, frameId
            )
            pipelineTiming["s3_preprocessing"] = preprocessResult.processingTimeMs
            
            if not preprocessResult.success or preprocessResult.croppedImage is None:
                self._configPanel.clearPreprocessedImage()
                self._ocrResultWidget.clear()
                return
            processedImage = preprocessResult.croppedImage
        else:
            # Without preprocessing, skip further processing
            self._configPanel.clearPreprocessedImage()
            self._ocrResultWidget.clear()
            return
        
        # S4: Enhancement (brightness, sharpness)
        if self._enhancementService.isEnabled():
            enhanceResult = self._enhancementService.enhance(processedImage, frameId)
            pipelineTiming["s4_enhancement"] = enhanceResult.processingTimeMs
            
            if enhanceResult.success and enhanceResult.enhancedImage is not None:
                processedImage = enhanceResult.enhancedImage
        
        # Update preprocessed image display
        self._configPanel.updatePreprocessedImage(processedImage)
        
        # S5: QR Detection
        qrResult = self._qrDetectionService.detectQr(processedImage, frameId)
        pipelineTiming["s5_qr_detection"] = qrResult.processingTimeMs
        
        if not qrResult.success or qrResult.qrData is None:
            self._ocrResultWidget.showError("No QR detected")
            self._logPipelineTiming(frameId, pipelineTiming, pipelineStartTime, shouldSaveDebug)
            return
        
        # S6: Component Extraction
        componentResult = self._componentExtractionService.extractComponents(
            processedImage, 
            qrResult.qrData.polygon,
            frameId
        )
        pipelineTiming["s6_component_extraction"] = componentResult.processingTimeMs
        
        if not componentResult.success or componentResult.mergedImage is None:
            self._ocrResultWidget.showError("Component extraction failed")
            self._logPipelineTiming(frameId, pipelineTiming, pipelineStartTime, shouldSaveDebug)
            return
        
        # S7: OCR
        ocrResult = self._ocrService.extractText(componentResult.mergedImage, frameId)
        pipelineTiming["s7_ocr"] = ocrResult.processingTimeMs
        
        textBlocks = ocrResult.ocrData.textBlocks if ocrResult.success and ocrResult.ocrData else []
        
        # S8: Postprocessing
        postResult = self._postprocessingService.process(
            textBlocks,
            qrResult.qrData,
            frameId
        )
        pipelineTiming["s8_postprocessing"] = postResult.processingTimeMs
        
        if postResult.success and postResult.labelData:
            self._ocrResultWidget.updateResult(
                postResult.labelData,
                postResult.processingTimeMs
            )
        else:
            self._ocrResultWidget.showError("Processing failed")
        
        # Log pipeline timing and update debug save time
        self._logPipelineTiming(frameId, pipelineTiming, pipelineStartTime, shouldSaveDebug)
        
        if shouldSaveDebug:
            self._lastDebugSave = currentTime
        
        # Update status
        self._statusBar.showMessage(f"Detected: {len(detections)} label(s)")
    
    def _logPipelineTiming(
        self, 
        frameId: str, 
        timing: dict, 
        startTime: float,
        saveToFile: bool = False
    ) -> None:
        """
        Log and optionally save pipeline timing information.
        
        When debug mode is enabled, displays detailed timing for each step.
        Otherwise, only shows FPS and total time.
        
        Args:
            frameId: Frame identifier.
            timing: Dictionary of service timings.
            startTime: Pipeline start time.
            saveToFile: Whether to save timing to debug file.
        """
        import time
        
        totalTime = (time.time() - startTime) * 1000
        timing["total_pipeline"] = round(totalTime, 2)
        
        # Round all timing values for cleaner output
        for key in timing:
            if isinstance(timing[key], float):
                timing[key] = round(timing[key], 2)
        
        # Log to console
        timingStr = " | ".join([f"{k}: {v:.1f}ms" for k, v in timing.items()])
        logger.info(f"[{frameId}] Pipeline timing: {timingStr}")
        
        # Update FPS label - show detailed timing when debug mode is enabled
        if self._showFpsInStatusBar and totalTime > 0:
            fps = 1000 / totalTime
            
            if self._cameraWidget.isDebugMode():
                # Detailed timing display when debug is on
                detailParts = [f"FPS: {fps:.1f}"]
                for key, value in timing.items():
                    if key != "total_pipeline":
                        # Shorten key name: s1_camera -> S1
                        shortKey = key.split("_")[0].upper()
                        detailParts.append(f"{shortKey}: {value:.0f}")
                detailParts.append(f"Total: {totalTime:.0f}ms")
                self._fpsLabel.setText(" | ".join(detailParts))
            else:
                # Simple display when debug is off
                self._fpsLabel.setText(f"FPS: {fps:.1f} | Total: {totalTime:.1f}ms")
        
        # Save to file if debug mode and cooldown elapsed
        if saveToFile:
            self._orchestrator.savePipelineTiming(frameId, timing)
    
    def _updateFpsDisplay(self):
        """Update FPS display in status bar."""
        # FPS is now updated in _logPipelineTiming
        pass
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop timers
        self._frameTimer.stop()
        self._fpsTimer.stop()
        
        # Release camera via service
        self._cameraService.closeCamera()
        
        logger.info("Application closed")
        event.accept()
