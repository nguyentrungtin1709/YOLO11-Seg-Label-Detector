"""
S5 QR Detection Service Implementation.

Step 5 of the pipeline: QR code detection and decoding.
Creates and manages QR detector from core layer using factory pattern.

Follows:
- SRP: Only handles QR detection operations
- DIP: Depends on IQrDetector abstraction (interface)
- OCP: Extends without modifying existing code
- Factory Pattern: Uses createQrDetector() for backend selection
"""

import os
import time
import logging
from typing import Optional

import cv2
import numpy as np

from core.interfaces.qr_detector_interface import IQrDetector, QrDetectionResult
from core.qr import createQrDetector, QrImagePreprocessor
from services.interfaces.qr_detection_service_interface import (
    IQrDetectionService,
    QrDetectionServiceResult
)
from services.interfaces.base_service_interface import BaseService


class S5QrDetectionService(IQrDetectionService, BaseService):
    """
    Step 5: QR Detection Service Implementation.
    
    Detects and decodes QR codes from label images.
    The QR code contains order information used for validation.
    
    Supports multiple backends (ZXing, WeChat) via factory pattern.
    Optionally applies preprocessing to improve detection rate.
    
    Preprocessing modes:
    - "minimal": Scale only (fast)
    - "full": Scale → Denoise → Binary → Morph → Invert (thorough)
    """
    
    SERVICE_NAME = "s5_qr_detection"
    
    def __init__(
        self,
        # Basic settings
        enabled: bool = True,
        
        # Backend selection
        backend: str = "zxing",
        
        # ZXing params (prefixed with 'zxing')
        zxingTryRotate: bool = True,
        zxingTryDownscale: bool = True,
        
        # WeChat params (prefixed with 'wechat')
        wechatModelDir: str = "models/wechat",
        
        # Preprocessing params (prefixed with 'preprocessing')
        preprocessingEnabled: bool = False,
        preprocessingMode: str = "full",
        preprocessingScaleFactor: float = 1.5,
        
        # Debug settings
        debugBasePath: str = "output/debug",
        debugEnabled: bool = False
    ):
        """
        Initialize S5QrDetectionService.
        
        Args:
            enabled: Whether QR detection is enabled.
            backend: QR detection backend ("zxing" or "wechat").
            zxingTryRotate: (ZXing) Try rotated barcodes (90/270 degrees).
            zxingTryDownscale: (ZXing) Try downscaled versions for better detection.
            wechatModelDir: (WeChat) Directory containing model files.
            preprocessingEnabled: Enable image preprocessing before detection.
            preprocessingMode: Preprocessing mode ("minimal" or "full").
            preprocessingScaleFactor: Scale factor for preprocessing.
            debugBasePath: Base path for debug output.
            debugEnabled: Whether to save debug output.
        """
        BaseService.__init__(
            self,
            serviceName=self.SERVICE_NAME,
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        
        # Create QR detector using factory
        self._qrDetector: IQrDetector = createQrDetector(
            backend=backend,
            zxingTryRotate=zxingTryRotate,
            zxingTryDownscale=zxingTryDownscale,
            wechatModelDir=wechatModelDir
        )
        
        # Create preprocessor (optional)
        self._preprocessor: Optional[QrImagePreprocessor] = None
        self._preprocessingMode = preprocessingMode
        if preprocessingEnabled:
            self._preprocessor = QrImagePreprocessor(
                enabled=preprocessingEnabled,
                mode=preprocessingMode,
                scaleFactor=preprocessingScaleFactor
            )
        
        self._enabled = enabled
        self._backend = backend
        
        # Ensure debug input directory exists
        self._debugInputPath = os.path.join(debugBasePath, self.SERVICE_NAME, "inputs")
        if debugEnabled:
            os.makedirs(self._debugInputPath, exist_ok=True)
        
        self._logger.info(
            f"S5QrDetectionService initialized "
            f"(backend={backend}, preprocessing={preprocessingEnabled}, "
            f"mode={preprocessingMode if preprocessingEnabled else 'none'})"
        )
    
    def detectQr(
        self,
        image: np.ndarray,
        frameId: str
    ) -> QrDetectionServiceResult:
        """
        Detect and decode QR code from an image.
        
        Timing includes both preprocessing and detection.
        Debug image saving is NOT included in timing.
        
        Args:
            image: Input image (grayscale from S4).
            frameId: Frame identifier.
            
        Returns:
            QrDetectionServiceResult with detection result.
        """
        startTime = time.time()
        
        # Check if QR detection is enabled
        if not self._enabled:
            return QrDetectionServiceResult(
                qrData=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        if image is None:
            self._logger.warning(f"[{frameId}] No image provided for QR detection")
            return QrDetectionServiceResult(
                qrData=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        try:
            # Step 1: Preprocessing (included in timing)
            if self._preprocessor is not None:
                processedImage = self._preprocessor.preprocess(image)
            else:
                processedImage = image
            
            # Step 2: QR Detection (included in timing)
            qrResult = self._qrDetector.detect(processedImage)
            
            # Step 3: Scale back coordinates if preprocessing was applied
            # QR coordinates are detected on scaled image, need to convert back to original size
            if qrResult is not None and self._preprocessor is not None:
                scaleFactor = self._preprocessor.scaleFactor
                if scaleFactor != 1.0:
                    qrResult = self._scaleBackCoordinates(qrResult, scaleFactor)
                    self._logger.debug(
                        f"[{frameId}] Scaled back QR coordinates by 1/{scaleFactor}"
                    )
            
            # Measure time BEFORE debug saving
            processingTimeMs = self._measureTime(startTime)
            
            # Step 4: Debug input saving (NOT included in timing)
            self._saveDebugInput(frameId, processedImage)
            
            if qrResult is None:
                self._logger.warning(
                    f"[{frameId}] No QR code detected "
                    f"(preprocessing={self._preprocessor is not None}, "
                    f"mode={self._preprocessingMode if self._preprocessor else 'none'}, "
                    f"time={processingTimeMs:.2f}ms)"
                )
                return QrDetectionServiceResult(
                    qrData=None,
                    frameId=frameId,
                    success=False,
                    processingTimeMs=processingTimeMs
                )
            
            # Save debug output (NOT included in timing)
            self._saveDebugOutput(frameId, qrResult)
            
            # Log timing and result
            self._logTiming(frameId, processingTimeMs)
            self._logger.info(
                f"[{frameId}] QR detected: {qrResult.text} "
                f"(preprocessing={self._preprocessor is not None}, "
                f"mode={self._preprocessingMode if self._preprocessor else 'none'}, "
                f"time={processingTimeMs:.2f}ms)"
            )
            
            return QrDetectionServiceResult(
                qrData=qrResult,
                frameId=frameId,
                success=True,
                processingTimeMs=processingTimeMs
            )
            
        except Exception as e:
            self._logger.error(f"[{frameId}] QR detection failed: {e}")
            return QrDetectionServiceResult(
                qrData=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
    
    def setEnabled(self, enabled: bool) -> None:
        """Enable or disable QR detection."""
        self._enabled = enabled
        self._logger.info(f"QR detection {'enabled' if enabled else 'disabled'}")
    
    def isEnabled(self) -> bool:
        """Check if QR detection is enabled."""
        return self._enabled
    
    def getBackend(self) -> str:
        """Get current QR detection backend."""
        return self._backend
    
    def isPreprocessingEnabled(self) -> bool:
        """Check if preprocessing is enabled."""
        return self._preprocessor is not None and self._preprocessor.isEnabled()
    
    def getPreprocessingMode(self) -> str:
        """Get current preprocessing mode."""
        if self._preprocessor is not None:
            return self._preprocessor.mode
        return "none"
    
    def _saveDebugInput(
        self,
        frameId: str,
        inputImage: np.ndarray
    ) -> None:
        """
        Save input image before QR detection for debugging.
        
        Saves to: output/debug/s5_qr_detection/inputs/
        
        Args:
            frameId: Frame identifier.
            inputImage: Preprocessed image to save.
        """
        if not self._debugEnabled:
            return
        
        if inputImage is None:
            return
        
        try:
            # Create inputs directory if not exists
            os.makedirs(self._debugInputPath, exist_ok=True)
            
            # Generate filename with mode info
            modeStr = self._preprocessingMode if self._preprocessor else "raw"
            filename = f"{frameId}_{modeStr}.png"
            filepath = os.path.join(self._debugInputPath, filename)
            
            # Save image
            cv2.imwrite(filepath, inputImage)
            self._logger.debug(f"[{frameId}] Saved debug input: {filepath}")
            
        except Exception as e:
            self._logger.error(f"[{frameId}] Failed to save debug input: {e}")
    
    def _scaleBackCoordinates(
        self,
        qrResult: QrDetectionResult,
        scaleFactor: float
    ) -> QrDetectionResult:
        """
        Scale QR coordinates back to original image size.
        
        When preprocessing scales the image, QR detection returns coordinates
        relative to the scaled image. This method converts them back to the
        original image coordinate system for use by S6 Component Extraction.
        
        Args:
            qrResult: QR detection result with coordinates on scaled image.
            scaleFactor: The scale factor that was applied during preprocessing.
            
        Returns:
            New QrDetectionResult with coordinates scaled back to original size.
        """
        # Scale polygon coordinates back
        scaledPolygon = [
            (int(x / scaleFactor), int(y / scaleFactor))
            for (x, y) in qrResult.polygon
        ]
        
        # Scale rect (x, y, w, h) back
        scaledRect = (
            int(qrResult.rect[0] / scaleFactor),
            int(qrResult.rect[1] / scaleFactor),
            int(qrResult.rect[2] / scaleFactor),
            int(qrResult.rect[3] / scaleFactor)
        )
        
        # Return new QrDetectionResult with adjusted coordinates
        return QrDetectionResult(
            text=qrResult.text,
            polygon=scaledPolygon,
            rect=scaledRect,
            confidence=qrResult.confidence,
            dateCode=qrResult.dateCode,
            facility=qrResult.facility,
            orderType=qrResult.orderType,
            orderNumber=qrResult.orderNumber,
            position=qrResult.position,
            revisionCount=qrResult.revisionCount
        )
    
    def _saveDebugOutput(
        self,
        frameId: str,
        qrResult: QrDetectionResult
    ) -> None:
        """Save debug output for QR detection step."""
        if not self._debugEnabled:
            return
        
        # Save QR data as JSON
        data = {
            "frameId": frameId,
            "text": qrResult.text,
            "polygon": qrResult.polygon,
            "rect": qrResult.rect,
            "confidence": qrResult.confidence,
            "backend": self._backend,
            "preprocessingEnabled": self._preprocessor is not None,
            "preprocessingMode": self._preprocessingMode if self._preprocessor else None,
            "parsed": {
                "dateCode": qrResult.dateCode,
                "facility": qrResult.facility,
                "orderType": qrResult.orderType,
                "orderNumber": qrResult.orderNumber,
                "position": qrResult.position,
                "revisionCount": qrResult.revisionCount
            }
        }
        self._saveDebugJson(frameId, data, "qr")
