"""
S5 QR Detection Service Implementation.

Step 5 of the pipeline: QR code detection and decoding.
Creates and manages ZxingQrDetector from core layer.

Follows:
- SRP: Only handles QR detection operations
- DIP: Depends on IQrDetector abstraction (interface)
- OCP: Extends without modifying existing code
"""

import time
import logging
from typing import Optional

import numpy as np

from core.interfaces.qr_detector_interface import IQrDetector, QrDetectionResult
from core.qr.zxing_qr_detector import ZxingQrDetector
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
    
    Creates ZxingQrDetector internally with provided parameters.
    """
    
    SERVICE_NAME = "s5_qr_detection"
    
    def __init__(
        self,
        enabled: bool = True,
        tryRotate: bool = True,
        tryDownscale: bool = True,
        debugBasePath: str = "output/debug",
        debugEnabled: bool = False
    ):
        """
        Initialize S5QrDetectionService.
        
        Args:
            enabled: Whether QR detection is enabled.
            tryRotate: Try rotated barcodes (90/270 degrees).
            tryDownscale: Try downscaled versions for better detection.
            debugBasePath: Base path for debug output.
            debugEnabled: Whether to save debug output.
        """
        BaseService.__init__(
            self,
            serviceName=self.SERVICE_NAME,
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        
        # Create core QR detector implementation
        self._qrDetector: IQrDetector = ZxingQrDetector(
            tryRotate=tryRotate,
            tryDownscale=tryDownscale
        )
        
        self._enabled = enabled
        
        self._logger.info(
            f"S5QrDetectionService initialized "
            f"(tryRotate={tryRotate}, tryDownscale={tryDownscale})"
        )
    
    def detectQr(
        self,
        image: np.ndarray,
        frameId: str
    ) -> QrDetectionServiceResult:
        """Detect and decode QR code from an image."""
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
            # Run QR detection
            qrResult = self._qrDetector.detect(image)
            
            processingTimeMs = self._measureTime(startTime)
            
            if qrResult is None:
                self._logger.warning(f"[{frameId}] No QR code detected")
                return QrDetectionServiceResult(
                    qrData=None,
                    frameId=frameId,
                    success=False,
                    processingTimeMs=processingTimeMs
                )
            
            # Save debug output
            self._saveDebugOutput(frameId, qrResult)
            
            # Log timing
            self._logTiming(frameId, processingTimeMs)
            self._logger.info(f"[{frameId}] QR detected: {qrResult.text}")
            
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
            "parsed": {
                "dateCode": qrResult.dateCode,
                "facility": qrResult.facility,
                "orderType": qrResult.orderType,
                "orderNumber": qrResult.orderNumber,
                "position": qrResult.position
            }
        }
        self._saveDebugJson(frameId, data, "qr")
