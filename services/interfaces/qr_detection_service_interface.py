"""
QR Detection Service Interface Module.

Defines the interface for QR code detection operations (Step 5 of the pipeline).
Responsible for detecting and decoding QR codes from label images.

Follows:
- SRP: Only handles QR detection operations
- DIP: Depends on IQrDetector abstraction from core layer
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

from core.interfaces.qr_detector_interface import QrDetectionResult


@dataclass
class QrDetectionServiceResult:
    """
    Result of the QR detection service.
    
    Attributes:
        qrData: QR detection result with decoded data.
        frameId: Frame identifier for debug output.
        success: Whether QR detection was successful.
        processingTimeMs: Time taken for QR detection.
    """
    qrData: Optional[QrDetectionResult]
    frameId: str
    success: bool
    processingTimeMs: float = 0.0


class IQrDetectionService(ABC):
    """
    Interface for QR detection operations (Step 5).
    
    Detects and decodes QR codes from label images.
    The QR code contains order information used for validation.
    """
    
    @abstractmethod
    def detectQr(
        self,
        image: np.ndarray,
        frameId: str
    ) -> QrDetectionServiceResult:
        """
        Detect and decode QR code from an image.
        
        Args:
            image: Input image (BGR format).
            frameId: Frame identifier for debug output.
            
        Returns:
            QrDetectionServiceResult: QR detection result with metadata.
        """
        pass
    
    @abstractmethod
    def setEnabled(self, enabled: bool) -> None:
        """
        Enable or disable QR detection.
        
        Args:
            enabled: True to enable QR detection.
        """
        pass
    
    @abstractmethod
    def isEnabled(self) -> bool:
        """
        Check if QR detection is enabled.
        
        Returns:
            bool: True if QR detection is enabled.
        """
        pass
