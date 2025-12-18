"""
QR Detector Interface Module.

This module defines the interface and data classes for QR code detection.
Follows the Interface Segregation Principle (ISP) from SOLID.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class QrDetectionResult:
    """
    Result of QR code detection.
    
    Attributes:
        text: Full QR code content (e.g., "110125-VA-M-000002-2")
        polygon: Four corners of QR code [(x,y), ...]
        rect: Bounding rectangle (left, top, width, height)
        confidence: Detection confidence score (0-1)
        dateCode: Parsed date code (MMDDYY)
        facility: Parsed facility code (VA, GA, ...)
        orderType: Parsed order type (M, S, ...)
        orderNumber: Parsed order number
        position: Parsed position number from QR
    """
    text: str
    polygon: List[Tuple[int, int]]
    rect: Tuple[int, int, int, int]
    confidence: float
    
    # Parsed fields from QR code (format: MMDDYY-FACILITY-TYPE-ORDER-POSITION)
    dateCode: str = ""
    facility: str = ""
    orderType: str = ""
    orderNumber: str = ""
    position: int = 0


class IQrDetector(ABC):
    """
    Interface for QR code detector.
    
    Implementations should detect QR codes in images and return
    structured results including parsed QR content.
    """
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> Optional[QrDetectionResult]:
        """
        Detect QR code in an image.
        
        Args:
            image: Input image (BGR or grayscale numpy array)
            
        Returns:
            QrDetectionResult if QR code found, None otherwise
        """
        pass
