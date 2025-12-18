"""
Pyzbar QR Detector Implementation.

This module provides QR code detection using the pyzbar library.
Follows the Single Responsibility Principle (SRP) and
Dependency Inversion Principle (DIP) from SOLID.
"""

import re
import logging
from typing import Optional, List

import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol, Decoded

from core.interfaces.qr_detector_interface import IQrDetector, QrDetectionResult


class PyzbarQrDetector(IQrDetector):
    """
    QR code detector using pyzbar library.
    
    Detects QR codes and parses content according to the format:
    MMDDYY-FACILITY-TYPE-ORDER-POSITION
    Example: 110125-VA-M-000002-2
    """
    
    # Pattern: MMDDYY-FACILITY-TYPE-ORDER-POSITION
    QR_PATTERN = re.compile(r'^(\d{6})-([A-Z]{2})-([A-Z])-(\d+)-(\d+)$')
    
    def __init__(
        self, 
        symbolTypes: Optional[List[ZBarSymbol]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize PyzbarQrDetector.
        
        Args:
            symbolTypes: List of barcode types to detect (default: QRCODE only)
            logger: Logger instance for debug output
        """
        self._symbolTypes = symbolTypes or [ZBarSymbol.QRCODE]
        self._logger = logger or logging.getLogger(__name__)
    
    def detect(self, image: np.ndarray) -> Optional[QrDetectionResult]:
        """
        Detect QR code in an image.
        
        Args:
            image: Input image (BGR or grayscale numpy array)
            
        Returns:
            QrDetectionResult if QR code found, None otherwise
        """
        try:
            # Decode QR codes from image
            results: List[Decoded] = decode(image, symbols=self._symbolTypes)
            
            if not results:
                self._logger.debug("No QR code detected in image")
                return None
            
            # Take the first QR code found
            qr = results[0]
            text = qr.data.decode('utf-8')
            
            self._logger.debug(f"QR code detected: {text}")
            
            # Create result with basic info
            result = QrDetectionResult(
                text=text,
                polygon=[(p.x, p.y) for p in qr.polygon],
                rect=(qr.rect.left, qr.rect.top, qr.rect.width, qr.rect.height),
                confidence=qr.quality / 100.0 if qr.quality else 1.0
            )
            
            # Parse QR code content if it matches expected pattern
            self._parseQrContent(result, text)
            
            return result
            
        except Exception as e:
            self._logger.error(f"Error detecting QR code: {e}")
            return None
    
    def _parseQrContent(self, result: QrDetectionResult, text: str) -> None:
        """
        Parse QR code content and populate result fields.
        
        Args:
            result: QrDetectionResult to populate
            text: QR code text content
        """
        match = self.QR_PATTERN.match(text)
        if match:
            result.dateCode = match.group(1)      # 110125
            result.facility = match.group(2)     # VA
            result.orderType = match.group(3)    # M
            result.orderNumber = match.group(4)  # 000002
            result.position = int(match.group(5)) # 2
            
            self._logger.debug(
                f"Parsed QR: date={result.dateCode}, facility={result.facility}, "
                f"type={result.orderType}, order={result.orderNumber}, "
                f"position={result.position}"
            )
        else:
            self._logger.warning(f"QR content does not match expected pattern: {text}")
