"""
ZXing QR Code Detector Implementation.

This module provides QR code detection using the zxing-cpp library.
zxing-cpp is a high-performance C++ implementation with Python bindings.

Follows the Single Responsibility Principle (SRP) from SOLID.
"""

import re
import logging
from typing import Optional, List, Tuple

import numpy as np

from core.interfaces.qr_detector_interface import IQrDetector, QrDetectionResult


# QR code format pattern: MMDDYY-FACILITY-TYPE-ORDER-POSITION[/REVISION]
# Examples: 
#   110125-VA-M-000002-2      (no revision)
#   110125-VA-M-000002-2/1    (revised once)
#   110125-VA-M-000002-2/2    (revised twice)
QR_PATTERN = re.compile(
    r'^(\d{6})-([A-Z]{2})-([A-Z])-(\d+)-(\d+)(?:/(\d+))?$'
)


class ZxingQrDetector(IQrDetector):
    """
    QR code detector using zxing-cpp library.
    
    Detects QR codes in images and parses the content according to
    the expected label format: MMDDYY-FACILITY-TYPE-ORDER-POSITION[/REVISION]
    """
    
    def __init__(
        self,
        tryRotate: bool = True,
        tryDownscale: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ZxingQrDetector.
        
        Args:
            tryRotate: Try rotated barcodes (90/270 degrees)
            tryDownscale: Try downscaled versions for better detection
            logger: Logger instance for debug output
        """
        self._tryRotate = tryRotate
        self._tryDownscale = tryDownscale
        self._logger = logger or logging.getLogger(__name__)
        self._zxingcpp = None
        
        self._logger.info(
            f"ZxingQrDetector initialized "
            f"(tryRotate={tryRotate}, tryDownscale={tryDownscale})"
        )
    
    def _ensureZxing(self) -> None:
        """Lazily import zxing-cpp module."""
        if self._zxingcpp is None:
            try:
                import zxingcpp
                self._zxingcpp = zxingcpp
                self._logger.info("zxing-cpp module loaded successfully")
            except ImportError as e:
                self._logger.error(
                    f"Failed to import zxing-cpp. "
                    f"Please install: pip install zxing-cpp. Error: {e}"
                )
                raise
    
    def detect(self, image: np.ndarray) -> Optional[QrDetectionResult]:
        """
        Detect and decode QR code in image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            QrDetectionResult if QR code found and parsed, None otherwise
        """
        self._ensureZxing()
        
        try:
            # Convert BGR to grayscale if needed for better detection
            if len(image.shape) == 3:
                import cv2
                grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                grayImage = image
            
            # Read barcodes using zxing-cpp
            # API: read_barcodes(image, formats, try_rotate, try_downscale, ...)
            barcodes = self._zxingcpp.read_barcodes(
                grayImage,
                formats=self._zxingcpp.BarcodeFormat.QRCode,
                try_rotate=self._tryRotate,
                try_downscale=self._tryDownscale
            )
            
            if not barcodes:
                self._logger.debug("No QR code detected")
                return None
            
            # Take the first valid QR code
            for barcode in barcodes:
                if not barcode.valid:
                    continue
                
                qrText = barcode.text
                self._logger.debug(f"QR code detected: {qrText}")
                
                # Extract polygon (4 corners)
                position = barcode.position
                polygon = [
                    (position.top_left.x, position.top_left.y),
                    (position.top_right.x, position.top_right.y),
                    (position.bottom_right.x, position.bottom_right.y),
                    (position.bottom_left.x, position.bottom_left.y)
                ]
                
                # Calculate bounding box (rect: left, top, width, height)
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                rect = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                
                # Parse QR content
                parsedData = self._parseQrContent(qrText)
                
                return QrDetectionResult(
                    text=qrText,
                    polygon=polygon,
                    rect=rect,
                    confidence=1.0,  # zxing-cpp doesn't provide confidence score
                    dateCode=parsedData.get('dateCode', ''),
                    facility=parsedData.get('facility', ''),
                    orderType=parsedData.get('orderType', ''),
                    orderNumber=parsedData.get('orderNumber', ''),
                    position=parsedData.get('position', 0),
                    revisionCount=parsedData.get('revisionCount', 0)
                )
            
            self._logger.debug("No valid QR code in detected barcodes")
            return None
            
        except Exception as e:
            self._logger.error(f"Error during QR detection: {e}")
            return None
    
    def _parseQrContent(self, text: str) -> dict:
        """
        Parse QR content according to expected format.
        
        Format: MMDDYY-FACILITY-TYPE-ORDER-POSITION[/REVISION]
        Examples: 
            110125-VA-M-000002-2      (no revision)
            110125-VA-M-000002-2/1    (revised once)
        
        Args:
            text: QR code text content
            
        Returns:
            Dictionary with parsed fields
        """
        result = {
            'dateCode': None,
            'facility': None,
            'orderType': None,
            'orderNumber': None,
            'position': None,
            'revisionCount': 0
        }
        
        if not text:
            return result
        
        match = QR_PATTERN.match(text.strip())
        if match:
            result['dateCode'] = match.group(1)      # MMDDYY
            result['facility'] = match.group(2)       # VA
            result['orderType'] = match.group(3)      # M or S
            result['orderNumber'] = match.group(4)    # 000002
            result['position'] = int(match.group(5))  # 2
            result['revisionCount'] = int(match.group(6)) if match.group(6) else 0  # 1, 2, ... or 0
            
            revInfo = f", rev={result['revisionCount']}" if result['revisionCount'] > 0 else ""
            self._logger.debug(
                f"QR parsed: date={result['dateCode']}, "
                f"facility={result['facility']}, "
                f"type={result['orderType']}, "
                f"order={result['orderNumber']}, "
                f"pos={result['position']}{revInfo}"
            )
        else:
            self._logger.warning(f"QR content does not match expected format: {text}")
        
        return result
