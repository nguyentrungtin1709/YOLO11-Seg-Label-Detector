"""
WeChat QR Code Detector Implementation.

This module provides QR code detection using OpenCV's WeChat QRCode module.
WeChat QRCode is a high-performance QR code detector with super-resolution support.

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


class WechatQrDetector(IQrDetector):
    """
    QR code detector using OpenCV WeChat QRCode module.
    
    Detects QR codes in images and parses the content according to
    the expected label format: MMDDYY-FACILITY-TYPE-ORDER-POSITION[/REVISION]
    
    WeChat QRCode provides:
    - Deep learning based detection
    - Super-resolution for small QR codes
    - High accuracy on various conditions
    """
    
    def __init__(
        self,
        modelDir: str = "models/wechat",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize WechatQrDetector.
        
        Args:
            modelDir: Directory containing WeChat QR model files:
                - detect.prototxt
                - detect.caffemodel
                - sr.prototxt
                - sr.caffemodel
            logger: Logger instance for debug output
        """
        self._modelDir = modelDir
        self._logger = logger or logging.getLogger(__name__)
        self._detector = None
        
        self._logger.info(
            f"WechatQrDetector initialized (modelDir={modelDir})"
        )
    
    def _ensureDetector(self) -> None:
        """Lazily initialize WeChat QR detector with model files."""
        if self._detector is None:
            try:
                import cv2
                import os
                
                detectProtoPath = os.path.join(self._modelDir, "detect.prototxt")
                detectModelPath = os.path.join(self._modelDir, "detect.caffemodel")
                srProtoPath = os.path.join(self._modelDir, "sr.prototxt")
                srModelPath = os.path.join(self._modelDir, "sr.caffemodel")
                
                # Validate model files exist
                for path in [detectProtoPath, detectModelPath, srProtoPath, srModelPath]:
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"WeChat QR model file not found: {path}")
                
                # Initialize WeChat QRCode detector
                # API: cv2.wechat_qrcode.WeChatQRCode(
                #     detector_prototxt_path,
                #     detector_caffe_model_path,
                #     super_resolution_prototxt_path,
                #     super_resolution_caffe_model_path
                # )
                self._detector = cv2.wechat_qrcode.WeChatQRCode(
                    detectProtoPath,
                    detectModelPath,
                    srProtoPath,
                    srModelPath
                )
                
                self._logger.info("WeChat QRCode detector loaded successfully")
                
            except AttributeError as e:
                self._logger.error(
                    f"WeChat QRCode module not available. "
                    f"Please install opencv-contrib-python: "
                    f"pip install opencv-contrib-python. Error: {e}"
                )
                raise ImportError(
                    "WeChat QRCode requires opencv-contrib-python. "
                    "Install with: pip install opencv-contrib-python"
                ) from e
            except Exception as e:
                self._logger.error(f"Failed to initialize WeChat QR detector: {e}")
                raise
    
    def detect(self, image: np.ndarray) -> Optional[QrDetectionResult]:
        """
        Detect and decode QR code in image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            QrDetectionResult if QR code found and parsed, None otherwise
        """
        self._ensureDetector()
        
        try:
            # WeChat QRCode API: detectAndDecode(image) -> (texts, points)
            # - texts: tuple of decoded strings (can detect multiple QR codes)
            # - points: numpy array of corner points for each QR code
            decodedTexts, points = self._detector.detectAndDecode(image)
            
            if not decodedTexts or not decodedTexts[0]:
                self._logger.debug("No QR code detected")
                return None
            
            # Take the first valid QR code
            for idx, qrText in enumerate(decodedTexts):
                if not qrText:
                    continue
                
                self._logger.debug(f"QR code detected: {qrText}")
                
                # Extract polygon (4 corners) for this QR code
                # points shape: (n_qrcodes, 4, 2) or list of arrays
                polygon = []
                if points is not None and len(points) > idx:
                    qrPoints = points[idx]
                    if qrPoints is not None and len(qrPoints) >= 4:
                        polygon = [
                            (int(qrPoints[0][0]), int(qrPoints[0][1])),
                            (int(qrPoints[1][0]), int(qrPoints[1][1])),
                            (int(qrPoints[2][0]), int(qrPoints[2][1])),
                            (int(qrPoints[3][0]), int(qrPoints[3][1]))
                        ]
                
                # Calculate bounding box (rect: left, top, width, height)
                rect = (0, 0, 0, 0)
                if polygon:
                    xs = [p[0] for p in polygon]
                    ys = [p[1] for p in polygon]
                    rect = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
                
                # Parse QR content
                parsedData = self._parseQrContent(qrText)
                
                return QrDetectionResult(
                    text=qrText,
                    polygon=polygon,
                    rect=rect,
                    confidence=1.0,  # WeChat QRCode doesn't provide confidence score
                    dateCode=parsedData.get('dateCode', ''),
                    facility=parsedData.get('facility', ''),
                    orderType=parsedData.get('orderType', ''),
                    orderNumber=parsedData.get('orderNumber', ''),
                    position=parsedData.get('position', 0),
                    revisionCount=parsedData.get('revisionCount', 0)
                )
            
            self._logger.debug("No valid QR code in detected results")
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
            result['revisionCount'] = int(match.group(6)) if match.group(6) else 0
            
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
