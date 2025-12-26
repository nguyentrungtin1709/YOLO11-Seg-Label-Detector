"""
QR Detector Factory Module.

Factory function for creating QR detector instances based on backend selection.
Supports ZXing-cpp and WeChat QRCode backends.

Follows:
- OCP (Open/Closed Principle): Easy to extend with new backends
- DIP (Dependency Inversion): Returns IQrDetector interface
- Factory Pattern: Encapsulates object creation logic
"""

import logging
from typing import Optional, List

from core.interfaces.qr_detector_interface import IQrDetector


logger = logging.getLogger(__name__)


def createQrDetector(
    backend: str = "zxing",
    # ZXing params (prefixed with 'zxing')
    zxingTryRotate: bool = True,
    zxingTryDownscale: bool = True,
    # WeChat params (prefixed with 'wechat')
    wechatModelDir: str = "models/wechat"
) -> IQrDetector:
    """
    Factory function to create QR detector based on backend.
    
    Supports:
    - "zxing": ZXing-cpp backend (fast, cross-platform)
    - "wechat": OpenCV WeChat QRCode backend (deep learning based)
    
    Args:
        backend: Backend name ("zxing" or "wechat").
        zxingTryRotate: (zxing) Try rotated barcodes (90/270 degrees).
        zxingTryDownscale: (zxing) Try downscaled versions for better detection.
        wechatModelDir: (wechat) Directory containing WeChat QR model files.
        
    Returns:
        IQrDetector: QR detector instance implementing IQrDetector interface.
        
    Raises:
        ValueError: If backend is invalid or not supported.
        ImportError: If required library is not installed.
        
    Examples:
        >>> # Create ZXing detector (default)
        >>> detector = createQrDetector(
        ...     backend="zxing",
        ...     zxingTryRotate=True,
        ...     zxingTryDownscale=True
        ... )
        
        >>> # Create WeChat detector
        >>> detector = createQrDetector(
        ...     backend="wechat",
        ...     wechatModelDir="models/wechat"
        ... )
    """
    # Normalize backend name
    backend = backend.lower().strip()
    
    # Validate backend
    supportedBackends = getSupportedQrBackends()
    if backend not in supportedBackends:
        errorMsg = (
            f"Invalid QR backend: '{backend}'. "
            f"Supported backends: {supportedBackends}"
        )
        logger.error(errorMsg)
        raise ValueError(errorMsg)
    
    # Create detector based on backend
    if backend == "wechat":
        return _createWechatDetector(
            wechatModelDir=wechatModelDir
        )
    
    elif backend == "zxing":
        return _createZxingDetector(
            zxingTryRotate=zxingTryRotate,
            zxingTryDownscale=zxingTryDownscale
        )
    
    # Should never reach here due to validation above
    raise ValueError(f"Unsupported QR backend: {backend}")


def _createZxingDetector(
    zxingTryRotate: bool,
    zxingTryDownscale: bool
) -> IQrDetector:
    """
    Create ZXing QR detector instance.
    
    Args:
        zxingTryRotate: Try rotated barcodes.
        zxingTryDownscale: Try downscaled versions.
        
    Returns:
        IQrDetector: ZXing QR detector instance.
        
    Raises:
        ImportError: If zxing-cpp is not installed.
    """
    try:
        from core.qr.zxing_qr_detector import ZxingQrDetector
        
        logger.info(
            f"Creating ZXing QR detector "
            f"(tryRotate={zxingTryRotate}, tryDownscale={zxingTryDownscale})"
        )
        
        detector = ZxingQrDetector(
            tryRotate=zxingTryRotate,
            tryDownscale=zxingTryDownscale
        )
        
        return detector
        
    except ImportError as e:
        errorMsg = (
            "ZXing-cpp is not installed. "
            "Install with: pip install zxing-cpp"
        )
        logger.error(errorMsg)
        logger.error(f"Import error details: {e}")
        raise ImportError(errorMsg) from e


def _createWechatDetector(
    wechatModelDir: str
) -> IQrDetector:
    """
    Create WeChat QR detector instance.
    
    Args:
        wechatModelDir: Directory containing model files.
        
    Returns:
        IQrDetector: WeChat QR detector instance.
        
    Raises:
        ImportError: If opencv-contrib-python is not installed.
    """
    try:
        from core.qr.wechat_qr_detector import WechatQrDetector
        
        logger.info(
            f"Creating WeChat QR detector "
            f"(modelDir={wechatModelDir})"
        )
        
        detector = WechatQrDetector(
            modelDir=wechatModelDir
        )
        
        return detector
        
    except ImportError as e:
        errorMsg = (
            "WeChat QRCode requires opencv-contrib-python. "
            "Install with: pip install opencv-contrib-python"
        )
        logger.error(errorMsg)
        logger.error(f"Import error details: {e}")
        raise ImportError(errorMsg) from e


def getSupportedQrBackends() -> List[str]:
    """
    Get list of supported QR backend names.
    
    Returns:
        List[str]: List of backend names ["zxing", "wechat"].
    """
    return ["zxing", "wechat"]


def isQrBackendAvailable(backend: str) -> bool:
    """
    Check if a QR backend is available (library installed).
    
    Args:
        backend: Backend name ("zxing" or "wechat").
        
    Returns:
        bool: True if backend library is installed and available.
    """
    backend = backend.lower().strip()
    
    if backend == "zxing":
        try:
            import zxingcpp
            return True
        except ImportError:
            return False
    
    elif backend == "wechat":
        try:
            import cv2
            # Check if wechat_qrcode module is available
            _ = cv2.wechat_qrcode.WeChatQRCode
            return True
        except (ImportError, AttributeError):
            return False
    
    return False
