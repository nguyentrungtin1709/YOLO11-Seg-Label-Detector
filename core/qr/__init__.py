"""QR Detection module."""

from core.qr.zxing_qr_detector import ZxingQrDetector
from core.qr.wechat_qr_detector import WechatQrDetector
from core.qr.qr_detector_factory import (
    createQrDetector,
    getSupportedQrBackends,
    isQrBackendAvailable
)
from core.qr.qr_image_preprocessor import QrImagePreprocessor

__all__ = [
    'ZxingQrDetector',
    'WechatQrDetector',
    'createQrDetector',
    'getSupportedQrBackends',
    'isQrBackendAvailable',
    'QrImagePreprocessor'
]
