# Services module for Label Detector
# Contains business logic services

from services.camera_service import CameraService
from services.detection_service import DetectionService
from services.image_saver_service import ImageSaverService

__all__ = [
    "CameraService",
    "DetectionService",
    "ImageSaverService",
]