# Core module for Label Detector
# Contains interfaces and implementations for camera, detector, and writer

from core.interfaces.camera_interface import ICameraCapture, CameraInfo
from core.interfaces.detector_interface import IDetector, Detection
from core.interfaces.writer_interface import IImageWriter
from core.camera.opencv_camera import OpenCVCamera
from core.detector.yolo_detector import YOLODetector
from core.writer.local_writer import LocalImageWriter

__all__ = [
    "ICameraCapture",
    "CameraInfo",
    "IDetector",
    "Detection",
    "IImageWriter",
    "OpenCVCamera",
    "YOLODetector",
    "LocalImageWriter",
]