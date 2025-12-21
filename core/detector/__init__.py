"""
Detector module for object detection and instance segmentation.

Provides:
- YOLODetector: ONNX Runtime implementation
- OpenVINODetector: OpenVINO Runtime implementation  
- createDetector: Factory function to create detector based on backend

Usage:
    >>> from core.detector import createDetector
    >>> 
    >>> # Create ONNX detector
    >>> detector = createDetector(backend="onnx", modelPath="model.onnx", ...)
    >>> 
    >>> # Create OpenVINO detector
    >>> detector = createDetector(backend="openvino", modelPath="model.xml", ...)
"""

from core.detector.detector_factory import (
    createDetector,
    getSupportedBackends,
    isBackendAvailable
)

__all__ = [
    'createDetector',
    'getSupportedBackends',
    'isBackendAvailable',
]
