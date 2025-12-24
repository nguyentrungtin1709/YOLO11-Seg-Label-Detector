"""
Detector Factory Module

Factory function for creating detector instances based on backend selection.
Supports ONNX Runtime and OpenVINO Runtime backends.

Follows:
- OCP (Open/Closed Principle): Easy to extend with new backends
- DIP (Dependency Inversion): Returns IDetector interface
- Factory Pattern: Encapsulates object creation logic
"""

import logging
from typing import Optional, List

from core.interfaces.detector_interface import IDetector


logger = logging.getLogger(__name__)


def createDetector(
    backend: str = "onnx",
    modelPath: str = "",
    inputSize: int = 640,
    classNames: Optional[List[str]] = None,
    isSegmentation: bool = False,
    openvinoConfig: Optional[dict] = None
) -> IDetector:
    """
    Factory function to create detector based on backend.
    
    Supports:
    - "onnx": ONNX Runtime backend (cross-platform, FP32)
    - "openvino": OpenVINO Runtime backend (Intel-optimized, INT8 support)
    
    Args:
        backend: Backend name ("onnx" or "openvino").
        modelPath: Path to model file (.onnx for ONNX, .xml for OpenVINO).
        inputSize: Model input size (default: 640).
        classNames: List of class names the model can detect.
        isSegmentation: If True, enable instance segmentation with mask output.
        openvinoConfig: OpenVINO-specific performance configuration dict with keys:
            - numThreads: Number of CPU threads (0 = auto)
            - numStreams: Number of inference streams (0 = auto)
            - performanceHint: 'LATENCY' or 'THROUGHPUT'
            - enableHyperThreading: Enable hyper-threading
            - enableCpuPinning: Pin threads to CPU cores
        
    Returns:
        IDetector: Detector instance implementing IDetector interface.
        
    Raises:
        ValueError: If backend is invalid or not supported.
        ImportError: If required library is not installed.
        
    Examples:
        >>> # Create ONNX detector
        >>> detector = createDetector(
        ...     backend="onnx",
        ...     modelPath="models/yolo11n-seg.onnx",
        ...     inputSize=640,
        ...     classNames=["label"],
        ...     isSegmentation=True
        ... )
        
        >>> # Create OpenVINO detector with performance config
        >>> detector = createDetector(
        ...     backend="openvino",
        ...     modelPath="models/yolo11n-seg.xml",
        ...     inputSize=640,
        ...     classNames=["label"],
        ...     isSegmentation=True,
        ...     openvinoConfig={
        ...         "numThreads": 4,
        ...         "performanceHint": "LATENCY"
        ...     }
        ... )
    """
    # Normalize backend name
    backend = backend.lower().strip()
    
    # Validate backend
    supportedBackends = ["onnx", "openvino"]
    if backend not in supportedBackends:
        errorMsg = (
            f"Invalid backend: '{backend}'. "
            f"Supported backends: {supportedBackends}"
        )
        logger.error(errorMsg)
        raise ValueError(errorMsg)
    
    # Create detector based on backend
    if backend == "openvino":
        return _createOpenVINODetector(
            modelPath=modelPath,
            inputSize=inputSize,
            classNames=classNames,
            isSegmentation=isSegmentation,
            openvinoConfig=openvinoConfig
        )
    
    elif backend == "onnx":
        return _createOnnxDetector(
            modelPath=modelPath,
            inputSize=inputSize,
            classNames=classNames,
            isSegmentation=isSegmentation
        )
    
    # Should never reach here due to validation above
    raise ValueError(f"Unsupported backend: {backend}")


def _createOpenVINODetector(
    modelPath: str,
    inputSize: int,
    classNames: Optional[List[str]],
    isSegmentation: bool,
    openvinoConfig: Optional[dict] = None
) -> IDetector:
    """
    Create OpenVINO detector instance.
    
    Args:
        modelPath: Path to OpenVINO IR model file (.xml).
        inputSize: Model input size.
        classNames: List of class names.
        isSegmentation: Enable segmentation mode.
        openvinoConfig: OpenVINO performance configuration dict.
        
    Returns:
        IDetector: OpenVINO detector instance.
        
    Raises:
        ImportError: If OpenVINO is not installed.
    """
    try:
        from core.detector.openvino_detector import OpenVINODetector
        
        # Extract OpenVINO config with defaults
        config = openvinoConfig or {}
        numThreads = config.get("numThreads", 0)
        numStreams = config.get("numStreams", 0)
        performanceHint = config.get("performanceHint", "LATENCY")
        enableHyperThreading = config.get("enableHyperThreading", False)
        enableCpuPinning = config.get("enableCpuPinning", True)
        
        logger.info(
            f"Creating OpenVINO detector (inputSize={inputSize}, segmentation={isSegmentation}, "
            f"threads={numThreads}, streams={numStreams}, hint={performanceHint})"
        )
        detector = OpenVINODetector(
            inputSize=inputSize,
            classNames=classNames,
            isSegmentation=isSegmentation,
            numThreads=numThreads,
            numStreams=numStreams,
            performanceHint=performanceHint,
            enableHyperThreading=enableHyperThreading,
            enableCpuPinning=enableCpuPinning
        )
        
        # Load model if path is provided
        if modelPath:
            success = detector.loadModel(modelPath)
            if not success:
                logger.warning(f"Failed to load OpenVINO model: {modelPath}")
        
        return detector
        
    except ImportError as e:
        errorMsg = (
            "OpenVINO Runtime is not installed. "
            "Install with: pip install openvino>=2024.0.0"
        )
        logger.error(errorMsg)
        logger.error(f"Import error details: {e}")
        raise ImportError(errorMsg) from e


def _createOnnxDetector(
    modelPath: str,
    inputSize: int,
    classNames: Optional[List[str]],
    isSegmentation: bool
) -> IDetector:
    """
    Create ONNX Runtime detector instance.
    
    Args:
        modelPath: Path to ONNX model file (.onnx).
        inputSize: Model input size.
        classNames: List of class names.
        isSegmentation: Enable segmentation mode.
        
    Returns:
        IDetector: ONNX detector instance.
        
    Raises:
        ImportError: If ONNX Runtime is not installed.
    """
    try:
        from core.detector.yolo_detector import YOLODetector
        
        logger.info(f"Creating ONNX detector (inputSize={inputSize}, segmentation={isSegmentation})")
        detector = YOLODetector(
            inputSize=inputSize,
            classNames=classNames,
            isSegmentation=isSegmentation
        )
        
        # Load model if path is provided
        if modelPath:
            success = detector.loadModel(modelPath)
            if not success:
                logger.warning(f"Failed to load ONNX model: {modelPath}")
        
        return detector
        
    except ImportError as e:
        errorMsg = (
            "ONNX Runtime is not installed. "
            "Install with: pip install onnxruntime>=1.16.0"
        )
        logger.error(errorMsg)
        logger.error(f"Import error details: {e}")
        raise ImportError(errorMsg) from e


def getSupportedBackends() -> List[str]:
    """
    Get list of supported backend names.
    
    Returns:
        List[str]: List of backend names ["onnx", "openvino"].
    """
    return ["onnx", "openvino"]


def isBackendAvailable(backend: str) -> bool:
    """
    Check if a backend is available (library installed).
    
    Args:
        backend: Backend name ("onnx" or "openvino").
        
    Returns:
        bool: True if backend library is installed and available.
    """
    backend = backend.lower().strip()
    
    if backend == "onnx":
        try:
            import onnxruntime
            return True
        except ImportError:
            return False
    
    elif backend == "openvino":
        try:
            from openvino.runtime import Core
            return True
        except ImportError:
            return False
    
    return False
