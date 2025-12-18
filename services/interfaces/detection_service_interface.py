"""
Detection Service Interface Module.

Defines the interface for object detection operations (Step 2 of the pipeline).
Responsible for running YOLO detection and filtering results.

Follows:
- SRP: Only handles detection operations
- DIP: Depends on IDetector abstraction from core layer
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from core.interfaces.detector_interface import Detection


@dataclass
class DetectionServiceResult:
    """
    Result of the detection service.
    
    Attributes:
        detections: List of filtered detections.
        annotatedFrame: Frame with detection overlays (optional).
        frameId: Frame identifier from CameraFrame.
        success: Whether detection was successful.
        processingTimeMs: Time taken for detection.
    """
    detections: List[Detection]
    annotatedFrame: Optional[np.ndarray]
    frameId: str
    success: bool
    processingTimeMs: float = 0.0


class IDetectionService(ABC):
    """
    Interface for detection operations (Step 2).
    
    Runs YOLO instance segmentation on frames and filters results
    based on confidence threshold, area ratio, and top N selection.
    """
    
    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        frameId: str
    ) -> DetectionServiceResult:
        """
        Run detection on a frame.
        
        Args:
            frame: Input frame (BGR format).
            frameId: Frame identifier for debug output.
            
        Returns:
            DetectionServiceResult: Detection results with metadata.
        """
        pass
    
    @abstractmethod
    def loadModel(self, modelPath: str) -> bool:
        """
        Load the detection model.
        
        Args:
            modelPath: Path to the ONNX model file.
            
        Returns:
            bool: True if model loaded successfully.
        """
        pass
    
    @abstractmethod
    def setConfidenceThreshold(self, threshold: float) -> None:
        """
        Set the confidence threshold for filtering detections.
        
        Args:
            threshold: Confidence threshold (0.0 to 1.0).
        """
        pass
    
    @abstractmethod
    def getConfidenceThreshold(self) -> float:
        """
        Get the current confidence threshold.
        
        Returns:
            float: Current confidence threshold.
        """
        pass
    
    @abstractmethod
    def setEnabled(self, enabled: bool) -> None:
        """
        Enable or disable detection.
        
        Args:
            enabled: True to enable detection.
        """
        pass
    
    @abstractmethod
    def isEnabled(self) -> bool:
        """
        Check if detection is enabled.
        
        Returns:
            bool: True if detection is enabled.
        """
        pass
    
    @abstractmethod
    def isModelLoaded(self) -> bool:
        """
        Check if a model is loaded.
        
        Returns:
            bool: True if model is loaded.
        """
        pass
