"""
Detector Interface Module

Defines the abstract interface for object detection and instance segmentation operations.
Follows OCP (Open/Closed Principle): New detectors can be added without modifying existing code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Detection:
    """
    Data class representing a single detection result.
    
    Attributes:
        bbox: Bounding box coordinates as (x1, y1, x2, y2).
        className: The class name of the detected object.
        confidence: Confidence score of the detection (0.0 to 1.0).
        mask: Optional segmentation mask (H x W) for instance segmentation.
    """
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    className: str
    confidence: float
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # Binary mask (H x W)
    
    def __repr__(self) -> str:
        maskInfo = f", mask={self.mask.shape}" if self.mask is not None else ""
        return f"Detection({self.className}: {self.confidence:.2f} @ {self.bbox}{maskInfo})"


class IDetector(ABC):
    """
    Abstract interface for object detection.
    
    Implementations should handle model loading, preprocessing,
    inference, and postprocessing for various detection models.
    
    Follows OCP: Can extend with new detector implementations without modifying this interface.
    """
    
    @abstractmethod
    def loadModel(self, modelPath: str) -> bool:
        """
        Load a detection model from file.
        
        Args:
            modelPath: Path to the model file (e.g., ONNX, PyTorch, etc.)
            
        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray, confidenceThreshold: float) -> List[Detection]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            confidenceThreshold: Minimum confidence score to include a detection.
            
        Returns:
            List[Detection]: List of detected objects with bounding boxes, classes, and confidence scores.
        """
        pass
    
    @abstractmethod
    def getClassNames(self) -> List[str]:
        """
        Get the list of class names the model can detect.
        
        Returns:
            List[str]: List of class names.
        """
        pass
