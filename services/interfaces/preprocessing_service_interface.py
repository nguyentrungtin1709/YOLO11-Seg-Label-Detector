"""
Preprocessing Service Interface Module.

Defines the interface for image preprocessing operations (Step 3 of the pipeline).
Responsible for cropping, rotating, and fixing orientation of detected labels.

Follows:
- SRP: Only handles preprocessing operations
- DIP: Depends on IImagePreprocessor abstraction from core layer
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from core.interfaces.detector_interface import Detection


@dataclass
class PreprocessingServiceResult:
    """
    Result of the preprocessing service.
    
    Attributes:
        croppedImage: Image after crop, rotate, and orientation fix.
        rotationAngle: Angle of rotation applied.
        orientationFixed: True if 180° orientation fix was applied.
        contourPoints: Contour points used for cropping.
        frameId: Frame identifier for debug output.
        success: Whether preprocessing was successful.
        processingTimeMs: Time taken for preprocessing.
    """
    croppedImage: Optional[np.ndarray]
    rotationAngle: float
    orientationFixed: bool
    contourPoints: Optional[List[List[int]]]
    frameId: str
    success: bool
    processingTimeMs: float = 0.0


class IPreprocessingService(ABC):
    """
    Interface for preprocessing operations (Step 3).
    
    Crops detected labels using segmentation masks, rotates to horizontal,
    and applies AI-based orientation correction (180° fix).
    """
    
    @abstractmethod
    def preprocess(
        self,
        frame: np.ndarray,
        detection: Detection,
        frameId: str
    ) -> PreprocessingServiceResult:
        """
        Preprocess a detected label.
        
        Extracts the label region using the segmentation mask,
        rotates to horizontal orientation, and applies AI 180° fix.
        
        Args:
            frame: Original frame (BGR format).
            detection: Detection with bounding box and mask.
            frameId: Frame identifier for debug output.
            
        Returns:
            PreprocessingServiceResult: Preprocessed image with metadata.
        """
        pass
    
    @abstractmethod
    def setEnabled(self, enabled: bool) -> None:
        """
        Enable or disable preprocessing.
        
        Args:
            enabled: True to enable preprocessing.
        """
        pass
    
    @abstractmethod
    def isEnabled(self) -> bool:
        """
        Check if preprocessing is enabled.
        
        Returns:
            bool: True if preprocessing is enabled.
        """
        pass
    
    @abstractmethod
    def setForceLandscape(self, enabled: bool) -> None:
        """
        Set force landscape orientation option.
        
        Args:
            enabled: True to force landscape (width >= height).
        """
        pass
    
    @abstractmethod
    def isForceLandscape(self) -> bool:
        """
        Check if force landscape is enabled.
        
        Returns:
            bool: True if force landscape is enabled.
        """
        pass
    
    @abstractmethod
    def setAiOrientationFix(self, enabled: bool) -> None:
        """
        Set AI orientation fix option.
        
        Args:
            enabled: True to enable AI 180° orientation fix.
        """
        pass
    
    @abstractmethod
    def isAiOrientationFix(self) -> bool:
        """
        Check if AI orientation fix is enabled.
        
        Returns:
            bool: True if AI orientation fix is enabled.
        """
        pass
    
    @abstractmethod
    def isAiAvailable(self) -> bool:
        """
        Check if AI orientation model is available.
        
        Returns:
            bool: True if AI model is loaded and ready.
        """
        pass
