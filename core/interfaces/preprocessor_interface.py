"""
Preprocessor Interface Module

Defines the abstract interface for image preprocessing operations.
Follows ISP (Interface Segregation Principle): Only contains preprocessing-related methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PreprocessingResult:
    """
    Data class representing the result of image preprocessing.
    
    Attributes:
        image: The preprocessed image as numpy array (BGR format).
        success: Whether preprocessing was successful.
        message: Optional message describing the result or any errors.
    """
    image: Optional[np.ndarray]
    success: bool
    message: str = ""
    
    def __repr__(self) -> str:
        if self.image is not None:
            return f"PreprocessingResult(success={self.success}, shape={self.image.shape})"
        return f"PreprocessingResult(success={self.success}, message={self.message})"


class IImagePreprocessor(ABC):
    """
    Abstract interface for image preprocessing operations.
    
    Implementations should handle cropping, rotation, orientation correction,
    and other preprocessing steps for document/label images.
    
    Follows ISP: Only contains methods related to image preprocessing.
    """
    
    @abstractmethod
    def process(
        self, 
        image: np.ndarray, 
        maskPoints: np.ndarray,
        forceLandscape: bool = True,
        useAiOrientationFix: bool = True
    ) -> PreprocessingResult:
        """
        Preprocess an image based on mask polygon points.
        
        Pipeline:
        1. Crop and rotate image based on minimum area rectangle of mask
        2. Force landscape orientation if requested
        3. Fix 180-degree rotation using AI if requested
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            maskPoints: Polygon points defining the region of interest (N x 2 array).
            forceLandscape: If True, ensure output is landscape orientation.
            useAiOrientationFix: If True, use AI to detect and fix 180-degree rotation.
            
        Returns:
            PreprocessingResult: Contains the preprocessed image and status.
        """
        pass
    
    @abstractmethod
    def isAiAvailable(self) -> bool:
        """
        Check if AI orientation correction is available.
        
        Returns:
            bool: True if PaddleOCR is loaded and ready.
        """
        pass
