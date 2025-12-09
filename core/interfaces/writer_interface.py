"""
Writer Interface Module

Defines the abstract interface for image writing operations.
Follows ISP: Only contains methods related to image saving.
"""

from abc import ABC, abstractmethod
import numpy as np


class IImageWriter(ABC):
    """
    Abstract interface for image writing operations.
    
    Implementations should handle saving images to various destinations
    (local filesystem, cloud storage, etc.)
    
    Follows ISP: Only contains methods related to image writing.
    """
    
    @abstractmethod
    def save(self, image: np.ndarray, filepath: str) -> bool:
        """
        Save an image to the specified filepath.
        
        Args:
            image: Image as numpy array (BGR format from OpenCV).
            filepath: Destination path for the image file.
            
        Returns:
            bool: True if image saved successfully, False otherwise.
        """
        pass
