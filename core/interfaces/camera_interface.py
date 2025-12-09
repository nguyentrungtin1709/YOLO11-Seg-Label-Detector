"""
Camera Interface Module

Defines the abstract interface for camera capture operations.
Follows ISP (Interface Segregation Principle): Only contains camera-related methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class CameraInfo:
    """Data class representing camera device information."""
    index: int
    name: str
    
    def __str__(self) -> str:
        return self.name


class ICameraCapture(ABC):
    """
    Abstract interface for camera capture operations.
    
    Implementations should handle video capture from various sources
    (USB cameras, built-in cameras, IP cameras, etc.)
    
    Follows ISP: Only contains methods related to camera capture.
    """
    
    @abstractmethod
    def listAvailableCameras(self) -> List[CameraInfo]:
        """
        List all available camera devices in the system.
        
        Returns:
            List[CameraInfo]: List of available camera devices with index and name.
        """
        pass
    
    @abstractmethod
    def open(self, cameraIndex: int, width: int = 640, height: int = 640) -> bool:
        """
        Open a camera device by its index.
        
        Args:
            cameraIndex: The index of the camera to open.
            width: Desired frame width.
            height: Desired frame height.
            
        Returns:
            bool: True if camera opened successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the opened camera.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: 
                - First element: True if frame read successfully, False otherwise.
                - Second element: The frame as numpy array (BGR format), or None if failed.
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """
        Release the camera device and free resources.
        """
        pass
    
    @abstractmethod
    def isOpened(self) -> bool:
        """
        Check if a camera is currently opened.
        
        Returns:
            bool: True if camera is opened, False otherwise.
        """
        pass
