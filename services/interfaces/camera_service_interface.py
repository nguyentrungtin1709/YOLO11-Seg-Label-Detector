"""
Camera Service Interface Module.

Defines the interface for camera operations (Step 1 of the pipeline).
Responsible for capturing frames and generating unique frame identifiers.

Follows:
- SRP: Only handles camera operations
- ISP: Minimal interface for camera functionality
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from core.interfaces.camera_interface import CameraInfo


@dataclass
class CameraFrame:
    """
    Result of a frame capture operation.
    
    Attributes:
        image: Captured frame as numpy array (BGR format).
        frameId: Unique identifier for this frame (e.g., "frame_20251218_024810_535").
        timestamp: Timestamp string for debug file naming.
        success: Whether the capture was successful.
        processingTimeMs: Time taken to capture the frame.
    """
    image: Optional[np.ndarray]
    frameId: str
    timestamp: str
    success: bool
    processingTimeMs: float = 0.0


class ICameraService(ABC):
    """
    Interface for camera operations (Step 1).
    
    Handles camera device management and frame capture.
    Generates unique frame IDs that are passed through the pipeline.
    """
    
    @abstractmethod
    def captureFrame(self) -> CameraFrame:
        """
        Capture a single frame from the camera.
        
        Creates a unique frameId based on timestamp that will be used
        throughout the pipeline for debug file naming consistency.
        
        Returns:
            CameraFrame: Captured frame with metadata.
        """
        pass
    
    @abstractmethod
    def getAvailableCameras(self) -> List[CameraInfo]:
        """
        List all available camera devices.
        
        Returns:
            List[CameraInfo]: Available cameras with index and name.
        """
        pass
    
    @abstractmethod
    def openCamera(self, index: int, width: int = 640, height: int = 640) -> bool:
        """
        Open a camera device.
        
        Args:
            index: Camera device index.
            width: Desired frame width.
            height: Desired frame height.
            
        Returns:
            bool: True if camera opened successfully.
        """
        pass
    
    @abstractmethod
    def closeCamera(self) -> None:
        """Close the current camera device."""
        pass
    
    @abstractmethod
    def isOpened(self) -> bool:
        """
        Check if a camera is currently open.
        
        Returns:
            bool: True if camera is open and ready.
        """
        pass
    
    @abstractmethod
    def getCurrentCameraIndex(self) -> Optional[int]:
        """
        Get the index of the currently open camera.
        
        Returns:
            Optional[int]: Camera index or None if no camera is open.
        """
        pass
