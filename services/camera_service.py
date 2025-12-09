"""
Camera Service

Orchestrates camera operations using ICameraCapture.
Follows DIP: Depends on abstraction (ICameraCapture), not concrete implementation.
"""

import logging
from typing import Optional
import numpy as np

from core.interfaces.camera_interface import ICameraCapture, CameraInfo


logger = logging.getLogger(__name__)


class CameraService:
    """
    Service for managing camera operations.
    
    Provides high-level camera control including listing cameras,
    opening/closing streams, and reading frames.
    
    Follows:
    - SRP: Only handles camera-related orchestration
    - DIP: Depends on ICameraCapture abstraction
    """
    
    def __init__(self, cameraCapture: ICameraCapture):
        """
        Initialize CameraService.
        
        Args:
            cameraCapture: Camera capture implementation.
        """
        self._cameraCapture = cameraCapture
        self._currentCameraIndex: Optional[int] = None
    
    def getAvailableCameras(self) -> list[CameraInfo]:
        """
        Get list of available cameras.
        
        Returns:
            List of CameraInfo for available cameras.
        """
        cameras = self._cameraCapture.listAvailableCameras()
        logger.info(f"Found {len(cameras)} available cameras")
        return cameras
    
    def openCamera(self, index: int, width: int = 640, height: int = 640) -> bool:
        """
        Open a camera by index.
        
        Closes any previously opened camera first.
        
        Args:
            index: Camera index to open.
            width: Desired frame width.
            height: Desired frame height.
            
        Returns:
            bool: True if camera opened successfully.
        """
        # Close existing camera if open
        if self._currentCameraIndex is not None:
            self.closeCamera()
        
        success = self._cameraCapture.open(index, width, height)
        if success:
            self._currentCameraIndex = index
            logger.info(f"Camera {index} opened ({width}x{height})")
        else:
            logger.error(f"Failed to open camera {index}")
        
        return success
    
    def closeCamera(self) -> None:
        """Close the current camera."""
        self._cameraCapture.release()
        logger.info(f"Camera {self._currentCameraIndex} closed")
        self._currentCameraIndex = None
    
    def readFrame(self) -> Optional[np.ndarray]:
        """
        Read a frame from the current camera.
        
        Returns:
            Frame as numpy array (BGR), or None if failed.
        """
        if self._currentCameraIndex is None:
            return None
        
        success, frame = self._cameraCapture.read()
        return frame if success else None
    
    def isOpened(self) -> bool:
        """
        Check if a camera is currently open.
        
        Returns:
            bool: True if camera is open.
        """
        return self._cameraCapture.isOpened()
    
    @property
    def currentCameraIndex(self) -> Optional[int]:
        """Get the index of the currently open camera."""
        return self._currentCameraIndex
