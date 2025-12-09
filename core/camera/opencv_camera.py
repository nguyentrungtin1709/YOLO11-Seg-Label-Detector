"""
OpenCV Camera Implementation

Implements ICameraCapture using OpenCV's VideoCapture.
Follows SRP: Only handles camera capture operations.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import cv2

from core.interfaces.camera_interface import ICameraCapture, CameraInfo


logger = logging.getLogger(__name__)


class OpenCVCamera(ICameraCapture):
    """
    Camera capture implementation using OpenCV VideoCapture.
    
    Supports USB cameras, built-in cameras, and other devices
    accessible through OpenCV's VideoCapture interface.
    
    Follows SRP: Only responsible for camera capture operations.
    """
    
    def __init__(self, maxCameraSearch: int = 10):
        """
        Initialize OpenCVCamera.
        
        Args:
            maxCameraSearch: Maximum number of camera indices to search for available cameras.
        """
        self._capture: Optional[cv2.VideoCapture] = None
        self._cameraIndex: int = -1
        self._maxCameraSearch = maxCameraSearch
    
    def listAvailableCameras(self) -> List[CameraInfo]:
        """
        List all available camera devices by probing camera indices.
        
        Returns:
            List[CameraInfo]: List of available cameras.
        """
        cameras = []
        
        for index in range(self._maxCameraSearch):
            try:
                tempCapture = cv2.VideoCapture(index)
                if tempCapture.isOpened():
                    # Try to read a frame to confirm camera is working
                    ret, _ = tempCapture.read()
                    if ret:
                        cameras.append(CameraInfo(index=index, name=f"Camera {index}"))
                    tempCapture.release()
            except Exception as e:
                logger.debug(f"Error probing camera {index}: {e}")
                continue
        
        if not cameras:
            logger.warning("No cameras found in the system")
        else:
            logger.info(f"Found {len(cameras)} camera(s)")
        
        return cameras
    
    def open(self, cameraIndex: int, width: int = 640, height: int = 640) -> bool:
        """
        Open a camera device by its index.
        
        Args:
            cameraIndex: The index of the camera to open.
            width: Desired frame width.
            height: Desired frame height.
            
        Returns:
            bool: True if camera opened successfully.
        """
        # Release any existing camera first
        if self._capture is not None:
            self.release()
        
        try:
            self._capture = cv2.VideoCapture(cameraIndex)
            
            if self._capture.isOpened():
                self._cameraIndex = cameraIndex
                
                # Set camera properties
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                logger.info(f"Camera {cameraIndex} opened successfully ({width}x{height})")
                return True
            else:
                logger.error(f"Failed to open camera {cameraIndex}")
                self._capture = None
                return False
                
        except Exception as e:
            logger.error(f"Error opening camera {cameraIndex}: {e}")
            self._capture = None
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the opened camera.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success flag and frame.
        """
        if self._capture is None or not self._capture.isOpened():
            return (False, None)
        
        try:
            ret, frame = self._capture.read()
            return (ret, frame if ret else None)
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return (False, None)
    
    def release(self) -> None:
        """
        Release the camera device and free resources.
        """
        if self._capture is not None:
            try:
                self._capture.release()
                logger.info(f"Camera {self._cameraIndex} released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            finally:
                self._capture = None
                self._cameraIndex = -1
    
    def isOpened(self) -> bool:
        """
        Check if a camera is currently opened.
        
        Returns:
            bool: True if camera is opened.
        """
        return self._capture is not None and self._capture.isOpened()
    
    def getCameraIndex(self) -> int:
        """
        Get the current camera index.
        
        Returns:
            int: Current camera index, or -1 if no camera is opened.
        """
        return self._cameraIndex
