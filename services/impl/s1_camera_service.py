"""
S1 Camera Service Implementation.

Step 1 of the pipeline: Camera capture and frame ID generation.
Creates and manages OpenCV camera from core layer.

Follows:
- SRP: Only handles camera operations
- DIP: Depends on ICameraCapture abstraction (interface)
- OCP: Extends without modifying existing code
"""

import time
import logging
from datetime import datetime
from typing import List, Optional

import numpy as np

from core.interfaces.camera_interface import ICameraCapture, CameraInfo
from core.camera.opencv_camera import OpenCVCamera
from services.interfaces.camera_service_interface import (
    ICameraService,
    CameraFrame
)
from services.interfaces.base_service_interface import BaseService


class S1CameraService(ICameraService, BaseService):
    """
    Step 1: Camera Service Implementation.
    
    Captures frames from camera and generates unique frame IDs
    for tracking through the pipeline.
    
    Creates OpenCVCamera internally with provided parameters.
    """
    
    SERVICE_NAME = "s1_camera"
    
    def __init__(
        self,
        frameWidth: int = 640,
        frameHeight: int = 640,
        maxCameraSearch: int = 2,
        debugBasePath: str = "output/debug",
        debugEnabled: bool = False
    ):
        """
        Initialize S1CameraService.
        
        Args:
            frameWidth: Default frame width.
            frameHeight: Default frame height.
            maxCameraSearch: Maximum number of camera indices to search.
            debugBasePath: Base path for debug output.
            debugEnabled: Whether to save debug output.
        """
        BaseService.__init__(
            self,
            serviceName=self.SERVICE_NAME,
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        
        # Create core camera implementation
        self._cameraCapture: ICameraCapture = OpenCVCamera(
            maxCameraSearch=maxCameraSearch
        )
        
        self._frameWidth = frameWidth
        self._frameHeight = frameHeight
        self._currentCameraIndex: Optional[int] = None
        
        self._logger.info(
            f"S1CameraService initialized "
            f"(frameSize={frameWidth}x{frameHeight}, maxCameraSearch={maxCameraSearch})"
        )
    
    def captureFrame(self) -> CameraFrame:
        """
        Capture a single frame from the camera.
        
        Generates a unique frameId based on timestamp.
        """
        startTime = time.time()
        
        # Generate timestamp and frameId
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
        frameId = f"frame_{timestamp}"
        
        # Check if camera is open
        if self._currentCameraIndex is None:
            self._logger.warning("No camera is open")
            return CameraFrame(
                image=None,
                frameId=frameId,
                timestamp=timestamp,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        # Capture frame
        success, frame = self._cameraCapture.read()
        
        if not success or frame is None:
            self._logger.warning(f"[{frameId}] Failed to capture frame")
            return CameraFrame(
                image=None,
                frameId=frameId,
                timestamp=timestamp,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        processingTimeMs = self._measureTime(startTime)
        
        # Save debug output
        self._saveDebugImage(frameId, frame)
        
        # Log timing
        self._logTiming(frameId, processingTimeMs)
        
        return CameraFrame(
            image=frame,
            frameId=frameId,
            timestamp=timestamp,
            success=True,
            processingTimeMs=processingTimeMs
        )
    
    def getAvailableCameras(self) -> List[CameraInfo]:
        """List all available camera devices."""
        cameras = self._cameraCapture.listAvailableCameras()
        self._logger.info(f"Found {len(cameras)} available cameras")
        return cameras
    
    def openCamera(
        self, 
        index: int, 
        width: int = 640, 
        height: int = 640
    ) -> bool:
        """Open a camera device."""
        # Close existing camera if open
        if self._currentCameraIndex is not None:
            self.closeCamera()
        
        success = self._cameraCapture.open(index, width, height)
        
        if success:
            self._currentCameraIndex = index
            self._frameWidth = width
            self._frameHeight = height
            self._logger.info(f"Camera {index} opened ({width}x{height})")
        else:
            self._logger.error(f"Failed to open camera {index}")
        
        return success
    
    def closeCamera(self) -> None:
        """Close the current camera device."""
        if self._currentCameraIndex is not None:
            self._cameraCapture.release()
            self._logger.info(f"Camera {self._currentCameraIndex} closed")
            self._currentCameraIndex = None
    
    def isOpened(self) -> bool:
        """Check if a camera is currently open."""
        return self._currentCameraIndex is not None and self._cameraCapture.isOpened()
    
    def getCurrentCameraIndex(self) -> Optional[int]:
        """Get the index of the currently open camera."""
        return self._currentCameraIndex
