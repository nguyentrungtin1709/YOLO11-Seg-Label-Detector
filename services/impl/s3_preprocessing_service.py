"""
S3 Preprocessing Service Implementation.

Step 3 of the pipeline: Crop, rotate, and orientation fix.
Creates and manages DocumentPreprocessor from core layer.

Follows:
- SRP: Only handles preprocessing operations
- DIP: Depends on IImagePreprocessor abstraction (interface)
- OCP: Extends without modifying existing code
"""

import time
import logging
from typing import Optional, List

import cv2
import numpy as np

from core.interfaces.preprocessor_interface import IImagePreprocessor
from core.interfaces.detector_interface import Detection
from core.preprocessor.document_preprocessor import DocumentPreprocessor
from core.preprocessor.orientation_corrector import OrientationCorrector
from services.interfaces.preprocessing_service_interface import (
    IPreprocessingService,
    PreprocessingServiceResult
)
from services.interfaces.base_service_interface import BaseService


class S3PreprocessingService(IPreprocessingService, BaseService):
    """
    Step 3: Preprocessing Service Implementation.
    
    Crops detected labels using segmentation masks, rotates to horizontal,
    and applies AI-based orientation correction (180° fix).
    
    Creates DocumentPreprocessor internally with provided parameters.
    """
    
    SERVICE_NAME = "s3_preprocessing"
    
    def __init__(
        self,
        enabled: bool = True,
        forceLandscape: bool = True,
        aiOrientationFix: bool = True,
        aiConfidenceThreshold: float = 0.6,
        paddleModelPath: Optional[str] = None,
        debugBasePath: str = "output/debug",
        debugEnabled: bool = False
    ):
        """
        Initialize S3PreprocessingService.
        
        Args:
            enabled: Whether preprocessing is enabled.
            forceLandscape: Force landscape orientation.
            aiOrientationFix: Use AI for 180° orientation fix.
            aiConfidenceThreshold: Confidence threshold for AI orientation.
            paddleModelPath: Path to Paddle orientation model.
            debugBasePath: Base path for debug output.
            debugEnabled: Whether to save debug output.
        """
        BaseService.__init__(
            self,
            serviceName=self.SERVICE_NAME,
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        
        # Create core preprocessor implementation
        orientationCorrector = OrientationCorrector(
            aiConfidenceThreshold=aiConfidenceThreshold,
            modelPath=paddleModelPath
        )
        
        self._preprocessor: IImagePreprocessor = DocumentPreprocessor(
            orientationCorrector=orientationCorrector,
            aiConfidenceThreshold=aiConfidenceThreshold
        )
        
        self._enabled = enabled
        self._forceLandscape = forceLandscape
        self._aiOrientationFix = aiOrientationFix
        
        self._logger.info(
            f"S3PreprocessingService initialized "
            f"(forceLandscape={forceLandscape}, aiOrientationFix={aiOrientationFix})"
        )
    
    def preprocess(
        self,
        frame: np.ndarray,
        detection: Detection,
        frameId: str
    ) -> PreprocessingServiceResult:
        """Preprocess a detected label."""
        startTime = time.time()
        
        # Check if preprocessing is enabled
        if not self._enabled:
            return PreprocessingServiceResult(
                croppedImage=None,
                rotationAngle=0.0,
                orientationFixed=False,
                contourPoints=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        # Check if detection has a mask
        if detection.mask is None:
            self._logger.warning(f"[{frameId}] No mask available for preprocessing")
            return PreprocessingServiceResult(
                croppedImage=None,
                rotationAngle=0.0,
                orientationFixed=False,
                contourPoints=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        try:
            # Extract contour points from mask
            contourPoints = self._extractContourPoints(detection.mask)
            
            if contourPoints is None or len(contourPoints) < 3:
                self._logger.warning(f"[{frameId}] Invalid contour points")
                return PreprocessingServiceResult(
                    croppedImage=None,
                    rotationAngle=0.0,
                    orientationFixed=False,
                    contourPoints=None,
                    frameId=frameId,
                    success=False,
                    processingTimeMs=self._measureTime(startTime)
                )
            
            # Run preprocessing
            result = self._preprocessor.process(
                image=frame,
                maskPoints=contourPoints,
                forceLandscape=self._forceLandscape,
                useAiOrientationFix=self._aiOrientationFix
            )
            
            if result.image is None:
                self._logger.warning(f"[{frameId}] Preprocessing returned no image")
                return PreprocessingServiceResult(
                    croppedImage=None,
                    rotationAngle=0.0,
                    orientationFixed=False,
                    contourPoints=contourPoints,
                    frameId=frameId,
                    success=False,
                    processingTimeMs=self._measureTime(startTime)
                )
            
            processingTimeMs = self._measureTime(startTime)
            
            # Save debug output
            self._saveDebugOutput(
                frameId, 
                result.image, 
                0.0,  # Core doesn't return rotation angle
                False,  # Core doesn't return orientation flag
                contourPoints
            )
            
            # Log timing
            self._logTiming(frameId, processingTimeMs)
            self._logger.debug(
                f"[{frameId}] Preprocessed successfully: {result.message}"
            )
            
            return PreprocessingServiceResult(
                croppedImage=result.image,
                rotationAngle=0.0,  # Core doesn't track this separately
                orientationFixed=False,  # Core doesn't track this separately
                contourPoints=contourPoints,
                frameId=frameId,
                success=True,
                processingTimeMs=processingTimeMs
            )
            
        except Exception as e:
            self._logger.error(f"[{frameId}] Preprocessing failed: {e}")
            return PreprocessingServiceResult(
                croppedImage=None,
                rotationAngle=0.0,
                orientationFixed=False,
                contourPoints=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
    
    def setEnabled(self, enabled: bool) -> None:
        """Enable or disable preprocessing."""
        self._enabled = enabled
        self._logger.info(f"Preprocessing {'enabled' if enabled else 'disabled'}")
    
    def isEnabled(self) -> bool:
        """Check if preprocessing is enabled."""
        return self._enabled
    
    def setForceLandscape(self, enabled: bool) -> None:
        """Set force landscape option."""
        self._forceLandscape = enabled
        self._logger.debug(f"Force landscape: {enabled}")
    
    def isForceLandscape(self) -> bool:
        """Check if force landscape is enabled."""
        return self._forceLandscape
    
    def setAiOrientationFix(self, enabled: bool) -> None:
        """Set AI orientation fix option."""
        self._aiOrientationFix = enabled
        self._logger.debug(f"AI orientation fix: {enabled}")
    
    def isAiOrientationFix(self) -> bool:
        """Check if AI orientation fix is enabled."""
        return self._aiOrientationFix
    
    def isAiAvailable(self) -> bool:
        """Check if AI orientation model is available."""
        return self._preprocessor.isAiAvailable()
    
    def _extractContourPoints(self, mask: np.ndarray) -> Optional[List[List[int]]]:
        """
        Extract contour points from a segmentation mask.
        
        Args:
            mask: Binary segmentation mask.
            
        Returns:
            List of [x, y] points forming the contour.
        """
        try:
            # Ensure mask is binary uint8
            binaryMask = (mask > 0).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(
                binaryMask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return None
            
            # Get largest contour
            largestContour = max(contours, key=cv2.contourArea)
            
            # Convert to list of [x, y] points
            points = largestContour.reshape(-1, 2).tolist()
            
            return points
            
        except Exception as e:
            self._logger.warning(f"Failed to extract contour: {e}")
            return None
    
    def _saveDebugOutput(
        self,
        frameId: str,
        croppedImage: np.ndarray,
        rotationAngle: float,
        orientationFixed: bool,
        contourPoints: List[List[int]]
    ) -> None:
        """Save debug output for preprocessing step."""
        if not self._debugEnabled:
            return
        
        # Save cropped image
        self._saveDebugImage(frameId, croppedImage, "preprocessing")
        
        # Save preprocessing info as JSON
        info = {
            "frameId": frameId,
            "rotationAngle": rotationAngle,
            "orientationFixed": orientationFixed,
            "imageShape": list(croppedImage.shape),
            "contourPointsCount": len(contourPoints) if contourPoints else 0
        }
        self._saveDebugJson(frameId, info, "preprocessing")
