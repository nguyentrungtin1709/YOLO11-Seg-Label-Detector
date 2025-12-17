"""
Preprocessing Service

Orchestrates image preprocessing for detected objects.
Processes detections and returns preprocessed images.

Follows:
- SRP: Only handles preprocessing orchestration
- DIP: Depends on IImagePreprocessor abstraction
"""

import logging
from typing import Optional
import numpy as np
import cv2

from core.interfaces.preprocessor_interface import IImagePreprocessor, PreprocessingResult
from core.interfaces.detector_interface import Detection


logger = logging.getLogger(__name__)


class PreprocessingService:
    """
    Service for managing image preprocessing operations.
    
    Provides high-level preprocessing control including:
    - Processing detected objects (crop, rotate, orientation fix)
    - Extracting contour points from detection masks
    - Managing preprocessing state
    
    Follows:
    - SRP: Only handles preprocessing orchestration
    - DIP: Depends on IImagePreprocessor abstraction
    """
    
    def __init__(
        self,
        preprocessor: IImagePreprocessor,
        enabled: bool = True,
        forceLandscape: bool = True,
        aiOrientationFix: bool = True
    ):
        """
        Initialize PreprocessingService.
        
        Args:
            preprocessor: Image preprocessor implementation.
            enabled: Enable/disable preprocessing.
            forceLandscape: Force landscape orientation on output.
            aiOrientationFix: Use AI to fix 180-degree rotation.
        """
        self._preprocessor = preprocessor
        self._enabled = enabled
        self._forceLandscape = forceLandscape
        self._aiOrientationFix = aiOrientationFix
    
    @property
    def isEnabled(self) -> bool:
        """Check if preprocessing is enabled."""
        return self._enabled
    
    @isEnabled.setter
    def isEnabled(self, value: bool) -> None:
        """Enable or disable preprocessing."""
        self._enabled = value
        logger.info(f"Preprocessing {'enabled' if value else 'disabled'}")
    
    @property
    def isAiAvailable(self) -> bool:
        """Check if AI orientation correction is available."""
        return self._preprocessor.isAiAvailable()
    
    def setForceLandscape(self, enabled: bool) -> None:
        """Set force landscape option."""
        self._forceLandscape = enabled
        logger.debug(f"Force landscape: {enabled}")
    
    def setAiOrientationFix(self, enabled: bool) -> None:
        """Set AI orientation fix option."""
        self._aiOrientationFix = enabled
        logger.debug(f"AI orientation fix: {enabled}")
    
    def processDetections(
        self,
        frame: np.ndarray,
        detections: list[Detection]
    ) -> list[PreprocessingResult]:
        """
        Process all detections and return preprocessed images.
        
        For each detection with a mask, extracts contour points and
        runs the preprocessing pipeline (crop, rotate, orientation fix).
        
        Args:
            frame: Original frame (BGR format).
            detections: List of Detection objects (already filtered).
            
        Returns:
            List of PreprocessingResult objects (same order as detections).
            Results with None image indicate preprocessing failed for that detection.
        """
        if not self._enabled:
            return []
        
        if not detections:
            return []
        
        results = []
        
        for idx, det in enumerate(detections):
            if det.mask is None:
                logger.debug(f"Detection {idx}: No mask available")
                results.append(PreprocessingResult(
                    image=None,
                    success=False,
                    message="No mask available"
                ))
                continue
            
            # Extract contour points from mask
            points = self._extractContourPoints(det.mask)
            
            if points is None or len(points) < 3:
                logger.debug(f"Detection {idx}: Insufficient contour points")
                results.append(PreprocessingResult(
                    image=None,
                    success=False,
                    message="Insufficient contour points"
                ))
                continue
            
            # Run preprocessing pipeline
            result = self._preprocessor.process(
                image=frame,
                maskPoints=points,
                forceLandscape=self._forceLandscape,
                useAiOrientationFix=self._aiOrientationFix
            )
            
            if result.success:
                logger.debug(f"Detection {idx}: Preprocessing successful")
            else:
                logger.debug(f"Detection {idx}: Preprocessing failed - {result.message}")
            
            results.append(result)
        
        return results
    
    def processFirstDetection(
        self,
        frame: np.ndarray,
        detections: list[Detection]
    ) -> Optional[PreprocessingResult]:
        """
        Process only the first detection (highest confidence).
        
        Convenience method for common use case where only top detection matters.
        
        Args:
            frame: Original frame (BGR format).
            detections: List of Detection objects (sorted by confidence).
            
        Returns:
            PreprocessingResult for first detection, or None if no valid detection.
        """
        if not self._enabled or not detections:
            return None
        
        # Process first detection only
        results = self.processDetections(frame, detections[:1])
        
        return results[0] if results else None
    
    def _extractContourPoints(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract contour points from a binary mask.
        
        Args:
            mask: Binary mask (H x W) with values 0 or 255/1.
            
        Returns:
            Numpy array of contour points (N x 2), or None if extraction fails.
        """
        try:
            # Ensure mask is binary (0 or 255)
            if mask.max() <= 1:
                binaryMask = (mask * 255).astype(np.uint8)
            else:
                binaryMask = mask.astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(
                binaryMask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return None
            
            # Get the largest contour
            largestContour = max(contours, key=cv2.contourArea)
            
            # Reshape to (N, 2) format
            points = largestContour.reshape(-1, 2)
            
            return points
            
        except Exception as e:
            logger.error(f"Error extracting contour points: {e}")
            return None
