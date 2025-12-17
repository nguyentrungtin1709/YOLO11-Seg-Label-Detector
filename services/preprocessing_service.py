"""
Preprocessing Service

Orchestrates image preprocessing for detected objects.
Processes detections and returns preprocessed images.

Follows:
- SRP: Only handles preprocessing orchestration
- DIP: Depends on IImagePreprocessor and IImageEnhancer abstractions
"""

import logging
from typing import Optional
from dataclasses import dataclass
import numpy as np
import cv2

from core.interfaces.preprocessor_interface import IImagePreprocessor, PreprocessingResult
from core.interfaces.enhancer_interface import IImageEnhancer
from core.interfaces.detector_interface import Detection


logger = logging.getLogger(__name__)


@dataclass
class FullPreprocessingResult:
    """
    Result of full preprocessing pipeline (crop + enhance).
    
    Attributes:
        croppedImage: Image after crop, rotate, orientation fix (before enhancement).
        enhancedImage: Image after brightness + sharpness enhancement (final result).
        success: Whether preprocessing was successful.
        message: Status message or error description.
    """
    croppedImage: Optional[np.ndarray] = None
    enhancedImage: Optional[np.ndarray] = None
    success: bool = False
    message: str = ""


class PreprocessingService:
    """
    Service for managing image preprocessing operations.
    
    Provides high-level preprocessing control including:
    - Processing detected objects (crop, rotate, orientation fix)
    - Image enhancement (brightness, sharpness)
    - Extracting contour points from detection masks
    - Managing preprocessing state
    
    Follows:
    - SRP: Only handles preprocessing orchestration
    - DIP: Depends on IImagePreprocessor and IImageEnhancer abstractions
    """
    
    def __init__(
        self,
        preprocessor: IImagePreprocessor,
        enhancer: Optional[IImageEnhancer] = None,
        enabled: bool = True,
        forceLandscape: bool = True,
        aiOrientationFix: bool = True,
        brightnessEnabled: bool = True,
        sharpnessEnabled: bool = True
    ):
        """
        Initialize PreprocessingService.
        
        Args:
            preprocessor: Image preprocessor implementation.
            enhancer: Image enhancer implementation (optional).
            enabled: Enable/disable preprocessing.
            forceLandscape: Force landscape orientation on output.
            aiOrientationFix: Use AI to fix 180-degree rotation.
            brightnessEnabled: Enable brightness enhancement.
            sharpnessEnabled: Enable sharpness enhancement.
        """
        self._preprocessor = preprocessor
        self._enhancer = enhancer
        self._enabled = enabled
        self._forceLandscape = forceLandscape
        self._aiOrientationFix = aiOrientationFix
        self._brightnessEnabled = brightnessEnabled
        self._sharpnessEnabled = sharpnessEnabled
    
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
    
    def setBrightnessEnabled(self, enabled: bool) -> None:
        """Set brightness enhancement option."""
        self._brightnessEnabled = enabled
        logger.debug(f"Brightness enhancement: {enabled}")
    
    def setSharpnessEnabled(self, enabled: bool) -> None:
        """Set sharpness enhancement option."""
        self._sharpnessEnabled = enabled
        logger.debug(f"Sharpness enhancement: {enabled}")
    
    @property
    def isBrightnessEnabled(self) -> bool:
        """Check if brightness enhancement is enabled."""
        return self._brightnessEnabled
    
    @property
    def isSharpnessEnabled(self) -> bool:
        """Check if sharpness enhancement is enabled."""
        return self._sharpnessEnabled
    
    @property
    def hasEnhancer(self) -> bool:
        """Check if enhancer is available."""
        return self._enhancer is not None

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
    
    def processFirstDetectionFull(
        self,
        frame: np.ndarray,
        detections: list[Detection]
    ) -> Optional[FullPreprocessingResult]:
        """
        Process first detection with full pipeline (crop + enhance).
        
        Returns both cropped image (for debug save) and enhanced image (for display).
        
        Pipeline:
        1. Crop, rotate, orientation fix (DocumentPreprocessor)
        2. Brightness enhancement (if enabled)
        3. Sharpness enhancement (if enabled)
        
        Args:
            frame: Original frame (BGR format).
            detections: List of Detection objects (sorted by confidence).
            
        Returns:
            FullPreprocessingResult with both cropped and enhanced images,
            or None if preprocessing is disabled or no valid detection.
        """
        if not self._enabled or not detections:
            return None
        
        # Step 1: Get cropped image from DocumentPreprocessor
        cropResult = self.processFirstDetection(frame, detections)
        
        if cropResult is None or not cropResult.success or cropResult.image is None:
            return FullPreprocessingResult(
                croppedImage=None,
                enhancedImage=None,
                success=False,
                message=cropResult.message if cropResult else "No detection to process"
            )
        
        croppedImage = cropResult.image
        enhancedImage = croppedImage.copy()
        
        # Step 2: Apply enhancement if enhancer is available
        if self._enhancer is not None:
            applyBrightness = self._brightnessEnabled
            applySharpness = self._sharpnessEnabled
            
            if applyBrightness or applySharpness:
                enhanceResult = self._enhancer.enhance(
                    image=croppedImage,
                    applyBrightness=applyBrightness,
                    applySharpness=applySharpness
                )
                enhancedImage = enhanceResult.image
                logger.debug(
                    f"Enhancement applied: brightness={enhanceResult.brightnessApplied}, "
                    f"sharpness={enhanceResult.sharpnessApplied}"
                )
        
        return FullPreprocessingResult(
            croppedImage=croppedImage,
            enhancedImage=enhancedImage,
            success=True,
            message="Full preprocessing completed"
        )
    
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
