"""
Document Preprocessor Module

Main preprocessor implementation that orchestrates the preprocessing pipeline.
Implements IImagePreprocessor interface.

Follows:
- SRP: Orchestrates preprocessing pipeline
- OCP: New preprocessing steps can be added without modifying existing code
- DIP: Depends on abstractions (GeometricTransformer, OrientationCorrector)
"""

import logging
from typing import Optional
import numpy as np

from core.interfaces.preprocessor_interface import IImagePreprocessor, PreprocessingResult
from core.preprocessor.geometric_transformer import GeometricTransformer
from core.preprocessor.orientation_corrector import OrientationCorrector


logger = logging.getLogger(__name__)


class DocumentPreprocessor(IImagePreprocessor):
    """
    Main preprocessor for document/label images.
    
    Orchestrates the preprocessing pipeline:
    1. Geometric transformation (crop & rotate)
    2. Force landscape orientation
    3. AI-based 180-degree rotation fix
    
    Implements IImagePreprocessor interface.
    
    Follows:
    - SRP: Only orchestrates preprocessing pipeline
    - OCP: Can add new steps without modifying existing code
    - DIP: Uses injected dependencies
    """
    
    def __init__(
        self,
        geometricTransformer: Optional[GeometricTransformer] = None,
        orientationCorrector: Optional[OrientationCorrector] = None,
        aiConfidenceThreshold: float = 0.6
    ):
        """
        Initialize DocumentPreprocessor.
        
        Args:
            geometricTransformer: GeometricTransformer instance (created if None).
            orientationCorrector: OrientationCorrector instance (created if None).
            aiConfidenceThreshold: Confidence threshold for AI orientation fix.
        """
        self._geometricTransformer = geometricTransformer or GeometricTransformer()
        self._orientationCorrector = orientationCorrector or OrientationCorrector(
            aiConfidenceThreshold=aiConfidenceThreshold
        )
    
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
        if image is None:
            return PreprocessingResult(
                image=None,
                success=False,
                message="Input image is None"
            )
        
        if maskPoints is None or len(maskPoints) < 3:
            return PreprocessingResult(
                image=None,
                success=False,
                message="Insufficient mask points (need at least 3)"
            )
        
        messages = []
        
        try:
            # Step 1: Geometric transformation (crop & rotate)
            result, geoMsg = self._geometricTransformer.applyCropAndRotate(image, maskPoints)
            messages.append(f"Step1: {geoMsg}")
            
            if result is None:
                return PreprocessingResult(
                    image=None,
                    success=False,
                    message=f"Geometric transform failed: {geoMsg}"
                )
            
            # Step 2: Force landscape orientation (if enabled)
            if forceLandscape:
                result, rotated = self._orientationCorrector.forceLandscape(result)
                messages.append(f"Step2: {'Rotated 90°' if rotated else 'Already landscape'}")
            else:
                messages.append("Step2: Skipped (landscape not forced)")
            
            # Step 3: AI orientation fix (if enabled and available)
            if useAiOrientationFix and self._orientationCorrector.isAiAvailable:
                result, aiMsg = self._orientationCorrector.correctOrientationWithAi(result)
                messages.append(f"Step3: {aiMsg}")
            else:
                if not useAiOrientationFix:
                    messages.append("Step3: Skipped (AI fix disabled)")
                else:
                    messages.append("Step3: Skipped (AI not available)")
            
            return PreprocessingResult(
                image=result,
                success=True,
                message=" | ".join(messages)
            )
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            return PreprocessingResult(
                image=None,
                success=False,
                message=f"Pipeline error: {str(e)}"
            )
    
    def isAiAvailable(self) -> bool:
        """
        Check if AI orientation correction is available.
        
        Returns:
            bool: True if PaddleOCR is loaded and ready.
        """
        return self._orientationCorrector.isAiAvailable
