"""
QR Image Preprocessor Module.

This module provides image preprocessing specifically for QR code detection.
Applies various image processing techniques to improve QR code detection rate.

Supports two modes:
- "minimal": Scale only (fast)
- "full": Scale → Denoise → Binary → Morph → Invert (thorough)

Note: This preprocessor expects GRAYSCALE input from S4 Enhancement Service.
Enhancement (CLAHE + Sharpen) is already applied in S4, so not included here.

Follows the Single Responsibility Principle (SRP) from SOLID.
"""

import logging
from typing import Optional

import cv2
import numpy as np


class QrImagePreprocessor:
    """
    Image preprocessor for QR code detection.
    
    Provides sequential preprocessing pipeline to improve QR detection
    on difficult images. Expects grayscale input from S4 Enhancement.
    
    Modes:
    - "minimal": Scale only (for good quality images)
    - "full": Scale → Denoise → Binary → Morph → Invert (for difficult images)
    
    Pipeline order rationale:
    1. Scale - Resize first for consistent processing
    2. Denoise - Remove noise BEFORE thresholding
    3. Binary - Convert to black/white
    4. Morph - Fill holes in binary image
    5. Invert - Flip colors if QR is white-on-black
    """
    
    # Supported preprocessing modes
    MODE_MINIMAL = "minimal"
    MODE_FULL = "full"
    SUPPORTED_MODES = [MODE_MINIMAL, MODE_FULL]
    
    def __init__(
        self,
        enabled: bool = True,
        mode: str = "full",
        scaleFactor: float = 1.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize QrImagePreprocessor.
        
        Args:
            enabled: Master switch to enable/disable preprocessing.
            mode: Preprocessing mode ("minimal" or "full").
            scaleFactor: Scale factor for resizing (default: 1.5).
            logger: Logger instance for debug output.
        """
        self._enabled = enabled
        self._mode = mode if mode in self.SUPPORTED_MODES else self.MODE_FULL
        self._scaleFactor = scaleFactor
        self._logger = logger or logging.getLogger(__name__)
        
        self._logger.info(
            f"QrImagePreprocessor initialized "
            f"(enabled={enabled}, mode={self._mode}, scaleFactor={scaleFactor}x)"
        )
    
    @property
    def mode(self) -> str:
        """Get current preprocessing mode."""
        return self._mode
    
    @property
    def scaleFactor(self) -> float:
        """Get current scale factor."""
        return self._scaleFactor
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for QR detection.
        
        Applies sequential pipeline based on mode:
        - Minimal: Scale only
        - Full: Scale → Denoise → Binary → Morph → Invert
        
        Args:
            image: Input image (Grayscale from S4).
            
        Returns:
            Single preprocessed image.
        """
        if not self._enabled:
            return image
        
        if image is None or image.size == 0:
            self._logger.warning("Input image is None or empty")
            return image
        
        if self._mode == self.MODE_MINIMAL:
            return self._applyMinimalPipeline(image)
        else:  # MODE_FULL
            return self._applyFullPipeline(image)
    
    def _applyMinimalPipeline(self, image: np.ndarray) -> np.ndarray:
        """
        Apply minimal preprocessing pipeline.
        
        Pipeline: Scale only
        
        Args:
            image: Input grayscale image.
            
        Returns:
            Scaled image.
        """
        self._logger.debug("Applying minimal pipeline (scale only)")
        return self._applyScale(image)
    
    def _applyFullPipeline(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Pipeline: Scale → Denoise → Binary → Morph → Invert
        
        Args:
            image: Input grayscale image.
            
        Returns:
            Fully preprocessed image.
        """
        self._logger.debug("Applying full pipeline (scale → denoise)")
        
        result = image
        
        # Step 1: Scale
        result = self._applyScale(result)
        
        # Step 2: Denoise (before binary to reduce noise)
        result = self._applyDenoise(result)
        
        # Step 3: Binary (adaptive threshold)
        # NOTE: Disabled - may destroy QR code structure
        # result = self._applyBinaryProcessing(result)
        
        # Step 4: Morphology (fill holes)
        # NOTE: Disabled - may destroy QR code structure
        # result = self._applyMorphology(result)
        
        # Step 5: Invert (flip colors for white-on-black QR)
        # NOTE: Disabled - standard QR is black-on-white
        # result = self._applyInvert(result)
        
        return result
    
    def _applyScale(self, image: np.ndarray) -> np.ndarray:
        """
        Scale image by configured factor.
        
        Args:
            image: Input image (grayscale or BGR).
            
        Returns:
            Scaled image.
        """
        if self._scaleFactor == 1.0:
            return image
        
        try:
            h, w = image.shape[:2]
            newW = int(w * self._scaleFactor)
            newH = int(h * self._scaleFactor)
            
            # Use appropriate interpolation method
            if self._scaleFactor > 1.0:
                interpolation = cv2.INTER_CUBIC  # Better for enlarging
            else:
                interpolation = cv2.INTER_AREA   # Better for shrinking
            
            scaled = cv2.resize(image, (newW, newH), interpolation=interpolation)
            self._logger.debug(f"Scale: {w}x{h} → {newW}x{newH} ({self._scaleFactor}x)")
            return scaled
            
        except Exception as e:
            self._logger.error(f"Scale failed: {e}")
            return image
    
    def _applyDenoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply median blur for denoising.
        
        Args:
            image: Input image (grayscale).
            
        Returns:
            Denoised image.
        """
        try:
            # Ensure grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            result = cv2.medianBlur(gray, 3)
            self._logger.debug("Denoise: median blur (kernel=3)")
            return result
            
        except Exception as e:
            self._logger.error(f"Denoise failed: {e}")
            return image
    
    def _applyBinaryProcessing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding to create binary image.
        
        Args:
            image: Input image (grayscale).
            
        Returns:
            Binary image.
        """
        try:
            # Ensure grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply Gaussian blur to reduce noise (light blur)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                19,  # Block size
                2    # C constant
            )
            
            self._logger.debug("Binary: adaptive threshold (block=19, C=2)")
            return thresh
            
        except Exception as e:
            self._logger.error(f"Binary failed: {e}")
            return image
    
    def _applyMorphology(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological closing operation.
        
        Args:
            image: Input image (grayscale/binary).
            
        Returns:
            Morphologically processed image.
        """
        try:
            # Ensure grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Create structuring element
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # Apply morphological closing (dilation followed by erosion)
            result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            self._logger.debug("Morph: closing (kernel=3x3)")
            return result
            
        except Exception as e:
            self._logger.error(f"Morphology failed: {e}")
            return image
    
    def _applyInvert(self, image: np.ndarray) -> np.ndarray:
        """
        Invert image colors (bitwise NOT).
        
        Args:
            image: Input image.
            
        Returns:
            Inverted image.
        """
        try:
            result = cv2.bitwise_not(image)
            self._logger.debug("Invert: bitwise NOT")
            return result
            
        except Exception as e:
            self._logger.error(f"Invert failed: {e}")
            return image
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Configuration Methods
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def setEnabled(self, enabled: bool) -> None:
        """Enable or disable preprocessing."""
        self._enabled = enabled
        self._logger.info(f"Preprocessing {'enabled' if enabled else 'disabled'}")
    
    def isEnabled(self) -> bool:
        """Check if preprocessing is enabled."""
        return self._enabled
    
    def setMode(self, mode: str) -> None:
        """
        Set preprocessing mode.
        
        Args:
            mode: "minimal" or "full"
        """
        if mode in self.SUPPORTED_MODES:
            self._mode = mode
            self._logger.info(f"Preprocessing mode set to: {mode}")
        else:
            self._logger.warning(
                f"Invalid mode '{mode}', keeping current mode: {self._mode}"
            )
    
    def setScaleFactor(self, factor: float) -> None:
        """Set scale factor."""
        self._scaleFactor = factor
        self._logger.debug(f"Scale factor set to: {factor}")
