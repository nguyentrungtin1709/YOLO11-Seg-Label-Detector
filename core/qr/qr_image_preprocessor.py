"""
QR Image Preprocessor Module.

This module provides image preprocessing specifically for QR code detection.
Applies various image processing techniques to improve QR code detection rate.

Supports two modes:
- "minimal": Scale only (fast)
- "full": Scale → Denoise (thorough)

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
    - "full": Scale → Denoise (for difficult images)
    
    Pipeline order rationale:
    1. Scale - Resize to target width for consistent processing
    2. Denoise - Remove noise for cleaner QR detection
    """
    
    # Supported preprocessing modes
    MODE_MINIMAL = "minimal"
    MODE_FULL = "full"
    SUPPORTED_MODES = [MODE_MINIMAL, MODE_FULL]
    
    # Default target width for scaling
    DEFAULT_TARGET_WIDTH = 480
    
    def __init__(
        self,
        enabled: bool = True,
        mode: str = "full",
        targetWidth: int = 480,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize QrImagePreprocessor.
        
        Args:
            enabled: Master switch to enable/disable preprocessing.
            mode: Preprocessing mode ("minimal" or "full").
            targetWidth: Target width for scaling (default: 480).
                        Image will be resized to this width, maintaining aspect ratio.
                        Scale factor is computed dynamically based on input image width.
            logger: Logger instance for debug output.
        """
        self._enabled = enabled
        self._mode = mode if mode in self.SUPPORTED_MODES else self.MODE_FULL
        self._targetWidth = targetWidth
        self._scaleFactor = 1.0  # Will be computed dynamically in _applyScale()
        self._logger = logger or logging.getLogger(__name__)
        
        self._logger.info(
            f"QrImagePreprocessor initialized "
            f"(enabled={enabled}, mode={self._mode}, targetWidth={targetWidth}px)"
        )
    
    @property
    def mode(self) -> str:
        """Get current preprocessing mode."""
        return self._mode
    
    @property
    def scaleFactor(self) -> float:
        """
        Get computed scale factor.
        
        This value is computed dynamically during preprocess() based on
        the input image width and targetWidth. Use this value to scale
        back QR coordinates to original image size.
        
        Returns:
            float: Scale factor (targetWidth / originalWidth)
        """
        return self._scaleFactor
    
    @property
    def targetWidth(self) -> int:
        """Get target width for scaling."""
        return self._targetWidth
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for QR detection.
        
        Applies sequential pipeline based on mode:
        - Minimal: Scale only
        - Full: Scale → Denoise
        
        After calling this method, use the scaleFactor property to get
        the computed scale factor for scaling back QR coordinates.
        
        Args:
            image: Input image (Grayscale from S4).
            
        Returns:
            Single preprocessed image.
        """
        if not self._enabled:
            self._scaleFactor = 1.0
            return image
        
        if image is None or image.size == 0:
            self._logger.warning("Input image is None or empty")
            self._scaleFactor = 1.0
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
        
        Pipeline: Scale → Denoise
        
        Args:
            image: Input grayscale image.
            
        Returns:
            Fully preprocessed image.
        """
        self._logger.debug("Applying full pipeline (scale → denoise)")
        
        result = image
        
        # Step 1: Scale to target width
        result = self._applyScale(result)
        
        # Step 2: Denoise (reduce noise for cleaner QR detection)
        result = self._applyDenoise(result)
        
        return result
    
    def _applyScale(self, image: np.ndarray) -> np.ndarray:
        """
        Scale image to target width, maintaining aspect ratio.
        
        Computes scale factor dynamically based on input image width
        and stores it in self._scaleFactor for later use (e.g., scaling
        back QR coordinates to original image size).
        
        Args:
            image: Input image (grayscale or BGR).
            
        Returns:
            Scaled image.
        """
        try:
            h, w = image.shape[:2]
            
            # Compute scale factor based on target width
            self._scaleFactor = self._targetWidth / w
            
            # If scale factor is very close to 1.0, skip scaling
            if abs(self._scaleFactor - 1.0) < 0.01:
                self._scaleFactor = 1.0
                self._logger.debug(f"Scale: skipped (width {w} ≈ target {self._targetWidth})")
                return image
            
            newW = self._targetWidth
            newH = int(h * self._scaleFactor)
            
            # Use appropriate interpolation method
            if self._scaleFactor > 1.0:
                interpolation = cv2.INTER_CUBIC  # Better for enlarging
            else:
                interpolation = cv2.INTER_AREA   # Better for shrinking
            
            scaled = cv2.resize(image, (newW, newH), interpolation=interpolation)
            self._logger.debug(f"Scale: {w}x{h} → {newW}x{newH} ({self._scaleFactor:.3f}x)")
            return scaled
            
        except Exception as e:
            self._logger.error(f"Scale failed: {e}")
            self._scaleFactor = 1.0
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
    
    def setTargetWidth(self, width: int) -> None:
        """
        Set target width for scaling.
        
        Args:
            width: Target width in pixels.
        """
        self._targetWidth = width
        self._logger.debug(f"Target width set to: {width}px")
