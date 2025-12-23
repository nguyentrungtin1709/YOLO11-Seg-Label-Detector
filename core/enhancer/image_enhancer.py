"""
Image Enhancer Module

Orchestrates brightness and sharpness enhancement operations.
Implements IImageEnhancer interface.

Follows OCP: Open for extension (new enhancers), closed for modification.
Follows DIP: Depends on injected enhancer components.
"""

import logging
from typing import Optional
import numpy as np
import cv2

from core.interfaces.enhancer_interface import IImageEnhancer, EnhancementResult
from core.enhancer.brightness_enhancer import BrightnessEnhancer
from core.enhancer.sharpness_enhancer import SharpnessEnhancer


logger = logging.getLogger(__name__)


class ImageEnhancer(IImageEnhancer):
    """
    Orchestrates image enhancement pipeline.
    
    Combines BrightnessEnhancer and SharpnessEnhancer to provide
    complete image quality enhancement on grayscale images.
    
    Pipeline order (important):
    1. Convert BGR to Grayscale FIRST
    2. Brightness enhancement SECOND - reveals hidden details
    3. Sharpness enhancement THIRD - sharpens revealed details
    
    Follows DIP: Receives enhancers via dependency injection.
    Follows SRP: Only orchestrates, doesn't implement enhancement logic.
    """
    
    def __init__(
        self,
        brightnessEnhancer: BrightnessEnhancer,
        sharpnessEnhancer: SharpnessEnhancer
    ):
        """
        Initialize ImageEnhancer with component enhancers.
        
        Args:
            brightnessEnhancer: Component for brightness enhancement.
            sharpnessEnhancer: Component for sharpness enhancement.
        """
        self._brightnessEnhancer = brightnessEnhancer
        self._sharpnessEnhancer = sharpnessEnhancer
        
        logger.info("ImageEnhancer initialized with brightness and sharpness enhancers")
    
    @property
    def brightnessEnhancer(self) -> BrightnessEnhancer:
        """Get brightness enhancer component."""
        return self._brightnessEnhancer
    
    @property
    def sharpnessEnhancer(self) -> SharpnessEnhancer:
        """Get sharpness enhancer component."""
        return self._sharpnessEnhancer
    
    def enhance(
        self,
        image: np.ndarray,
        applyBrightness: bool = True,
        applySharpness: bool = True
    ) -> EnhancementResult:
        """
        Enhance image quality through brightness and sharpness adjustments on grayscale.
        
        Pipeline:
        1. Convert BGR to Grayscale
        2. If applyBrightness: Apply CLAHE brightness enhancement
        3. If applySharpness: Apply Unsharp Mask sharpness enhancement
        
        Args:
            image: Input image (BGR format, numpy array).
            applyBrightness: Whether to apply brightness enhancement.
            applySharpness: Whether to apply sharpness enhancement.
            
        Returns:
            EnhancementResult containing:
            - image: Enhanced grayscale image (H, W)
            - brightnessApplied: Whether brightness was applied
            - sharpnessApplied: Whether sharpness was applied
        """
        if image is None or image.size == 0:
            logger.warning("Input image is None or empty")
            return EnhancementResult(
                image=image,
                brightnessApplied=False,
                sharpnessApplied=False
            )
        
        # Convert BGR to Grayscale at the beginning
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        result = gray.copy()
        brightnessApplied = False
        sharpnessApplied = False
        
        # Step 1: Brightness enhancement (CLAHE on grayscale)
        if applyBrightness:
            result = self._brightnessEnhancer.enhanceBrightness(result)
            brightnessApplied = True
            logger.debug("Brightness enhancement step completed")
        
        # Step 2: Sharpness enhancement (Unsharp Mask on grayscale)
        if applySharpness:
            result = self._sharpnessEnhancer.enhanceSharpness(result)
            sharpnessApplied = True
            logger.debug("Sharpness enhancement step completed")
        
        if not applyBrightness and not applySharpness:
            logger.debug("No enhancement applied, returning grayscale image")
        
        return EnhancementResult(
            image=result,
            brightnessApplied=brightnessApplied,
            sharpnessApplied=sharpnessApplied
        )
