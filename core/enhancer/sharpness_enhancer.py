"""
Sharpness Enhancer Module

Enhances image sharpness using Unsharp Mask technique.
Particularly effective for blurry images in factory/industrial conditions.

Follows SRP: Only handles sharpness enhancement operations.
"""

import logging
from typing import Optional
import numpy as np
import cv2


logger = logging.getLogger(__name__)


class SharpnessEnhancer:
    """
    Enhances image sharpness using Unsharp Mask algorithm.
    
    Unsharp Mask Formula:
        Sharpened = Original + amount × (Original - Blurred)
    
    Process:
    1. Apply Gaussian blur to original image (low-pass filter)
    2. Subtract blurred from original = high frequency details (edges)
    3. Add high frequency details to original with amount coefficient
    
    Advantages:
    - Enhances edges and fine details
    - Adjustable strength via amount parameter
    - Preserves overall image structure
    
    Follows SRP: Only responsible for sharpness enhancement.
    """
    
    def __init__(
        self,
        sigma: float = 1.0,
        amount: float = 1.5
    ):
        """
        Initialize SharpnessEnhancer.
        
        Args:
            sigma: Sigma for Gaussian blur (0.5 - 3.0).
                   Higher values = more blur = stronger edge detection.
            amount: Sharpen coefficient (1.0 - 3.0).
                    1.0 = light sharpening, 2.0+ = strong sharpening.
        """
        self._sigma = sigma
        self._amount = amount
        
        logger.info(
            f"SharpnessEnhancer initialized: sigma={sigma}, amount={amount}"
        )
    
    @property
    def sigma(self) -> float:
        """Get current sigma value."""
        return self._sigma
    
    @property
    def amount(self) -> float:
        """Get current amount value."""
        return self._amount
    
    def enhanceSharpness(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image sharpness using Unsharp Mask.
        
        Formula: Sharpened = Original + amount × (Original - Blurred)
        Using cv2.addWeighted:
            dst = src1 × alpha + src2 × beta + gamma
            Sharpened = Original × (1 + amount) + Blurred × (-amount) + 0
        
        Args:
            image: Input image (BGR format, numpy array).
            
        Returns:
            Sharpness-enhanced image (BGR format).
            Returns original image if processing fails.
        """
        if image is None or image.size == 0:
            logger.warning("Input image is None or empty")
            return image
        
        try:
            # 1. Apply Gaussian blur
            # ksize=(0,0) means kernel size is computed from sigma
            blurred = cv2.GaussianBlur(image, (0, 0), self._sigma)
            
            # 2. Apply Unsharp Mask formula using addWeighted
            # Sharpened = Original × (1 + amount) + Blurred × (-amount)
            sharpened = cv2.addWeighted(
                image, 1.0 + self._amount,   # src1 and alpha
                blurred, -self._amount,       # src2 and beta
                0                             # gamma
            )
            
            logger.debug("Sharpness enhancement applied successfully")
            return sharpened
            
        except Exception as e:
            logger.error(f"Sharpness enhancement failed: {e}")
            return image.copy() if image is not None else None
