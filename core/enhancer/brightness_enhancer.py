"""
Brightness Enhancer Module

Enhances image brightness using CLAHE (Contrast Limited Adaptive Histogram Equalization).
Particularly effective for dark images in factory/industrial conditions.

Follows SRP: Only handles brightness enhancement operations.
"""

import logging
from typing import Tuple, Optional
import numpy as np
import cv2


logger = logging.getLogger(__name__)


class BrightnessEnhancer:
    """
    Enhances image brightness using CLAHE algorithm.
    
    CLAHE (Contrast Limited Adaptive Histogram Equalization):
    - Divides image into tiles and equalizes histogram for each tile separately
    - Uses clip limit to avoid noise amplification
    - Applied directly on grayscale images
    
    Advantages over direct brightness adjustment:
    - Adaptive: processes each region separately
    - Avoids over-exposure in already bright areas
    - Efficient on single-channel grayscale images
    
    Follows SRP: Only responsible for brightness enhancement.
    """
    
    def __init__(
        self,
        clipLimit: float = 2.5,
        tileGridSize: Tuple[int, int] = (8, 8)
    ):
        """
        Initialize BrightnessEnhancer.
        
        Args:
            clipLimit: Contrast limit to avoid noise amplification (1.0 - 5.0).
                       Higher values = more contrast but more noise.
            tileGridSize: Grid size for dividing image into tiles.
                          Smaller tiles = more localized enhancement.
        """
        self._clipLimit = clipLimit
        self._tileGridSize = tileGridSize
        
        logger.info(
            f"BrightnessEnhancer initialized: clipLimit={clipLimit}, "
            f"tileGridSize={tileGridSize}"
        )
    
    @property
    def clipLimit(self) -> float:
        """Get current clip limit."""
        return self._clipLimit
    
    @property
    def tileGridSize(self) -> Tuple[int, int]:
        """Get current tile grid size."""
        return self._tileGridSize
    
    def enhanceBrightness(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image brightness using CLAHE.
        
        Process:
        1. Ensure input is 2D grayscale array
        2. Apply CLAHE directly on grayscale image
        
        Args:
            image: Input grayscale image (H, W) or (H, W, 1).
            
        Returns:
            Brightness-enhanced grayscale image (H, W).
            Returns original image if processing fails.
        """
        if image is None or image.size == 0:
            logger.warning("Input image is None or empty")
            return image
        
        try:
            # Ensure input is 2D array
            if len(image.shape) == 3:
                if image.shape[2] == 1:
                    image = image[:, :, 0]
                else:
                    logger.warning("Expected grayscale image, got multi-channel image")
                    return image
            
            # Create CLAHE object and apply directly on grayscale
            clahe = cv2.createCLAHE(
                clipLimit=self._clipLimit,
                tileGridSize=self._tileGridSize
            )
            enhanced = clahe.apply(image)
            
            logger.debug("Brightness enhancement applied successfully")
            return enhanced
            
        except Exception as e:
            logger.error(f"Brightness enhancement failed: {e}")
            return image.copy() if image is not None else None
