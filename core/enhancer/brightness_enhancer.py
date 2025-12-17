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
    - Processes only L channel in LAB color space to preserve colors
    
    Advantages over direct brightness adjustment:
    - Preserves colors (only processes lightness)
    - Adaptive: processes each region separately
    - Avoids over-exposure in already bright areas
    
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
        1. Convert BGR to LAB color space
        2. Split into L (Lightness), A, B channels
        3. Apply CLAHE only to L channel
        4. Merge channels and convert back to BGR
        
        Args:
            image: Input image (BGR format, numpy array).
            
        Returns:
            Brightness-enhanced image (BGR format).
            Returns original image if processing fails.
        """
        if image is None or image.size == 0:
            logger.warning("Input image is None or empty")
            return image
        
        try:
            # 1. Convert BGR to LAB color space
            # LAB: L = Lightness, A = Green-Red, B = Blue-Yellow
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # 2. Split channels: L, A, B
            lChannel, aChannel, bChannel = cv2.split(lab)
            
            # 3. Create CLAHE object and apply to L channel only
            clahe = cv2.createCLAHE(
                clipLimit=self._clipLimit,
                tileGridSize=self._tileGridSize
            )
            lChannelEnhanced = clahe.apply(lChannel)
            
            # 4. Merge channels back
            labEnhanced = cv2.merge([lChannelEnhanced, aChannel, bChannel])
            
            # 5. Convert back to BGR
            result = cv2.cvtColor(labEnhanced, cv2.COLOR_LAB2BGR)
            
            logger.debug("Brightness enhancement applied successfully")
            return result
            
        except Exception as e:
            logger.error(f"Brightness enhancement failed: {e}")
            return image.copy() if image is not None else None
