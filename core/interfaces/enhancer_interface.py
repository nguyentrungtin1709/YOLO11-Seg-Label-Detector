"""
Image Enhancer Interface Module

Defines the abstract interface for image enhancement operations.
Follows ISP: Focused interface for enhancement only.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class EnhancementResult:
    """
    Result of image enhancement operation.
    
    Attributes:
        image: Enhanced image (BGR format, numpy array).
        brightnessApplied: Whether brightness enhancement was applied.
        sharpnessApplied: Whether sharpness enhancement was applied.
    """
    image: np.ndarray
    brightnessApplied: bool = False
    sharpnessApplied: bool = False


class IImageEnhancer(ABC):
    """
    Abstract interface for image enhancement.
    
    Defines contract for enhancing image quality through
    brightness and sharpness adjustments.
    
    Follows DIP: High-level modules depend on this abstraction.
    """
    
    @abstractmethod
    def enhance(
        self,
        image: np.ndarray,
        applyBrightness: bool = True,
        applySharpness: bool = True
    ) -> EnhancementResult:
        """
        Enhance image quality.
        
        Args:
            image: Input image (BGR format, numpy array).
            applyBrightness: Whether to apply brightness enhancement.
            applySharpness: Whether to apply sharpness enhancement.
            
        Returns:
            EnhancementResult containing enhanced image and flags.
        """
        pass
