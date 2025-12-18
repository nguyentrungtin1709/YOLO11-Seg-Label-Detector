"""
Enhancement Service Interface Module.

Defines the interface for image enhancement operations (Step 4 of the pipeline).
Responsible for brightness and sharpness enhancement.

Follows:
- SRP: Only handles enhancement operations
- DIP: Depends on IImageEnhancer abstraction from core layer
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class EnhancementServiceResult:
    """
    Result of the enhancement service.
    
    Attributes:
        enhancedImage: Image after enhancement.
        brightnessApplied: True if brightness enhancement was applied.
        sharpnessApplied: True if sharpness enhancement was applied.
        frameId: Frame identifier for debug output.
        success: Whether enhancement was successful.
        processingTimeMs: Time taken for enhancement.
    """
    enhancedImage: Optional[np.ndarray]
    brightnessApplied: bool
    sharpnessApplied: bool
    frameId: str
    success: bool
    processingTimeMs: float = 0.0


class IEnhancementService(ABC):
    """
    Interface for image enhancement operations (Step 4).
    
    Applies brightness enhancement (CLAHE on LAB color space)
    and sharpness enhancement (Unsharp Masking).
    """
    
    @abstractmethod
    def enhance(
        self,
        image: np.ndarray,
        frameId: str
    ) -> EnhancementServiceResult:
        """
        Enhance an image.
        
        Applies brightness and sharpness enhancement based on settings.
        
        Args:
            image: Input image (BGR format).
            frameId: Frame identifier for debug output.
            
        Returns:
            EnhancementServiceResult: Enhanced image with metadata.
        """
        pass
    
    @abstractmethod
    def setEnabled(self, enabled: bool) -> None:
        """
        Enable or disable enhancement.
        
        Args:
            enabled: True to enable enhancement.
        """
        pass
    
    @abstractmethod
    def isEnabled(self) -> bool:
        """
        Check if enhancement is enabled.
        
        Returns:
            bool: True if enhancement is enabled.
        """
        pass
    
    @abstractmethod
    def setBrightnessEnabled(self, enabled: bool) -> None:
        """
        Enable or disable brightness enhancement.
        
        Args:
            enabled: True to enable brightness enhancement.
        """
        pass
    
    @abstractmethod
    def isBrightnessEnabled(self) -> bool:
        """
        Check if brightness enhancement is enabled.
        
        Returns:
            bool: True if brightness enhancement is enabled.
        """
        pass
    
    @abstractmethod
    def setSharpnessEnabled(self, enabled: bool) -> None:
        """
        Enable or disable sharpness enhancement.
        
        Args:
            enabled: True to enable sharpness enhancement.
        """
        pass
    
    @abstractmethod
    def isSharpnessEnabled(self) -> bool:
        """
        Check if sharpness enhancement is enabled.
        
        Returns:
            bool: True if sharpness enhancement is enabled.
        """
        pass
