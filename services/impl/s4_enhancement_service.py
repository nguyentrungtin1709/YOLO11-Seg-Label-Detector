"""
S4 Enhancement Service Implementation.

Step 4 of the pipeline: Brightness and sharpness enhancement.
Creates and manages ImageEnhancer from core layer.

Follows:
- SRP: Only handles enhancement operations
- DIP: Depends on IImageEnhancer abstraction (interface)
- OCP: Extends without modifying existing code
"""

import time
import logging
from typing import Optional, Tuple

import numpy as np

from core.interfaces.enhancer_interface import IImageEnhancer
from core.enhancer.image_enhancer import ImageEnhancer
from core.enhancer.brightness_enhancer import BrightnessEnhancer
from core.enhancer.sharpness_enhancer import SharpnessEnhancer
from services.interfaces.enhancement_service_interface import (
    IEnhancementService,
    EnhancementServiceResult
)
from services.interfaces.base_service_interface import BaseService


class S4EnhancementService(IEnhancementService, BaseService):
    """
    Step 4: Enhancement Service Implementation.
    
    Applies brightness enhancement (CLAHE on LAB color space)
    and sharpness enhancement (Unsharp Masking).
    
    Creates ImageEnhancer internally with provided parameters.
    """
    
    SERVICE_NAME = "s4_enhancement"
    
    def __init__(
        self,
        enabled: bool = True,
        brightnessEnabled: bool = True,
        brightnessClipLimit: float = 2.5,
        brightnessTileSize: int = 8,
        sharpnessEnabled: bool = True,
        sharpnessSigma: float = 1.0,
        sharpnessAmount: float = 1.5,
        debugBasePath: str = "output/debug",
        debugEnabled: bool = False
    ):
        """
        Initialize S4EnhancementService.
        
        Args:
            enabled: Whether enhancement is enabled overall.
            brightnessEnabled: Whether brightness enhancement is enabled.
            brightnessClipLimit: CLAHE clip limit for brightness.
            brightnessTileSize: CLAHE tile size for brightness.
            sharpnessEnabled: Whether sharpness enhancement is enabled.
            sharpnessSigma: Sigma for sharpness Gaussian blur.
            sharpnessAmount: Sharpen coefficient.
            debugBasePath: Base path for debug output.
            debugEnabled: Whether to save debug output.
        """
        BaseService.__init__(
            self,
            serviceName=self.SERVICE_NAME,
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        
        # Create core enhancer implementation
        brightnessEnhancer = BrightnessEnhancer(
            clipLimit=brightnessClipLimit,
            tileGridSize=(brightnessTileSize, brightnessTileSize)
        )
        sharpnessEnhancer = SharpnessEnhancer(
            sigma=sharpnessSigma,
            amount=sharpnessAmount
        )
        
        self._enhancer: IImageEnhancer = ImageEnhancer(
            brightnessEnhancer=brightnessEnhancer,
            sharpnessEnhancer=sharpnessEnhancer
        )
        
        self._enabled = enabled
        self._brightnessEnabled = brightnessEnabled
        self._sharpnessEnabled = sharpnessEnabled
        
        self._logger.info(
            f"S4EnhancementService initialized "
            f"(brightness={brightnessEnabled}, sharpness={sharpnessEnabled})"
        )
    
    def enhance(
        self,
        image: np.ndarray,
        frameId: str
    ) -> EnhancementServiceResult:
        """Enhance an image with brightness and sharpness."""
        startTime = time.time()
        
        # Check if enhancement is enabled
        if not self._enabled:
            return EnhancementServiceResult(
                enhancedImage=image.copy() if image is not None else None,
                brightnessApplied=False,
                sharpnessApplied=False,
                frameId=frameId,
                success=True,
                processingTimeMs=self._measureTime(startTime)
            )
        
        if image is None:
            self._logger.warning(f"[{frameId}] No image provided for enhancement")
            return EnhancementServiceResult(
                enhancedImage=None,
                brightnessApplied=False,
                sharpnessApplied=False,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        try:
            # Apply enhancement - returns EnhancementResult
            enhanceResult = self._enhancer.enhance(
                image=image,
                applyBrightness=self._brightnessEnabled,
                applySharpness=self._sharpnessEnabled
            )
            
            processingTimeMs = self._measureTime(startTime)
            
            # Save debug output
            self._saveDebugOutput(
                frameId,
                enhanceResult.image,
                enhanceResult.brightnessApplied,
                enhanceResult.sharpnessApplied
            )
            
            # Log timing
            self._logTiming(frameId, processingTimeMs)
            self._logger.debug(
                f"[{frameId}] Enhanced: brightness={enhanceResult.brightnessApplied}, "
                f"sharpness={enhanceResult.sharpnessApplied}"
            )
            
            return EnhancementServiceResult(
                enhancedImage=enhanceResult.image,
                brightnessApplied=enhanceResult.brightnessApplied,
                sharpnessApplied=enhanceResult.sharpnessApplied,
                frameId=frameId,
                success=True,
                processingTimeMs=processingTimeMs
            )
            
        except Exception as e:
            self._logger.error(f"[{frameId}] Enhancement failed: {e}")
            return EnhancementServiceResult(
                enhancedImage=None,
                brightnessApplied=False,
                sharpnessApplied=False,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
    
    def setEnabled(self, enabled: bool) -> None:
        """Enable or disable enhancement."""
        self._enabled = enabled
        self._logger.info(f"Enhancement {'enabled' if enabled else 'disabled'}")
    
    def isEnabled(self) -> bool:
        """Check if enhancement is enabled."""
        return self._enabled
    
    def setBrightnessEnabled(self, enabled: bool) -> None:
        """Enable or disable brightness enhancement."""
        self._brightnessEnabled = enabled
        self._logger.debug(f"Brightness enhancement: {enabled}")
    
    def isBrightnessEnabled(self) -> bool:
        """Check if brightness enhancement is enabled."""
        return self._brightnessEnabled
    
    def setSharpnessEnabled(self, enabled: bool) -> None:
        """Enable or disable sharpness enhancement."""
        self._sharpnessEnabled = enabled
        self._logger.debug(f"Sharpness enhancement: {enabled}")
    
    def isSharpnessEnabled(self) -> bool:
        """Check if sharpness enhancement is enabled."""
        return self._sharpnessEnabled
    
    def _saveDebugOutput(
        self,
        frameId: str,
        enhancedImage: np.ndarray,
        brightnessApplied: bool,
        sharpnessApplied: bool
    ) -> None:
        """Save debug output for enhancement step."""
        if not self._debugEnabled:
            return
        
        # Save enhanced image
        self._saveDebugImage(frameId, enhancedImage, "enhancement")
        
        # Save enhancement info as JSON
        info = {
            "frameId": frameId,
            "brightnessApplied": brightnessApplied,
            "sharpnessApplied": sharpnessApplied,
            "imageShape": list(enhancedImage.shape) if enhancedImage is not None else None
        }
        self._saveDebugJson(frameId, info, "enhancement")
