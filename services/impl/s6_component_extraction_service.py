"""
S6 Component Extraction Service Implementation.

Step 6 of the pipeline: Extract and merge text regions.
Creates and manages LabelComponentExtractor from core layer.

Follows:
- SRP: Only handles component extraction operations
- DIP: Depends on IComponentExtractor abstraction (interface)
- OCP: Extends without modifying existing code
"""

import time
import logging
from typing import Optional, List

import numpy as np

from core.interfaces.component_extractor_interface import (
    IComponentExtractor,
    ComponentResult
)
from core.extractor.label_component_extractor import LabelComponentExtractor
from services.interfaces.component_extraction_service_interface import (
    IComponentExtractionService,
    ComponentExtractionServiceResult
)
from services.interfaces.base_service_interface import BaseService


class S6ComponentExtractionService(IComponentExtractionService, BaseService):
    """
    Step 6: Component Extraction Service Implementation.
    
    Extracts text regions based on QR code position and merges them
    into a single image for efficient OCR processing.
    
    Creates LabelComponentExtractor internally with provided parameters.
    """
    
    SERVICE_NAME = "s6_component_extraction"
    
    def __init__(
        self,
        enabled: bool = True,
        aboveQrWidthRatio: float = 0.35,
        aboveQrHeightRatio: float = 0.20,
        belowQrWidthRatio: float = 0.65,
        belowQrHeightRatio: float = 0.45,
        padding: int = 5,
        aboveQrScaleFactor: float = 2.0,
        debugBasePath: str = "output/debug",
        debugEnabled: bool = False
    ):
        """
        Initialize S6ComponentExtractionService.
        
        Args:
            enabled: Whether component extraction is enabled.
            aboveQrWidthRatio: Width ratio of above-QR region.
            aboveQrHeightRatio: Height ratio of above-QR region.
            belowQrWidthRatio: Width ratio of below-QR region.
            belowQrHeightRatio: Height ratio of below-QR region.
            padding: Padding around extracted regions.
            aboveQrScaleFactor: Scale factor for above-QR region (default: 2.0).
            debugBasePath: Base path for debug output.
            debugEnabled: Whether to save debug output.
        """
        BaseService.__init__(
            self,
            serviceName=self.SERVICE_NAME,
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        
        # Create core component extractor implementation
        self._componentExtractor: IComponentExtractor = LabelComponentExtractor(
            aboveQrWidthRatio=aboveQrWidthRatio,
            aboveQrHeightRatio=aboveQrHeightRatio,
            belowQrWidthRatio=belowQrWidthRatio,
            belowQrHeightRatio=belowQrHeightRatio,
            padding=padding,
            aboveQrScaleFactor=aboveQrScaleFactor
        )
        
        self._enabled = enabled
        
        self._logger.info(
            f"S6ComponentExtractionService initialized "
            f"(aboveQr={aboveQrWidthRatio}x{aboveQrHeightRatio}, "
            f"belowQr={belowQrWidthRatio}x{belowQrHeightRatio}, "
            f"scale={aboveQrScaleFactor}x, "
            f"input=grayscale)"
        )
    
    def extractComponents(
        self,
        image: np.ndarray,
        qrPolygon: List[List[int]],
        frameId: str
    ) -> ComponentExtractionServiceResult:
        """Extract and merge text components from a grayscale image.
        
        Args:
            image: Grayscale image from S4 enhancement (H, W)
            qrPolygon: QR code polygon coordinates
            frameId: Frame identifier
            
        Returns:
            ComponentExtractionServiceResult with grayscale merged image
        """
        startTime = time.time()
        
        # Check if component extraction is enabled
        if not self._enabled:
            return ComponentExtractionServiceResult(
                componentData=None,
                mergedImage=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        if image is None:
            self._logger.warning(f"[{frameId}] No image provided for component extraction")
            return ComponentExtractionServiceResult(
                componentData=None,
                mergedImage=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        if not qrPolygon or len(qrPolygon) < 4:
            self._logger.warning(f"[{frameId}] Invalid QR polygon for component extraction")
            return ComponentExtractionServiceResult(
                componentData=None,
                mergedImage=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        try:
            # Run component extraction
            componentResult = self._componentExtractor.extractAndMerge(
                image, 
                qrPolygon
            )
            
            processingTimeMs = self._measureTime(startTime)
            
            if componentResult is None:
                self._logger.warning(f"[{frameId}] Component extraction returned no result")
                return ComponentExtractionServiceResult(
                    componentData=None,
                    mergedImage=None,
                    frameId=frameId,
                    success=False,
                    processingTimeMs=processingTimeMs
                )
            
            # Save debug output
            self._saveDebugOutput(frameId, componentResult)
            
            # Log timing
            self._logTiming(frameId, processingTimeMs)
            self._logger.debug(
                f"[{frameId}] Components extracted: "
                f"merged shape={componentResult.mergedImage.shape if componentResult.mergedImage is not None else None}"
            )
            
            return ComponentExtractionServiceResult(
                componentData=componentResult,
                mergedImage=componentResult.mergedImage,
                frameId=frameId,
                success=True,
                processingTimeMs=processingTimeMs
            )
            
        except Exception as e:
            self._logger.error(f"[{frameId}] Component extraction failed: {e}")
            return ComponentExtractionServiceResult(
                componentData=None,
                mergedImage=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
    
    def setEnabled(self, enabled: bool) -> None:
        """Enable or disable component extraction."""
        self._enabled = enabled
        self._logger.info(f"Component extraction {'enabled' if enabled else 'disabled'}")
    
    def isEnabled(self) -> bool:
        """Check if component extraction is enabled."""
        return self._enabled
    
    def _saveDebugOutput(
        self,
        frameId: str,
        componentResult: ComponentResult
    ) -> None:
        """Save debug output for component extraction step."""
        if not self._debugEnabled:
            return
        
        # Save merged image
        if componentResult.mergedImage is not None:
            self._saveDebugImage(frameId, componentResult.mergedImage, "components")
        
        # Save above QR ROI
        if componentResult.aboveQrRoi is not None:
            self._saveDebugImage(frameId, componentResult.aboveQrRoi, "above_qr")
        
        # Save below QR ROI
        if componentResult.belowQrRoi is not None:
            self._saveDebugImage(frameId, componentResult.belowQrRoi, "below_qr")
        
        # Save component info as JSON
        info = {
            "frameId": frameId,
            "mergedImageShape": list(componentResult.mergedImage.shape) if componentResult.mergedImage is not None else None,
            "aboveQrRoiShape": list(componentResult.aboveQrRoi.shape) if componentResult.aboveQrRoi is not None else None,
            "belowQrRoiShape": list(componentResult.belowQrRoi.shape) if componentResult.belowQrRoi is not None else None
        }
        self._saveDebugJson(frameId, info, "components")
