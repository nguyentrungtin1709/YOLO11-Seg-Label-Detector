"""
S8 Postprocessing Service Implementation.

Step 8 of the pipeline: Text postprocessing with fuzzy matching and validation.
Creates and manages LabelTextProcessor from core layer.

Follows:
- SRP: Only handles postprocessing operations
- DIP: Depends on ITextProcessor abstraction (interface)
- OCP: Extends without modifying existing code
"""

import time
import logging
from dataclasses import asdict
from typing import Optional, List

import numpy as np

from core.interfaces.text_processor_interface import ITextProcessor, LabelData
from core.interfaces.ocr_extractor_interface import TextBlock
from core.interfaces.qr_detector_interface import QrDetectionResult
from core.processor.label_text_processor import LabelTextProcessor
from services.interfaces.postprocessing_service_interface import (
    IPostprocessingService,
    PostprocessingServiceResult
)
from services.interfaces.base_service_interface import BaseService


class S8PostprocessingService(IPostprocessingService, BaseService):
    """
    Step 8: Postprocessing Service Implementation.
    
    Applies fuzzy matching to correct OCR errors using product/size/color
    databases. Validates OCR results against QR code data.
    
    Creates LabelTextProcessor internally with provided parameters.
    """
    
    SERVICE_NAME = "s8_postprocessing"
    
    def __init__(
        self,
        enabled: bool = True,
        minFuzzyScore: float = 0.80,
        productsJsonPath: Optional[str] = None,
        sizesJsonPath: Optional[str] = None,
        colorsJsonPath: Optional[str] = None,
        debugBasePath: str = "output/debug",
        debugEnabled: bool = False
    ):
        """
        Initialize S8PostprocessingService.
        
        Args:
            enabled: Whether postprocessing is enabled.
            minFuzzyScore: Minimum fuzzy matching score.
            productsJsonPath: Path to products JSON file.
            sizesJsonPath: Path to sizes JSON file.
            colorsJsonPath: Path to colors JSON file.
            debugBasePath: Base path for debug output.
            debugEnabled: Whether to save debug output.
        """
        BaseService.__init__(
            self,
            serviceName=self.SERVICE_NAME,
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        
        # Create core text processor implementation
        self._textProcessor: ITextProcessor = LabelTextProcessor(
            productsJsonPath=productsJsonPath,
            sizesJsonPath=sizesJsonPath,
            colorsJsonPath=colorsJsonPath,
            minFuzzyScore=minFuzzyScore
        )
        
        self._enabled = enabled
        
        self._logger.info(
            f"S8PostprocessingService initialized "
            f"(minFuzzyScore={minFuzzyScore})"
        )
    
    def process(
        self,
        textBlocks: List[TextBlock],
        qrResult: QrDetectionResult,
        frameId: str
    ) -> PostprocessingServiceResult:
        """Process OCR results with fuzzy matching and validation."""
        startTime = time.time()
        
        # Check if postprocessing is enabled
        if not self._enabled:
            return PostprocessingServiceResult(
                labelData=None,
                isValid=False,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        if textBlocks is None:
            textBlocks = []
        
        if qrResult is None:
            self._logger.warning(f"[{frameId}] No QR result for postprocessing")
            return PostprocessingServiceResult(
                labelData=None,
                isValid=False,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        try:
            # Run postprocessing
            labelData = self._textProcessor.process(textBlocks, qrResult)
            
            processingTimeMs = self._measureTime(startTime)
            
            # Save debug output
            self._saveDebugOutput(frameId, labelData, processingTimeMs)
            
            # Log timing
            self._logTiming(frameId, processingTimeMs)
            self._logger.info(
                f"[{frameId}] Postprocessing complete: "
                f"product={labelData.productCode}, size={labelData.size}, "
                f"color={labelData.color}, valid={labelData.isValid}"
            )
            
            return PostprocessingServiceResult(
                labelData=labelData,
                isValid=labelData.isValid,
                frameId=frameId,
                success=True,
                processingTimeMs=processingTimeMs
            )
            
        except Exception as e:
            self._logger.error(f"[{frameId}] Postprocessing failed: {e}")
            return PostprocessingServiceResult(
                labelData=None,
                isValid=False,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
    
    def setEnabled(self, enabled: bool) -> None:
        """Enable or disable postprocessing."""
        self._enabled = enabled
        self._logger.info(f"Postprocessing {'enabled' if enabled else 'disabled'}")
    
    def isEnabled(self) -> bool:
        """Check if postprocessing is enabled."""
        return self._enabled
    
    def _saveDebugOutput(
        self,
        frameId: str,
        labelData: LabelData,
        processingTimeMs: float
    ) -> None:
        """Save debug output for postprocessing step."""
        if not self._debugEnabled:
            return
        
        # Save result as JSON
        data = {
            "frameId": frameId,
            "labelData": asdict(labelData),
            "processingTimeMs": processingTimeMs
        }
        self._saveDebugJson(frameId, data, "result")
