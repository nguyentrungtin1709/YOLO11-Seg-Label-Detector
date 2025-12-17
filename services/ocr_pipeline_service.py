"""
OCR Pipeline Service.

This module orchestrates the complete OCR pipeline:
1. QR Detection (Step 5)
2. Component Extraction (Step 6)
3. OCR Text Extraction (Step 7)
4. Text Post-Processing (Step 8)

Follows the Single Responsibility Principle (SRP) and
Dependency Inversion Principle (DIP) from SOLID.
"""

import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from core.interfaces.qr_detector_interface import IQrDetector, QrDetectionResult
from core.interfaces.component_extractor_interface import (
    IComponentExtractor, 
    ComponentResult
)
from core.interfaces.ocr_extractor_interface import IOcrExtractor, OcrResult
from core.interfaces.text_processor_interface import ITextProcessor, LabelData


@dataclass
class OcrPipelineResult:
    """
    Complete result from OCR pipeline.
    
    Attributes:
        labelData: Structured label data after post-processing
        qrResult: QR detection result
        componentResult: Extracted component regions
        ocrResult: Raw OCR result
        processingTimeMs: Total processing time in milliseconds
    """
    labelData: LabelData
    qrResult: Optional[QrDetectionResult]
    componentResult: Optional[ComponentResult]
    ocrResult: Optional[OcrResult]
    processingTimeMs: float


class OcrPipelineService:
    """
    Orchestrates the complete OCR pipeline with debug output.
    
    Coordinates QR detection, component extraction, OCR, and
    post-processing stages. Optionally saves debug output at each step.
    """
    
    def __init__(
        self,
        qrDetector: IQrDetector,
        componentExtractor: IComponentExtractor,
        ocrExtractor: IOcrExtractor,
        textProcessor: ITextProcessor,
        debugEnabled: bool = False,
        debugBasePath: str = "output/debug",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize OcrPipelineService.
        
        Args:
            qrDetector: QR code detector implementation
            componentExtractor: Component extractor implementation
            ocrExtractor: OCR extractor implementation
            textProcessor: Text post-processor implementation
            debugEnabled: Enable saving debug output
            debugBasePath: Base path for debug output files
            logger: Logger instance for debug output
        """
        self._qrDetector = qrDetector
        self._componentExtractor = componentExtractor
        self._ocrExtractor = ocrExtractor
        self._textProcessor = textProcessor
        self._debugEnabled = debugEnabled
        self._debugBasePath = Path(debugBasePath)
        self._logger = logger or logging.getLogger(__name__)
        
        # Create debug directories if enabled
        if self._debugEnabled:
            self._ensureDebugDirectories()
        
        self._logger.info(
            f"OcrPipelineService initialized (debug={debugEnabled})"
        )
    
    def setDebugEnabled(self, enabled: bool) -> None:
        """
        Enable or disable debug output.
        
        Args:
            enabled: True to enable debug output
        """
        self._debugEnabled = enabled
        if enabled:
            self._ensureDebugDirectories()
        self._logger.info(f"OCR Pipeline debug {'enabled' if enabled else 'disabled'}")
    
    def isDebugEnabled(self) -> bool:
        """Get debug enabled state."""
        return self._debugEnabled
    
    def _ensureDebugDirectories(self) -> None:
        """Create debug output directories if they don't exist."""
        for subdir in ["qr-code", "components", "ocr-raw-text", "result"]:
            (self._debugBasePath / subdir).mkdir(parents=True, exist_ok=True)
    
    def process(
        self, 
        image: np.ndarray, 
        timestamp: Optional[str] = None
    ) -> Optional[OcrPipelineResult]:
        """
        Process a label image through the complete OCR pipeline.
        
        Args:
            image: Preprocessed label image (from steps 1-4)
            timestamp: Optional timestamp for debug files
            
        Returns:
            OcrPipelineResult if successful, None if QR detection fails
        """
        startTime = time.time()
        ts = timestamp or time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Step 5: Detect QR code
        self._logger.debug("Step 5: Detecting QR code...")
        qrResult = self._qrDetector.detect(image)
        
        if qrResult is None:
            self._logger.warning("No QR code detected, aborting pipeline")
            return None
        
        self._saveQrDebug(qrResult, ts)
        self._logger.info(f"QR detected: {qrResult.text}")
        
        # Step 6: Extract components
        self._logger.debug("Step 6: Extracting components...")
        componentResult = self._componentExtractor.extractAndMerge(
            image, 
            qrResult.polygon
        )
        
        if componentResult is None:
            self._logger.warning("Failed to extract components, aborting pipeline")
            return None
        
        self._saveComponentsDebug(componentResult, ts)
        
        # Step 7: OCR extraction
        self._logger.debug("Step 7: Running OCR...")
        ocrResult = self._ocrExtractor.extract(componentResult.mergedImage)
        
        if not ocrResult.textBlocks:
            self._logger.warning("OCR returned no text blocks")
            # Continue with empty result rather than aborting
        
        self._saveOcrDebug(ocrResult, ts)
        
        # Step 8: Post-processing
        self._logger.debug("Step 8: Post-processing text...")
        labelData = self._textProcessor.process(ocrResult.textBlocks, qrResult)
        
        processingTimeMs = (time.time() - startTime) * 1000
        
        result = OcrPipelineResult(
            labelData=labelData,
            qrResult=qrResult,
            componentResult=componentResult,
            ocrResult=ocrResult,
            processingTimeMs=processingTimeMs
        )
        
        self._saveResultDebug(result, ts)
        
        self._logger.info(
            f"OCR pipeline completed in {processingTimeMs:.2f}ms, "
            f"valid={labelData.isValid}"
        )
        
        return result
    
    def _saveQrDebug(self, qrResult: QrDetectionResult, ts: str) -> None:
        """Save Step 5 QR detection debug output."""
        if not self._debugEnabled:
            return
        
        try:
            data = {
                "text": qrResult.text,
                "polygon": qrResult.polygon,
                "rect": qrResult.rect,
                "confidence": qrResult.confidence,
                "parsed": {
                    "dateCode": qrResult.dateCode,
                    "facility": qrResult.facility,
                    "orderType": qrResult.orderType,
                    "orderNumber": qrResult.orderNumber,
                    "position": qrResult.position
                }
            }
            
            path = self._debugBasePath / "qr-code" / f"qr_{ts}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            self._logger.debug(f"Saved QR debug to {path}")
            
        except Exception as e:
            self._logger.warning(f"Failed to save QR debug: {e}")
    
    def _saveComponentsDebug(self, componentResult: ComponentResult, ts: str) -> None:
        """Save Step 6 component extraction debug output."""
        if not self._debugEnabled:
            return
        
        try:
            basePath = self._debugBasePath / "components"
            
            cv2.imwrite(
                str(basePath / f"merged_{ts}.png"), 
                componentResult.mergedImage
            )
            cv2.imwrite(
                str(basePath / f"above_qr_{ts}.png"), 
                componentResult.aboveQrRoi
            )
            cv2.imwrite(
                str(basePath / f"below_qr_{ts}.png"), 
                componentResult.belowQrRoi
            )
            
            self._logger.debug(f"Saved component debug to {basePath}")
            
        except Exception as e:
            self._logger.warning(f"Failed to save components debug: {e}")
    
    def _saveOcrDebug(self, ocrResult: OcrResult, ts: str) -> None:
        """Save Step 7 OCR debug output."""
        if not self._debugEnabled:
            return
        
        try:
            data = {
                "textBlocks": [
                    {
                        "text": block.text,
                        "confidence": block.confidence,
                        "bbox": block.bbox
                    } 
                    for block in ocrResult.textBlocks
                ]
            }
            
            path = self._debugBasePath / "ocr-raw-text" / f"ocr_{ts}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self._logger.debug(f"Saved OCR debug to {path}")
            
        except Exception as e:
            self._logger.warning(f"Failed to save OCR debug: {e}")
    
    def _saveResultDebug(self, result: OcrPipelineResult, ts: str) -> None:
        """Save Step 8 final result debug output."""
        if not self._debugEnabled:
            return
        
        try:
            data = {
                "labelData": asdict(result.labelData),
                "processingTimeMs": result.processingTimeMs
            }
            
            path = self._debugBasePath / "result" / f"result_{ts}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self._logger.debug(f"Saved result debug to {path}")
            
        except Exception as e:
            self._logger.warning(f"Failed to save result debug: {e}")
