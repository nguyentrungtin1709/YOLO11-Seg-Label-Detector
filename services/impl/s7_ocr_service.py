"""
S7 OCR Service Implementation.

Step 7 of the pipeline: OCR text extraction.
Creates and manages PaddleOcrExtractor from core layer.

Follows:
- SRP: Only handles OCR operations
- DIP: Depends on IOcrExtractor abstraction (interface)
- OCP: Extends without modifying existing code
"""

import time
import logging
from typing import Optional

import numpy as np

from core.interfaces.ocr_extractor_interface import IOcrExtractor, OcrResult
from core.ocr.paddle_ocr_extractor import PaddleOcrExtractor
from services.interfaces.ocr_service_interface import (
    IOcrService,
    OcrServiceResult
)
from services.interfaces.base_service_interface import BaseService


class S7OcrService(IOcrService, BaseService):
    """
    Step 7: OCR Service Implementation.
    
    Extracts text from images using PaddleOCR.
    Returns structured text blocks with positions and confidence.
    
    Creates PaddleOcrExtractor internally with provided parameters.
    """
    
    SERVICE_NAME = "s7_ocr"
    
    def __init__(
        self,
        enabled: bool = True,
        lang: str = "en",
        useTextlineOrientation: bool = False,
        textDetThresh: float = 0.3,
        textDetBoxThresh: float = 0.5,
        textRecScoreThresh: float = 0.5,
        textDetUnclipRatio: float = 1.5,
        textDetLimitType: str = "min",
        textDetLimitSideLen: int = 736,
        textDetectionModelName: Optional[str] = None,
        textRecognitionModelName: Optional[str] = None,
        precision: str = "fp32",
        enableMkldnn: bool = True,
        mkldnnCacheCapacity: int = 10,
        cpuThreads: int = 8,
        device: str = "cpu",
        debugBasePath: str = "output/debug",
        debugEnabled: bool = False
    ):
        """
        Initialize S7OcrService.
        
        Args:
            enabled: Whether OCR is enabled.
            lang: Language for OCR (default: 'en' for English).
            useTextlineOrientation: Use textline orientation classification.
            textDetThresh: Text detection threshold.
            textDetBoxThresh: Text detection box threshold.
            textRecScoreThresh: Text recognition score threshold.
            textDetUnclipRatio: Text box unclip ratio for expansion.
            textDetLimitType: Image resize limit type ('min' or 'max').
            textDetLimitSideLen: Side length limit for image resize.
            textDetectionModelName: Detection model name (e.g., PP-OCRv5_server_det).
            textRecognitionModelName: Recognition model name (e.g., PP-OCRv5_server_rec).
            precision: Inference precision ('fp32' or 'fp16').
            enableMkldnn: Enable MKL-DNN acceleration.
            mkldnnCacheCapacity: MKL-DNN cache capacity.
            cpuThreads: Number of CPU threads.
            device: Device for inference ('cpu' or 'gpu').
            debugBasePath: Base path for debug output.
            debugEnabled: Whether to save debug output.
        """
        BaseService.__init__(
            self,
            serviceName=self.SERVICE_NAME,
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        
        # Create core OCR extractor implementation
        self._ocrExtractor: IOcrExtractor = PaddleOcrExtractor(
            lang=lang,
            useTextlineOrientation=useTextlineOrientation,
            textDetThresh=textDetThresh,
            textDetBoxThresh=textDetBoxThresh,
            textRecScoreThresh=textRecScoreThresh,
            textDetUnclipRatio=textDetUnclipRatio,
            textDetLimitType=textDetLimitType,
            textDetLimitSideLen=textDetLimitSideLen,
            textDetectionModelName=textDetectionModelName,
            textRecognitionModelName=textRecognitionModelName,
            precision=precision,
            enableMkldnn=enableMkldnn,
            mkldnnCacheCapacity=mkldnnCacheCapacity,
            cpuThreads=cpuThreads,
            device=device
        )
        
        self._enabled = enabled
        
        self._logger.info(
            f"S7OcrService initialized "
            f"(lang={lang}, device={device}, limit_type={textDetLimitType}, "
            f"limit_side_len={textDetLimitSideLen})"
        )
    
    def extractText(
        self,
        image: np.ndarray,
        frameId: str
    ) -> OcrServiceResult:
        """Extract text from an image."""
        startTime = time.time()
        
        # Check if OCR is enabled
        if not self._enabled:
            return OcrServiceResult(
                ocrData=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        if image is None:
            self._logger.warning(f"[{frameId}] No image provided for OCR")
            return OcrServiceResult(
                ocrData=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        try:
            # Run OCR
            ocrResult = self._ocrExtractor.extract(image)
            
            processingTimeMs = self._measureTime(startTime)
            
            if ocrResult is None or not ocrResult.textBlocks:
                self._logger.warning(f"[{frameId}] OCR returned no text blocks")
                # Still consider it a success, just no text found
                return OcrServiceResult(
                    ocrData=ocrResult,
                    frameId=frameId,
                    success=True,
                    processingTimeMs=processingTimeMs
                )
            
            # Save debug output
            self._saveDebugOutput(frameId, ocrResult)
            
            # Log timing
            self._logTiming(frameId, processingTimeMs)
            
            # Log extracted text
            textLines = [block.text for block in ocrResult.textBlocks]
            self._logger.info(
                f"[{frameId}] OCR extracted {len(textLines)} text blocks: {textLines}"
            )
            
            return OcrServiceResult(
                ocrData=ocrResult,
                frameId=frameId,
                success=True,
                processingTimeMs=processingTimeMs
            )
            
        except Exception as e:
            self._logger.error(f"[{frameId}] OCR failed: {e}")
            return OcrServiceResult(
                ocrData=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
    
    def setEnabled(self, enabled: bool) -> None:
        """Enable or disable OCR."""
        self._enabled = enabled
        self._logger.info(f"OCR {'enabled' if enabled else 'disabled'}")
    
    def isEnabled(self) -> bool:
        """Check if OCR is enabled."""
        return self._enabled
    
    def _saveDebugOutput(
        self,
        frameId: str,
        ocrResult: OcrResult
    ) -> None:
        """Save debug output for OCR step."""
        if not self._debugEnabled:
            return
        
        # Save OCR data as JSON
        data = {
            "frameId": frameId,
            "textBlocks": [
                {
                    "text": block.text,
                    "confidence": block.confidence,
                    "bbox": block.bbox
                }
                for block in ocrResult.textBlocks
            ]
        }
        self._saveDebugJson(frameId, data, "ocr")
