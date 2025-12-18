"""
PaddleOCR Extractor Implementation.

This module provides OCR text extraction using PaddleOCR library.
CPU-only mode is used for broader compatibility.
Follows the Single Responsibility Principle (SRP) from SOLID.

Note: Compatible with PaddleOCR 3.x API.
"""

import logging
from typing import Optional

import numpy as np

from core.interfaces.ocr_extractor_interface import (
    IOcrExtractor, 
    OcrResult, 
    TextBlock
)


class PaddleOcrExtractor(IOcrExtractor):
    """
    OCR text extractor using PaddleOCR.
    
    Extracts text from images with confidence scores and bounding boxes.
    Uses CPU mode for compatibility (no GPU required).
    
    Compatible with PaddleOCR 3.x API.
    """
    
    def __init__(
        self,
        lang: str = 'en',
        useTextlineOrientation: bool = False,
        textDetThresh: float = 0.3,
        textDetBoxThresh: float = 0.5,
        textRecScoreThresh: float = 0.5,
        device: str = 'cpu',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize PaddleOcrExtractor.
        
        Args:
            lang: Language for OCR (default: 'en' for English)
            useTextlineOrientation: Use textline orientation classification
            textDetThresh: Text detection threshold
            textDetBoxThresh: Text detection box threshold
            textRecScoreThresh: Text recognition score threshold
            device: Device for inference ('cpu' or 'gpu')
            logger: Logger instance for debug output
        """
        self._logger = logger or logging.getLogger(__name__)
        self._ocrEngine = None
        
        # Store config for lazy initialization
        # PaddleOCR 3.x API parameters
        self._config = {
            'lang': lang,
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
            'use_textline_orientation': useTextlineOrientation,
            'text_det_thresh': textDetThresh,
            'text_det_box_thresh': textDetBoxThresh,
            'text_rec_score_thresh': textRecScoreThresh,
            'device': device
        }
        
        self._logger.info(
            f"PaddleOcrExtractor initialized with lang={lang}, device={device}"
        )
    
    def _ensureOcrEngine(self) -> None:
        """Lazily initialize PaddleOCR engine on first use."""
        if self._ocrEngine is None:
            try:
                from paddleocr import PaddleOCR
                self._ocrEngine = PaddleOCR(**self._config)
                self._logger.info("PaddleOCR engine initialized successfully")
            except ImportError as e:
                self._logger.error(
                    f"Failed to import PaddleOCR. "
                    f"Please install: pip install paddlepaddle paddleocr. Error: {e}"
                )
                raise
            except Exception as e:
                self._logger.error(f"Failed to initialize PaddleOCR: {e}")
                raise
    
    def extract(self, image: np.ndarray) -> OcrResult:
        """
        Extract text from an image using PaddleOCR.
        
        Args:
            image: Input image for OCR processing (BGR or grayscale)
            
        Returns:
            OcrResult with list of text blocks containing text,
            confidence scores, and bounding boxes
        """
        self._ensureOcrEngine()
        
        textBlocks = []
        rawResult = None
        
        try:
            # Run OCR - PaddleOCR 3.x uses predict() method
            rawResult = self._ocrEngine.predict(image)
            
            # Parse results based on PaddleOCR 3.x format
            if rawResult:
                for res in rawResult:
                    # Access structured output
                    recTexts = res.get('rec_texts', [])
                    recScores = res.get('rec_scores', [])
                    dtPolys = res.get('dt_polys', [])
                    
                    for i, text in enumerate(recTexts):
                        confidence = recScores[i] if i < len(recScores) else 0.0
                        bbox = dtPolys[i].tolist() if i < len(dtPolys) else []
                        
                        textBlocks.append(TextBlock(
                            text=str(text),
                            confidence=float(confidence),
                            bbox=bbox
                        ))
                        
                        self._logger.debug(
                            f"OCR detected: '{text}' (confidence: {confidence:.3f})"
                        )
            else:
                self._logger.debug("OCR returned no results")
                
        except Exception as e:
            self._logger.error(f"Error during OCR extraction: {e}")
        
        self._logger.info(f"OCR extracted {len(textBlocks)} text blocks")
        
        return OcrResult(textBlocks=textBlocks, rawResult=rawResult)
