"""
OCR Service Interface Module.

Defines the interface for OCR operations (Step 7 of the pipeline).
Responsible for extracting text from images using OCR.

Follows:
- SRP: Only handles OCR operations
- DIP: Depends on IOcrExtractor abstraction from core layer
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

from core.interfaces.ocr_extractor_interface import OcrResult


@dataclass
class OcrServiceResult:
    """
    Result of the OCR service.
    
    Attributes:
        ocrData: OCR extraction result with text blocks.
        frameId: Frame identifier for debug output.
        success: Whether OCR was successful.
        processingTimeMs: Time taken for OCR.
    """
    ocrData: Optional[OcrResult]
    frameId: str
    success: bool
    processingTimeMs: float = 0.0


class IOcrService(ABC):
    """
    Interface for OCR operations (Step 7).
    
    Extracts text from images using PaddleOCR.
    Returns structured text blocks with positions and confidence.
    """
    
    @abstractmethod
    def extractText(
        self,
        image: np.ndarray,
        frameId: str
    ) -> OcrServiceResult:
        """
        Extract text from an image.
        
        Args:
            image: Input image (BGR format).
            frameId: Frame identifier for debug output.
            
        Returns:
            OcrServiceResult: OCR result with text blocks.
        """
        pass
    
    @abstractmethod
    def setEnabled(self, enabled: bool) -> None:
        """
        Enable or disable OCR.
        
        Args:
            enabled: True to enable OCR.
        """
        pass
    
    @abstractmethod
    def isEnabled(self) -> bool:
        """
        Check if OCR is enabled.
        
        Returns:
            bool: True if OCR is enabled.
        """
        pass
