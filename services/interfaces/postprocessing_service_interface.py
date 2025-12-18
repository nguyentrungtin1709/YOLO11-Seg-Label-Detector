"""
Postprocessing Service Interface Module.

Defines the interface for text postprocessing operations (Step 8 of the pipeline).
Responsible for fuzzy matching and validation of OCR results.

Follows:
- SRP: Only handles postprocessing operations
- DIP: Depends on ITextProcessor abstraction from core layer
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from core.interfaces.text_processor_interface import LabelData
from core.interfaces.ocr_extractor_interface import TextBlock
from core.interfaces.qr_detector_interface import QrDetectionResult


@dataclass
class PostprocessingServiceResult:
    """
    Result of the postprocessing service.
    
    Attributes:
        labelData: Structured label data after processing.
        isValid: True if QR position matches OCR position.
        frameId: Frame identifier for debug output.
        success: Whether postprocessing was successful.
        processingTimeMs: Time taken for postprocessing.
    """
    labelData: Optional[LabelData]
    isValid: bool
    frameId: str
    success: bool
    processingTimeMs: float = 0.0


class IPostprocessingService(ABC):
    """
    Interface for postprocessing operations (Step 8).
    
    Applies fuzzy matching to correct OCR errors using product/size/color
    databases. Validates OCR results against QR code data.
    """
    
    @abstractmethod
    def process(
        self,
        textBlocks: List[TextBlock],
        qrResult: QrDetectionResult,
        frameId: str
    ) -> PostprocessingServiceResult:
        """
        Process OCR results with fuzzy matching and validation.
        
        Args:
            textBlocks: List of text blocks from OCR.
            qrResult: QR detection result for validation.
            frameId: Frame identifier for debug output.
            
        Returns:
            PostprocessingServiceResult: Processed result with validation.
        """
        pass
    
    @abstractmethod
    def setEnabled(self, enabled: bool) -> None:
        """
        Enable or disable postprocessing.
        
        Args:
            enabled: True to enable postprocessing.
        """
        pass
    
    @abstractmethod
    def isEnabled(self) -> bool:
        """
        Check if postprocessing is enabled.
        
        Returns:
            bool: True if postprocessing is enabled.
        """
        pass
