"""
OCR Extractor Interface Module.

This module defines the interface and data classes for OCR text extraction.
Follows the Interface Segregation Principle (ISP) from SOLID.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any
import numpy as np


@dataclass
class TextBlock:
    """
    A single text block extracted by OCR.
    
    Attributes:
        text: Extracted text content
        confidence: OCR confidence score (0-1)
        bbox: Bounding box as list of 4 corner points [[x,y], ...]
    """
    text: str
    confidence: float
    bbox: List[List[float]]


@dataclass
class OcrResult:
    """
    Result of OCR extraction.
    
    Attributes:
        textBlocks: List of extracted text blocks
        rawResult: Raw result from OCR engine for debugging
    """
    textBlocks: List[TextBlock]
    rawResult: Any = None


class IOcrExtractor(ABC):
    """
    Interface for OCR text extractor.
    
    Implementations should extract text from images using OCR
    and return structured results with confidence scores.
    """
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> OcrResult:
        """
        Extract text from an image using OCR.
        
        Args:
            image: Input image for OCR processing
            
        Returns:
            OcrResult with list of text blocks
        """
        pass
