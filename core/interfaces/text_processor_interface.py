"""
Text Processor Interface Module.

This module defines the interface and data classes for post-processing
OCR text results with fuzzy matching and validation.
Follows the Interface Segregation Principle (ISP) from SOLID.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict

from core.interfaces.ocr_extractor_interface import TextBlock
from core.interfaces.qr_detector_interface import QrDetectionResult


@dataclass
class LabelData:
    """
    Structured label data after post-processing.
    
    Attributes:
        # QR Code Information
        fullOrderCode: Full QR code content (e.g., "110125-VA-M-000002-2/1")
        dateCode: Date code from QR (MMDDYY)
        facility: Facility code from QR (VA, GA, ...)
        orderType: Order type from QR (M, S, ...)
        orderNumber: Order number from QR
        qrPosition: Position from QR code
        qrRevisionCount: Revision count from QR code (0 if not present)
        
        # OCR Information
        positionQuantity: Position/quantity string (e.g., "1/1")
        ocrPosition: Position parsed from OCR
        quantity: Quantity parsed from OCR
        productCode: Product code (e.g., "340")
        size: Size (e.g., "3T")
        color: Color (e.g., "MIDNIGHT")
        
        # Validation
        isValid: True if qrPosition matches ocrPosition
        fieldConfidences: Confidence scores for each field
    """
    # QR Code Information
    fullOrderCode: str = ""
    dateCode: str = ""
    facility: str = ""
    orderType: str = ""
    orderNumber: str = ""
    qrPosition: int = 0
    qrRevisionCount: int = 0
    
    # OCR Information
    positionQuantity: str = ""
    ocrPosition: int = 0
    quantity: int = 0
    productCode: str = ""
    size: str = ""
    color: str = ""
    
    # Validation
    isValid: bool = False
    fieldConfidences: Dict[str, float] = field(default_factory=dict)


class ITextProcessor(ABC):
    """
    Interface for text post-processor.
    
    Implementations should process OCR results, apply fuzzy matching
    to correct text, and validate against QR code data.
    """
    
    @abstractmethod
    def process(
        self, 
        textBlocks: List[TextBlock], 
        qrResult: QrDetectionResult
    ) -> LabelData:
        """
        Process OCR text blocks and create structured label data.
        
        Args:
            textBlocks: List of text blocks from OCR
            qrResult: QR detection result for validation
            
        Returns:
            LabelData with structured and validated label information
        """
        pass
