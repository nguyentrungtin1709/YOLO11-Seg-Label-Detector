"""
Label Text Processor Implementation.

This module post-processes OCR results by:
1. Identifying field types (position/quantity, product, size, color)
2. Applying fuzzy matching against known valid values
3. Validating against QR code data

Follows the Single Responsibility Principle (SRP) from SOLID.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Optional

from core.interfaces.text_processor_interface import (
    ITextProcessor, 
    LabelData
)
from core.interfaces.ocr_extractor_interface import TextBlock
from core.interfaces.qr_detector_interface import QrDetectionResult
from core.processor.fuzzy_matcher import FuzzyMatcher


class LabelTextProcessor(ITextProcessor):
    """
    Post-processes OCR text with fuzzy matching and validation.
    
    Matches extracted text against databases of valid products,
    sizes, and colors to correct OCR errors.
    """
    
    # Pattern for position/quantity: "1/1", "2/10", etc.
    POSITION_QUANTITY_PATTERN = re.compile(r'^(\d+)\s*/\s*(\d+)$')
    
    def __init__(
        self,
        validProducts: Optional[List[str]] = None,
        validSizes: Optional[List[str]] = None,
        validColors: Optional[List[str]] = None,
        productsJsonPath: Optional[str] = None,
        sizesJsonPath: Optional[str] = None,
        colorsJsonPath: Optional[str] = None,
        minFuzzyScore: float = 0.75,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize LabelTextProcessor.
        
        Args:
            validProducts: List of valid product codes
            validSizes: List of valid sizes
            validColors: List of valid colors
            productsJsonPath: Path to products JSON file
            sizesJsonPath: Path to sizes JSON file
            colorsJsonPath: Path to colors JSON file
            minFuzzyScore: Minimum fuzzy match score (default: 0.75)
            logger: Logger instance for debug output
        """
        self._logger = logger or logging.getLogger(__name__)
        self._minFuzzyScore = minFuzzyScore
        
        # Load valid values from provided lists or JSON files
        self._validProducts = validProducts or []
        self._validSizes = validSizes or []
        self._validColors = validColors or []
        
        # Load from JSON files if paths provided
        if productsJsonPath:
            self._loadProducts(productsJsonPath)
        if sizesJsonPath:
            self._loadSizes(sizesJsonPath)
        if colorsJsonPath:
            self._loadColors(colorsJsonPath)
        
        self._logger.info(
            f"LabelTextProcessor initialized with {len(self._validProducts)} products, "
            f"{len(self._validSizes)} sizes, {len(self._validColors)} colors"
        )
    
    def _loadProducts(self, jsonPath: str) -> None:
        """Load valid products from JSON file."""
        try:
            path = Path(jsonPath)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Extract Code field from each product
                self._validProducts = [
                    str(item.get('Code', '')).strip() 
                    for item in data 
                    if item.get('Code')
                ]
                self._logger.debug(f"Loaded {len(self._validProducts)} products from {jsonPath}")
        except Exception as e:
            self._logger.warning(f"Failed to load products from {jsonPath}: {e}")
    
    def _loadSizes(self, jsonPath: str) -> None:
        """Load valid sizes from JSON file."""
        try:
            path = Path(jsonPath)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Extract name field from each size
                self._validSizes = [
                    str(item.get('name', '')).strip().upper() 
                    for item in data 
                    if item.get('name')
                ]
                self._logger.debug(f"Loaded {len(self._validSizes)} sizes from {jsonPath}")
        except Exception as e:
            self._logger.warning(f"Failed to load sizes from {jsonPath}: {e}")
    
    def _loadColors(self, jsonPath: str) -> None:
        """Load valid colors from JSON file."""
        try:
            path = Path(jsonPath)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Extract name field from each color
                self._validColors = [
                    str(item.get('name', '')).strip().upper() 
                    for item in data 
                    if item.get('name')
                ]
                self._logger.debug(f"Loaded {len(self._validColors)} colors from {jsonPath}")
        except Exception as e:
            self._logger.warning(f"Failed to load colors from {jsonPath}: {e}")
    
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
        result = LabelData()
        result.fieldConfidences = {}
        
        # Copy information from QR code
        result.fullOrderCode = qrResult.text
        result.dateCode = qrResult.dateCode
        result.facility = qrResult.facility
        result.orderType = qrResult.orderType
        result.orderNumber = qrResult.orderNumber
        result.qrPosition = qrResult.position
        
        # Sort text blocks by Y position (top to bottom)
        sortedBlocks = sorted(textBlocks, key=lambda b: b.bbox[0][1])
        
        self._logger.debug(f"Processing {len(sortedBlocks)} text blocks")
        
        for block in sortedBlocks:
            text = block.text.strip()
            textUpper = text.upper()
            confidence = block.confidence
            
            self._logger.debug(f"Processing text: '{text}' (confidence: {confidence:.3f})")
            
            # Try to identify field type and extract value
            if not result.positionQuantity:
                if self._tryParsePositionQuantity(result, text, confidence):
                    continue
            
            if not result.productCode:
                if self._tryMatchProduct(result, textUpper, confidence):
                    continue
            
            if not result.size:
                if self._tryMatchSize(result, textUpper, confidence):
                    continue
            
            if not result.color:
                if self._tryMatchColor(result, textUpper, confidence):
                    continue
        
        # Validate: qrPosition should match ocrPosition
        result.isValid = (
            result.qrPosition == result.ocrPosition and 
            result.ocrPosition > 0
        )
        
        self._logger.info(
            f"Processed label: product={result.productCode}, size={result.size}, "
            f"color={result.color}, position={result.positionQuantity}, valid={result.isValid}"
        )
        
        return result
    
    def _tryParsePositionQuantity(
        self, 
        result: LabelData, 
        text: str, 
        confidence: float
    ) -> bool:
        """
        Try to parse position/quantity from text.
        
        Returns True if successfully parsed.
        """
        match = self.POSITION_QUANTITY_PATTERN.match(text)
        if match:
            result.positionQuantity = text
            result.ocrPosition = int(match.group(1))
            result.quantity = int(match.group(2))
            result.fieldConfidences['positionQuantity'] = confidence
            self._logger.debug(f"Matched position/quantity: {text}")
            return True
        return False
    
    def _tryMatchProduct(
        self, 
        result: LabelData, 
        text: str, 
        confidence: float
    ) -> bool:
        """
        Try to match product code from text.
        
        Returns True if successfully matched.
        """
        # First try exact match in valid products
        if text in self._validProducts:
            result.productCode = text
            result.fieldConfidences['productCode'] = confidence
            self._logger.debug(f"Exact product match: {text}")
            return True
        
        # Try fuzzy match
        matched, score = FuzzyMatcher.bestMatch(
            text, 
            self._validProducts, 
            self._minFuzzyScore
        )
        if matched:
            result.productCode = matched
            result.fieldConfidences['productCode'] = confidence * score
            self._logger.debug(f"Fuzzy product match: {text} -> {matched} (score: {score:.3f})")
            return True
        
        # Accept short alphanumeric codes that look like product codes
        if len(text) <= 8 and (text.isdigit() or text.isalnum()):
            result.productCode = text
            result.fieldConfidences['productCode'] = confidence * 0.7  # Lower confidence
            self._logger.debug(f"Accepted product code: {text}")
            return True
        
        return False
    
    def _tryMatchSize(
        self, 
        result: LabelData, 
        text: str, 
        confidence: float
    ) -> bool:
        """
        Try to match size from text.
        
        Returns True if successfully matched.
        """
        # First try exact match in valid sizes
        if text in self._validSizes:
            result.size = text
            result.fieldConfidences['size'] = confidence
            self._logger.debug(f"Exact size match: {text}")
            return True
        
        # Try fuzzy match
        matched, score = FuzzyMatcher.bestMatch(
            text, 
            self._validSizes, 
            self._minFuzzyScore
        )
        if matched:
            result.size = matched
            result.fieldConfidences['size'] = confidence * score
            self._logger.debug(f"Fuzzy size match: {text} -> {matched} (score: {score:.3f})")
            return True
        
        return False
    
    def _tryMatchColor(
        self, 
        result: LabelData, 
        text: str, 
        confidence: float
    ) -> bool:
        """
        Try to match color from text.
        
        Returns True if successfully matched.
        """
        # First try exact match in valid colors
        if text in self._validColors:
            result.color = text
            result.fieldConfidences['color'] = confidence
            self._logger.debug(f"Exact color match: {text}")
            return True
        
        # Try fuzzy match
        matched, score = FuzzyMatcher.bestMatch(
            text, 
            self._validColors, 
            self._minFuzzyScore
        )
        if matched:
            result.color = matched
            result.fieldConfidences['color'] = confidence * score
            self._logger.debug(f"Fuzzy color match: {text} -> {matched} (score: {score:.3f})")
            return True
        
        return False
