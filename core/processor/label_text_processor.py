"""
Label Text Processor Implementation.

This module post-processes OCR results by:
1. Identifying field types (position/quantity, product, size, color)
2. Applying fuzzy matching against known valid values
3. Validating against QR code data
4. Sequential field extraction with index tracking
5. Intelligent fallback for unmatched fields

Follows the Single Responsibility Principle (SRP) from SOLID.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

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
    
    Uses sequential field extraction strategy:
    1. Find position/quantity (with recovery cases)
    2. Find product from position+1
    3. Find size from max(product+1, position+1)
    4. Find color from max(size+1, product+1, position+1)
    5. Apply fallback with unused indexes
    """
    
    # Pattern for position/quantity: "1/1", "2/10", etc.
    POSITION_QUANTITY_PATTERN = re.compile(r'^(\d+)\s*/\s*(\d+)$')
    
    # Pattern for recovery: "/" misread as "1", "|", "l", "I", "!", "t", "i", "j"
    # Examples: "313" -> "3/3", "21|5" -> "2/5"
    POSITION_RECOVERY_SEPARATORS = ['1', '|', 'l', 'I', '!', 't', 'i', 'j']
    
    # Regex patterns verified from actual data
    # Product: 3-9 chars, alphanumeric, supports lowercase (e.g., "3001rcy", "ic47mr")
    PRODUCT_PATTERN = re.compile(r'^[A-Za-z0-9]{3,9}$')
    
    # Size: 1-7 chars, uppercase alphanumeric with "-" and "/" (e.g., "M", "12M-18M")
    SIZE_PATTERN = re.compile(r'^[A-Z0-9/-]{1,7}$')
    
    # Color: 3+ chars, letters with spaces and "/" only, no digits (e.g., "RED", "SOLID PREMIUM HEATHER")
    COLOR_PATTERN = re.compile(r'^[A-Za-z /]{3,}$')
    
    def __init__(
        self,
        validProducts: Optional[List[str]] = None,
        validSizes: Optional[List[str]] = None,
        validColors: Optional[List[str]] = None,
        productsJsonPath: Optional[str] = None,
        sizesJsonPath: Optional[str] = None,
        colorsJsonPath: Optional[str] = None,
        minFuzzyScore: float = 0.90,  # Updated from 0.80 to 0.90
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
            minFuzzyScore: Minimum fuzzy match score (default: 0.90)
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
            f"{len(self._validSizes)} sizes, {len(self._validColors)} colors, "
            f"minFuzzyScore={minFuzzyScore}"
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
        
        Uses sequential extraction strategy:
        1. Find position/quantity (REQUIRED - stop if not found)
        2. Find product from position_index+1
        3. Find size from max(product_index+1, position_index+1)
        4. Find color from max(size_index+1, product_index+1, position_index+1)
        5. Apply fallback for unmatched fields (excluding used indexes)
        
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
        result.qrRevisionCount = qrResult.revisionCount
        
        # Sort text blocks by Y position (top to bottom)
        sortedBlocks = sorted(textBlocks, key=lambda b: b.bbox[0][1])
        
        self._logger.debug(f"Processing {len(sortedBlocks)} text blocks")
        
        # Track used indexes to avoid reusing blocks
        usedIndexes = set()
        
        # PHASE 1: Extract Position/Quantity (REQUIRED)
        indexPosition, position, quantity, posConfidence = self._extractPositionQuantity(
            sortedBlocks, qrResult
        )
        
        if indexPosition == -1:
            # Position/quantity NOT found → STOP processing
            self._logger.warning("Position/quantity not found - stopping processing")
            result.isValid = False
            return result
        
        # Position/quantity found
        result.positionQuantity = f"{position}/{quantity}"
        result.ocrPosition = position
        result.quantity = quantity
        result.fieldConfidences['positionQuantity'] = posConfidence
        usedIndexes.add(indexPosition)
        
        self._logger.info(
            f"Phase 1: Position/quantity found at index {indexPosition}: {result.positionQuantity}"
        )
        
        # PHASE 2: Extract Product (from position_index + 1)
        indexProduct = self._extractProduct(
            sortedBlocks, startIndex=indexPosition + 1, usedIndexes=usedIndexes, result=result
        )
        
        if indexProduct != -1:
            usedIndexes.add(indexProduct)
            self._logger.info(f"Phase 2: Product found at index {indexProduct}: {result.productCode}")
        else:
            self._logger.debug("Phase 2: Product not found in sequential search")
        
        # PHASE 3: Extract Size (from max(product_index + 1, position_index + 1))
        startIndex = max(indexProduct + 1, indexPosition + 1) if indexProduct != -1 else indexPosition + 1
        indexSize = self._extractSize(
            sortedBlocks, startIndex=startIndex, usedIndexes=usedIndexes, result=result
        )
        
        if indexSize != -1:
            usedIndexes.add(indexSize)
            self._logger.info(f"Phase 3: Size found at index {indexSize}: {result.size}")
        else:
            self._logger.debug("Phase 3: Size not found in sequential search")
        
        # PHASE 4: Extract Color (from max(size_index + 1, product_index + 1, position_index + 1))
        startIndex = max(
            indexSize + 1 if indexSize != -1 else 0,
            indexProduct + 1 if indexProduct != -1 else 0,
            indexPosition + 1
        )
        indexColor = self._extractColor(
            sortedBlocks, startIndex=startIndex, usedIndexes=usedIndexes, result=result
        )
        
        if indexColor != -1:
            usedIndexes.add(indexColor)
            self._logger.info(f"Phase 4: Color found at index {indexColor}: {result.color}")
        else:
            self._logger.debug("Phase 4: Color not found in sequential search")
        
        # PHASE 5: Fallback for unmatched fields (excluding used indexes)
        if not result.productCode:
            self._logger.debug("Phase 5: Trying fallback for Product")
            self._fallbackExtractProduct(sortedBlocks, usedIndexes, result)
        
        if not result.size:
            self._logger.debug("Phase 5: Trying fallback for Size")
            self._fallbackExtractSize(sortedBlocks, usedIndexes, result)
        
        if not result.color:
            self._logger.debug("Phase 5: Trying fallback for Color")
            self._fallbackExtractColor(sortedBlocks, usedIndexes, result)
        
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
    
    
    def _extractPositionQuantity(
        self,
        sortedBlocks: List[TextBlock],
        qrResult: QrDetectionResult
    ) -> Tuple[int, int, int, float]:
        """
        Extract position/quantity from text blocks.
        
        Handles multiple cases:
        - Case 1: Standard "3/5" format (with QR validation)
        - Case 2A: 3-char recovery "315" → "3/5" (separator in POSITION_RECOVERY_SEPARATORS)
        - Case 2B: 2-char recovery "35" → "3/5" (missing separator)
        - Fallback: Scan all remaining blocks for strict "number/number" format
        
        All cases MUST satisfy: quantity >= position
        
        Args:
            sortedBlocks: List of sorted text blocks
            qrResult: QR detection result for validation
            
        Returns:
            Tuple of (index, position, quantity, confidence)
            Returns (-1, 0, 0, 0.0) if not found
        """
        qrPosition = qrResult.position if qrResult.position > 0 else None
        qrPosStr = str(qrPosition) if qrPosition else None
        
        self._logger.debug(f"Extracting position/quantity (QR position: {qrPosition})")
        
        for i, block in enumerate(sortedBlocks):
            text = block.text.strip()
            confidence = block.confidence
            
            # Case 1: Standard format "3/5"
            match = self.POSITION_QUANTITY_PATTERN.match(text)
            if match:
                position = int(match.group(1))
                quantity = int(match.group(2))
                
                # Validate: quantity >= position
                if quantity < position:
                    self._logger.debug(
                        f"[Case 1] Invalid: {text} (quantity {quantity} < position {position})"
                    )
                    continue
                
                # Validate against QR position if available
                if qrPosition and position != qrPosition:
                    self._logger.warning(
                        f"[Case 1] Position mismatch: OCR={position}, QR={qrPosition}"
                    )
                    # Still accept but log warning
                
                self._logger.info(f"[Case 1] Matched position/quantity at index {i}: {text}")
                return (i, position, quantity, confidence)
            
            # Case 2: Recovery cases (only if QR position available)
            if qrPosStr and text.startswith(qrPosStr):
                afterPos = text[len(qrPosStr):]
                
                # Case 2A: 3-char format "315" → "3/5" (separator misread)
                if len(afterPos) == 2:
                    sepChar = afterPos[0]
                    quantityChar = afterPos[1]
                    
                    if sepChar in self.POSITION_RECOVERY_SEPARATORS and quantityChar.isdigit():
                        position = qrPosition
                        quantity = int(quantityChar)
                        
                        # Validate: quantity >= position
                        if quantity < position:
                            self._logger.debug(
                                f"[Case 2A] Invalid: {text} (quantity {quantity} < position {position})"
                            )
                            continue
                        
                        recoveredText = f"{position}/{quantity}"
                        self._logger.info(
                            f"[Case 2A] Recovered position/quantity at index {i}: "
                            f"'{text}' → '{recoveredText}' (separator '{sepChar}')"
                        )
                        return (i, position, quantity, confidence * 0.9)
                
                # Case 2B: 2-char format "35" → "3/5" (missing separator)
                elif len(afterPos) == 1 and afterPos.isdigit():
                    position = qrPosition
                    quantity = int(afterPos)
                    
                    # Validate: quantity >= position
                    if quantity < position:
                        self._logger.debug(
                            f"[Case 2B] Invalid: {text} (quantity {quantity} < position {position})"
                        )
                        continue
                    
                    recoveredText = f"{position}/{quantity}"
                    self._logger.info(
                        f"[Case 2B] Recovered position/quantity at index {i}: "
                        f"'{text}' → '{recoveredText}' (missing separator)"
                    )
                    return (i, position, quantity, confidence * 0.85)
        
        # Fallback: Scan all remaining blocks for strict "number/number" format
        self._logger.debug("Position/quantity not found in first pass - trying fallback")
        
        for i, block in enumerate(sortedBlocks):
            text = block.text.strip()
            confidence = block.confidence
            
            match = self.POSITION_QUANTITY_PATTERN.match(text)
            if match:
                position = int(match.group(1))
                quantity = int(match.group(2))
                
                # Validate: quantity >= position
                if quantity < position:
                    self._logger.debug(
                        f"[Fallback] Invalid: {text} (quantity {quantity} < position {position})"
                    )
                    continue
                
                self._logger.info(f"[Fallback] Found position/quantity at index {i}: {text}")
                return (i, position, quantity, confidence * 0.8)
        
        # Not found
        self._logger.warning("Position/quantity NOT found after all attempts")
        return (-1, 0, 0, 0.0)
    
    def _extractProduct(
        self,
        sortedBlocks: List[TextBlock],
        startIndex: int,
        usedIndexes: set,
        result: LabelData
    ) -> int:
        """
        Extract product from text blocks starting at startIndex.
        
        Uses:
        1. Regex pattern validation
        2. Exact match in valid products
        3. Fuzzy match with threshold 0.9
        
        Args:
            sortedBlocks: List of sorted text blocks
            startIndex: Index to start searching from
            usedIndexes: Set of already used indexes
            result: LabelData to update
            
        Returns:
            Index where product was found, or -1 if not found
        """
        for i in range(startIndex, len(sortedBlocks)):
            if i in usedIndexes:
                continue
            
            text = sortedBlocks[i].text.strip()
            confidence = sortedBlocks[i].confidence
            
            # Check regex pattern first
            if not self.PRODUCT_PATTERN.match(text):
                continue
            
            # Try exact match
            if text in self._validProducts or text.upper() in self._validProducts:
                result.productCode = text.upper() if text.upper() in self._validProducts else text
                result.fieldConfidences['productCode'] = confidence
                self._logger.debug(f"[Product] Exact match at index {i}: {text}")
                return i
            
            # Try fuzzy match
            matched, score = FuzzyMatcher.bestMatch(
                text.upper(), 
                self._validProducts, 
                self._minFuzzyScore
            )
            if matched:
                result.productCode = matched
                result.fieldConfidences['productCode'] = confidence * score
                self._logger.debug(
                    f"[Product] Fuzzy match at index {i}: {text} → {matched} (score: {score:.3f})"
                )
                return i
        
        return -1
    
    def _extractSize(
        self,
        sortedBlocks: List[TextBlock],
        startIndex: int,
        usedIndexes: set,
        result: LabelData
    ) -> int:
        """
        Extract size from text blocks starting at startIndex.
        
        Uses:
        1. Regex pattern validation
        2. Exact match in valid sizes
        3. Fuzzy match with threshold 0.9
        
        Args:
            sortedBlocks: List of sorted text blocks
            startIndex: Index to start searching from
            usedIndexes: Set of already used indexes
            result: LabelData to update
            
        Returns:
            Index where size was found, or -1 if not found
        """
        for i in range(startIndex, len(sortedBlocks)):
            if i in usedIndexes:
                continue
            
            text = sortedBlocks[i].text.strip().upper()
            confidence = sortedBlocks[i].confidence
            
            # Check regex pattern first
            if not self.SIZE_PATTERN.match(text):
                continue
            
            # Try exact match
            if text in self._validSizes:
                result.size = text
                result.fieldConfidences['size'] = confidence
                self._logger.debug(f"[Size] Exact match at index {i}: {text}")
                return i
            
            # Try fuzzy match
            matched, score = FuzzyMatcher.bestMatch(
                text, 
                self._validSizes, 
                self._minFuzzyScore
            )
            if matched:
                result.size = matched
                result.fieldConfidences['size'] = confidence * score
                self._logger.debug(
                    f"[Size] Fuzzy match at index {i}: {text} → {matched} (score: {score:.3f})"
                )
                return i
        
        return -1
    
    def _extractColor(
        self,
        sortedBlocks: List[TextBlock],
        startIndex: int,
        usedIndexes: set,
        result: LabelData
    ) -> int:
        """
        Extract color from text blocks starting at startIndex.
        
        Uses:
        1. Regex pattern validation
        2. Exact match in valid colors
        3. Fuzzy match with threshold 0.9
        
        Args:
            sortedBlocks: List of sorted text blocks
            startIndex: Index to start searching from
            usedIndexes: Set of already used indexes
            result: LabelData to update
            
        Returns:
            Index where color was found, or -1 if not found
        """
        for i in range(startIndex, len(sortedBlocks)):
            if i in usedIndexes:
                continue
            
            text = sortedBlocks[i].text.strip().upper()
            confidence = sortedBlocks[i].confidence
            
            # Check regex pattern first (min 3 chars, no digits)
            if not self.COLOR_PATTERN.match(text) or len(text) < 3:
                continue
            
            # Try exact match
            if text in self._validColors:
                result.color = text
                result.fieldConfidences['color'] = confidence
                self._logger.debug(f"[Color] Exact match at index {i}: {text}")
                return i
            
            # Try fuzzy match
            matched, score = FuzzyMatcher.bestMatch(
                text, 
                self._validColors, 
                self._minFuzzyScore
            )
            if matched:
                result.color = matched
                result.fieldConfidences['color'] = confidence * score
                self._logger.debug(
                    f"[Color] Fuzzy match at index {i}: {text} → {matched} (score: {score:.3f})"
                )
                return i
        
        return -1
    
    def _fallbackExtractProduct(
        self,
        sortedBlocks: List[TextBlock],
        usedIndexes: set,
        result: LabelData
    ) -> None:
        """
        Fallback: scan all unused blocks for product.
        
        Args:
            sortedBlocks: List of sorted text blocks
            usedIndexes: Set of already used indexes
            result: LabelData to update
        """
        for i, block in enumerate(sortedBlocks):
            if i in usedIndexes:
                continue
            
            text = block.text.strip()
            confidence = block.confidence
            
            # Check regex pattern first
            if not self.PRODUCT_PATTERN.match(text):
                continue
            
            # Try exact match
            if text in self._validProducts or text.upper() in self._validProducts:
                result.productCode = text.upper() if text.upper() in self._validProducts else text
                result.fieldConfidences['productCode'] = confidence * 0.8
                self._logger.info(f"[Fallback Product] Exact match at index {i}: {text}")
                return
            
            # Try fuzzy match
            matched, score = FuzzyMatcher.bestMatch(
                text.upper(), 
                self._validProducts, 
                self._minFuzzyScore
            )
            if matched:
                result.productCode = matched
                result.fieldConfidences['productCode'] = confidence * score * 0.8
                self._logger.info(
                    f"[Fallback Product] Fuzzy match at index {i}: {text} → {matched} (score: {score:.3f})"
                )
                return
    
    def _fallbackExtractSize(
        self,
        sortedBlocks: List[TextBlock],
        usedIndexes: set,
        result: LabelData
    ) -> None:
        """
        Fallback: scan all unused blocks for size.
        
        Args:
            sortedBlocks: List of sorted text blocks
            usedIndexes: Set of already used indexes
            result: LabelData to update
        """
        for i, block in enumerate(sortedBlocks):
            if i in usedIndexes:
                continue
            
            text = block.text.strip().upper()
            confidence = block.confidence
            
            # Check regex pattern first
            if not self.SIZE_PATTERN.match(text):
                continue
            
            # Try exact match
            if text in self._validSizes:
                result.size = text
                result.fieldConfidences['size'] = confidence * 0.8
                self._logger.info(f"[Fallback Size] Exact match at index {i}: {text}")
                return
            
            # Try fuzzy match
            matched, score = FuzzyMatcher.bestMatch(
                text, 
                self._validSizes, 
                self._minFuzzyScore
            )
            if matched:
                result.size = matched
                result.fieldConfidences['size'] = confidence * score * 0.8
                self._logger.info(
                    f"[Fallback Size] Fuzzy match at index {i}: {text} → {matched} (score: {score:.3f})"
                )
                return
    
    def _fallbackExtractColor(
        self,
        sortedBlocks: List[TextBlock],
        usedIndexes: set,
        result: LabelData
    ) -> None:
        """
        Fallback: scan all unused blocks for color.
        
        Args:
            sortedBlocks: List of sorted text blocks
            usedIndexes: Set of already used indexes
            result: LabelData to update
        """
        for i, block in enumerate(sortedBlocks):
            if i in usedIndexes:
                continue
            
            text = block.text.strip().upper()
            confidence = block.confidence
            
            # Check regex pattern first (min 3 chars, no digits)
            if not self.COLOR_PATTERN.match(text) or len(text) < 3:
                continue
            
            # Try exact match
            if text in self._validColors:
                result.color = text
                result.fieldConfidences['color'] = confidence * 0.8
                self._logger.info(f"[Fallback Color] Exact match at index {i}: {text}")
                return
            
            # Try fuzzy match
            matched, score = FuzzyMatcher.bestMatch(
                text, 
                self._validColors, 
                self._minFuzzyScore
            )
            if matched:
                result.color = matched
                result.fieldConfidences['color'] = confidence * score * 0.8
                self._logger.info(
                    f"[Fallback Color] Fuzzy match at index {i}: {text} → {matched} (score: {score:.3f})"
                )
                return
