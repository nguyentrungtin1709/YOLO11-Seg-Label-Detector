"""
Label Component Extractor Implementation.

This module extracts regions of interest from label images based on
QR code position. The label structure is:
- Above QR: Position/Quantity (e.g., "1/1")
- Below QR: Product Code, Size, Color

Follows the Single Responsibility Principle (SRP) from SOLID.
"""

import logging
from typing import Optional, List, Tuple

import cv2
import numpy as np

from core.interfaces.component_extractor_interface import (
    IComponentExtractor, 
    ComponentResult
)


class LabelComponentExtractor(IComponentExtractor):
    """
    Extracts and merges label component regions.
    
    Based on actual label structure:
    ┌─────────────────────────────────────────────────────────────────┐
    │  VA-S-002410-1                              11/19               │
    │  PTFY-API                                    1/1                │ ← ABOVE QR
    │                                            ┌───┐                │
    │                                            │QR │                │
    │                                            └───┘                │
    │  340                                                            │ ← BELOW QR
    │  3T                                                             │ ← BELOW QR
    │  MIDNIGHT                                                       │ ← BELOW QR
    └─────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        aboveQrWidthRatio: float = 0.35,
        aboveQrHeightRatio: float = 0.20,
        belowQrWidthRatio: float = 0.65,
        belowQrHeightRatio: float = 0.45,
        padding: int = 5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize LabelComponentExtractor.
        
        Args:
            aboveQrWidthRatio: Width of above-QR region as ratio of image width
            aboveQrHeightRatio: Height of above-QR region as ratio of image height
            belowQrWidthRatio: Width of below-QR region as ratio of image width
            belowQrHeightRatio: Height of below-QR region as ratio of image height
            padding: Padding around extracted regions
            logger: Logger instance for debug output
        """
        self._aboveQrWidthRatio = aboveQrWidthRatio
        self._aboveQrHeightRatio = aboveQrHeightRatio
        self._belowQrWidthRatio = belowQrWidthRatio
        self._belowQrHeightRatio = belowQrHeightRatio
        self._padding = padding
        self._logger = logger or logging.getLogger(__name__)
    
    def extractAndMerge(
        self, 
        image: np.ndarray, 
        qrPolygon: List[Tuple[int, int]]
    ) -> Optional[ComponentResult]:
        """
        Extract and merge regions of interest from label image.
        
        Args:
            image: Preprocessed label image
            qrPolygon: Four corners of detected QR code
            
        Returns:
            ComponentResult if successful, None otherwise
        """
        try:
            h, w = image.shape[:2]
            
            # Calculate QR code position
            qrCenterX = sum(p[0] for p in qrPolygon) // 4
            qrCenterY = sum(p[1] for p in qrPolygon) // 4
            qrTop = min(p[1] for p in qrPolygon)
            qrBottom = max(p[1] for p in qrPolygon)
            qrRight = max(p[0] for p in qrPolygon)
            
            self._logger.debug(
                f"QR position: center=({qrCenterX}, {qrCenterY}), "
                f"top={qrTop}, bottom={qrBottom}, right={qrRight}"
            )
            
            # Extract region above QR (position/quantity: "1/1")
            aboveQrRoi = self._extractAboveQr(image, qrTop, qrRight, w, h)
            
            # Extract region below QR (product, size, color)
            belowQrRoi = self._extractBelowQr(image, qrBottom, w, h)
            
            # Merge two regions into one image (above on top, below at bottom)
            mergedImage = self._mergeComponents(aboveQrRoi, belowQrRoi)
            
            self._logger.debug(
                f"Extracted regions: above={aboveQrRoi.shape}, "
                f"below={belowQrRoi.shape}, merged={mergedImage.shape}"
            )
            
            return ComponentResult(
                mergedImage=mergedImage,
                aboveQrRoi=aboveQrRoi,
                belowQrRoi=belowQrRoi,
                qrPolygon=qrPolygon
            )
            
        except Exception as e:
            self._logger.error(f"Error extracting components: {e}")
            return None
    
    def _extractAboveQr(
        self, 
        image: np.ndarray, 
        qrTop: int, 
        qrRight: int,
        imgWidth: int, 
        imgHeight: int
    ) -> np.ndarray:
        """
        Extract region above QR code, centered horizontally.
        
        This region contains position/quantity (e.g., "1/1").
        Centered around QR right edge with padding on both sides.
        """
        roiHeight = int(imgHeight * self._aboveQrHeightRatio)
        roiWidth = int(imgWidth * self._aboveQrWidthRatio)
        
        # Center horizontally around QR right edge
        centerX = qrRight
        halfWidth = roiWidth // 2
        
        # Region: centered around QR right, above QR top
        y1 = max(0, qrTop - roiHeight)
        y2 = qrTop
        x1 = max(0, centerX - halfWidth)  # Left side with padding
        x2 = min(imgWidth, centerX + halfWidth)  # Right side with padding
        
        # Ensure valid region
        if y1 >= y2 or x1 >= x2:
            self._logger.warning("Invalid above-QR region, using fallback")
            y1 = max(0, qrTop - 40)
            y2 = qrTop
            x1 = max(0, qrRight - 60)
            x2 = min(imgWidth, qrRight + 60)
        
        roi = image[y1:y2, x1:x2]
        
        # Return empty image if extraction failed
        if roi.size == 0:
            self._logger.warning("Empty above-QR region, creating placeholder")
            return np.zeros((40, 120, 3), dtype=np.uint8)
        
        return roi.copy()
    
    def _extractBelowQr(
        self, 
        image: np.ndarray, 
        qrBottom: int, 
        imgWidth: int, 
        imgHeight: int
    ) -> np.ndarray:
        """
        Extract region below QR code.
        
        This region contains product code, size, and color.
        """
        roiHeight = int(imgHeight * self._belowQrHeightRatio)
        roiWidth = int(imgWidth * self._belowQrWidthRatio)
        
        # Region below QR, from left side
        y1 = qrBottom + self._padding
        y2 = min(imgHeight, qrBottom + roiHeight)
        x1 = 0
        x2 = roiWidth
        
        # Ensure valid region
        if y1 >= y2 or x1 >= x2:
            self._logger.warning("Invalid below-QR region, using fallback")
            y1 = min(imgHeight - 1, qrBottom + 5)
            y2 = imgHeight
            x1 = 0
            x2 = imgWidth // 2
        
        roi = image[y1:y2, x1:x2]
        
        # Return empty image if extraction failed
        if roi.size == 0:
            self._logger.warning("Empty below-QR region, creating placeholder")
            return np.zeros((100, 200, 3), dtype=np.uint8)
        
        return roi.copy()
    
    def _mergeComponents(
        self, 
        aboveQrRoi: np.ndarray, 
        belowQrRoi: np.ndarray
    ) -> np.ndarray:
        """
        Merge above-QR and below-QR regions into a single image.
        
        Regions are stacked vertically with above-QR on top.
        Above-QR region is centered horizontally relative to below-QR region.
        """
        # Convert to same number of channels if needed
        if len(aboveQrRoi.shape) == 2:
            aboveQrRoi = cv2.cvtColor(aboveQrRoi, cv2.COLOR_GRAY2BGR)
        if len(belowQrRoi.shape) == 2:
            belowQrRoi = cv2.cvtColor(belowQrRoi, cv2.COLOR_GRAY2BGR)
        
        # Use max width for canvas
        targetWidth = max(aboveQrRoi.shape[1], belowQrRoi.shape[1])
        
        def centerPadToWidth(img: np.ndarray, width: int) -> np.ndarray:
            """Center image horizontally with white padding on both sides."""
            if img.shape[1] >= width:
                return img
            totalPad = width - img.shape[1]
            leftPad = totalPad // 2
            rightPad = totalPad - leftPad
            leftPadding = np.ones((img.shape[0], leftPad, 3), dtype=np.uint8) * 255
            rightPadding = np.ones((img.shape[0], rightPad, 3), dtype=np.uint8) * 255
            return np.hstack([leftPadding, img, rightPadding])
        
        def leftPadToWidth(img: np.ndarray, width: int) -> np.ndarray:
            """Left-align image with white padding on right."""
            if img.shape[1] >= width:
                return img
            padWidth = width - img.shape[1]
            padding = np.ones((img.shape[0], padWidth, 3), dtype=np.uint8) * 255
            return np.hstack([img, padding])
        
        # Center above-QR, left-align below-QR
        aboveQrRoi = centerPadToWidth(aboveQrRoi, targetWidth)
        belowQrRoi = leftPadToWidth(belowQrRoi, targetWidth)
        
        # Add separator line
        separator = np.ones((3, targetWidth, 3), dtype=np.uint8) * 128
        
        # Stack vertically
        merged = np.vstack([aboveQrRoi, separator, belowQrRoi])
        
        return merged
