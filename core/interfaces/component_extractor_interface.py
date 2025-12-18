"""
Component Extractor Interface Module.

This module defines the interface and data classes for extracting
label components (regions of interest) based on QR code position.
Follows the Interface Segregation Principle (ISP) from SOLID.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class ComponentResult:
    """
    Result of component extraction.
    
    Attributes:
        mergedImage: Combined image of all extracted regions
        aboveQrRoi: Region above QR code (position/quantity text)
        belowQrRoi: Region below QR code (product, size, color)
        qrPolygon: QR code position in original image
    """
    mergedImage: np.ndarray
    aboveQrRoi: np.ndarray
    belowQrRoi: np.ndarray
    qrPolygon: List[Tuple[int, int]]


class IComponentExtractor(ABC):
    """
    Interface for label component extractor.
    
    Implementations should extract regions of interest from label images
    based on QR code position for OCR processing.
    """
    
    @abstractmethod
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
        pass
