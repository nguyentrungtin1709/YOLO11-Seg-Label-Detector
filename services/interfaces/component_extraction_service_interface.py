"""
Component Extraction Service Interface Module.

Defines the interface for component extraction operations (Step 6 of the pipeline).
Responsible for extracting text regions and merging them into a single image.

Follows:
- SRP: Only handles component extraction operations
- DIP: Depends on IComponentExtractor abstraction from core layer
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from core.interfaces.component_extractor_interface import ComponentResult


@dataclass
class ComponentExtractionServiceResult:
    """
    Result of the component extraction service.
    
    Attributes:
        componentData: Component extraction result with ROIs.
        mergedImage: Merged image of all text regions.
        frameId: Frame identifier for debug output.
        success: Whether extraction was successful.
        processingTimeMs: Time taken for extraction.
    """
    componentData: Optional[ComponentResult]
    mergedImage: Optional[np.ndarray]
    frameId: str
    success: bool
    processingTimeMs: float = 0.0


class IComponentExtractionService(ABC):
    """
    Interface for component extraction operations (Step 6).
    
    Extracts text regions based on QR code position and merges them
    into a single image for efficient OCR processing.
    """
    
    @abstractmethod
    def extractComponents(
        self,
        image: np.ndarray,
        qrPolygon: List[List[int]],
        frameId: str
    ) -> ComponentExtractionServiceResult:
        """
        Extract and merge text components from an image.
        
        Uses QR code position to locate text regions above and below.
        Merges extracted regions into a single image.
        
        Args:
            image: Input image (BGR format).
            qrPolygon: QR code polygon coordinates.
            frameId: Frame identifier for debug output.
            
        Returns:
            ComponentExtractionServiceResult: Extraction result with metadata.
        """
        pass
    
    @abstractmethod
    def setEnabled(self, enabled: bool) -> None:
        """
        Enable or disable component extraction.
        
        Args:
            enabled: True to enable component extraction.
        """
        pass
    
    @abstractmethod
    def isEnabled(self) -> bool:
        """
        Check if component extraction is enabled.
        
        Returns:
            bool: True if component extraction is enabled.
        """
        pass
