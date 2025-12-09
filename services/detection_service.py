"""
Detection Service

Orchestrates label detection using IDetector.
Manages detection state and provides detection results.
"""

import logging
import time
from typing import Optional, Dict

import numpy as np

from core.interfaces.detector_interface import IDetector, Detection
from services.performance_logger import PerformanceLogger


logger = logging.getLogger(__name__)


class DetectionService:
    """
    Service for managing detection operations.
    
    Provides high-level detection control including
    running detection, filtering by size, and selecting top N detections.
    
    Follows:
    - SRP: Only handles detection orchestration
    - DIP: Depends on IDetector abstraction
    """
    
    def __init__(
        self, 
        detector: IDetector,
        maxAreaRatio: float = 0.15,
        topNDetections: int = 3,
        performanceLogger: Optional[PerformanceLogger] = None
    ):
        """
        Initialize DetectionService.
        
        Args:
            detector: Detector implementation.
            maxAreaRatio: Maximum area ratio threshold (0.0-1.0). 
                         Objects larger than this ratio of image area are filtered out.
            topNDetections: Number of top detections to keep (sorted by confidence).
            performanceLogger: Optional performance logger for timing metrics.
        """
        self._detector = detector
        self._isEnabled = False
        self._confidenceThreshold = 0.5
        self._modelLoaded = False
        
        # Filter settings
        self._maxAreaRatio = max(0.0, min(1.0, maxAreaRatio))
        self._topNDetections = max(1, topNDetections)
        
        # Performance logging
        self._performanceLogger = performanceLogger
    
    def loadModel(self, modelPath: str) -> bool:
        """
        Load detection model.
        
        Args:
            modelPath: Path to the ONNX model file.
            
        Returns:
            bool: True if model loaded successfully.
        """
        success = self._detector.loadModel(modelPath)
        if success:
            self._modelLoaded = True
            logger.info(f"Model loaded: {modelPath}")
        else:
            logger.error(f"Failed to load model: {modelPath}")
        
        return success
    
    def setConfidenceThreshold(self, threshold: float) -> None:
        """
        Set confidence threshold for detection.
        
        Args:
            threshold: Confidence threshold (0.0 - 1.0).
        """
        self._confidenceThreshold = max(0.0, min(1.0, threshold))
        logger.debug(f"Confidence threshold set to {self._confidenceThreshold}")
    
    def setEnabled(self, enabled: bool) -> None:
        """
        Enable or disable detection.
        
        Args:
            enabled: True to enable detection.
        """
        self._isEnabled = enabled
        logger.info(f"Detection {'enabled' if enabled else 'disabled'}")
    
    def isEnabled(self) -> bool:
        """
        Check if detection is enabled.
        
        Returns:
            bool: True if detection is enabled.
        """
        return self._isEnabled
    
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run detection on a frame with filtering.
        
        Applies two filters:
        1. Area filter: Removes objects larger than maxAreaRatio of image
        2. Top N filter: Keeps only top N detections by confidence
        
        Returns empty list if detection is disabled or model not loaded.
        
        Args:
            frame: Input frame (BGR format).
            
        Returns:
            List of Detection objects (filtered).
        """
        if not self._isEnabled or not self._modelLoaded:
            return []
        
        # Use timing method if performance logger is available
        if self._performanceLogger is not None:
            detections, timing = self._detector.detectWithTiming(
                frame, self._confidenceThreshold
            )
            
            # Measure filter time
            filterStart = time.perf_counter()
            filteredDetections = self._filterByArea(
                detections, frame.shape[1], frame.shape[0]
            )
            filteredDetections = self._selectTopN(filteredDetections)
            filterTime = (time.perf_counter() - filterStart) * 1000
            
            # Record timing
            timing["filter_ms"] = filterTime
            self._performanceLogger.recordTiming(timing)
        else:
            detections = self._detector.detect(frame, self._confidenceThreshold)
            
            # Apply filters
            filteredDetections = self._filterByArea(
                detections, frame.shape[1], frame.shape[0]
            )
            filteredDetections = self._selectTopN(filteredDetections)
        
        return filteredDetections
    
    def _filterByArea(
        self, 
        detections: list[Detection], 
        imageWidth: int, 
        imageHeight: int
    ) -> list[Detection]:
        """
        Filter out detections larger than maxAreaRatio.
        
        Args:
            detections: List of detections to filter.
            imageWidth: Width of the original image.
            imageHeight: Height of the original image.
            
        Returns:
            Filtered list of detections.
        """
        if self._maxAreaRatio >= 1.0:
            return detections  # No filtering if threshold is 100%
        
        imageArea = imageWidth * imageHeight
        filtered = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            detectionArea = (x2 - x1) * (y2 - y1)
            areaRatio = detectionArea / imageArea
            
            if areaRatio <= self._maxAreaRatio:
                filtered.append(det)
            else:
                logger.debug(
                    f"Filtered out {det.className} (area ratio: {areaRatio:.3f} > {self._maxAreaRatio})"
                )
        
        return filtered
    
    def _selectTopN(self, detections: list[Detection]) -> list[Detection]:
        """
        Select top N detections by confidence score.
        
        Args:
            detections: List of detections to select from.
            
        Returns:
            Top N detections sorted by confidence (descending).
        """
        if len(detections) <= self._topNDetections:
            return detections
        
        # Sort by confidence descending and take top N
        sorted_detections = sorted(
            detections, 
            key=lambda d: d.confidence, 
            reverse=True
        )
        
        return sorted_detections[:self._topNDetections]
    
    @property
    def isModelLoaded(self) -> bool:
        """Check if model is loaded."""
        return self._modelLoaded
    
    @property
    def confidenceThreshold(self) -> float:
        """Get current confidence threshold."""
        return self._confidenceThreshold
    
    @property
    def performanceLogger(self) -> Optional[PerformanceLogger]:
        """Get the performance logger."""
        return self._performanceLogger
