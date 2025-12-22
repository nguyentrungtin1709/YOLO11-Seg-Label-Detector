"""
S2 Detection Service Implementation.

Step 2 of the pipeline: YOLO instance segmentation detection.
Creates and manages YOLODetector from core layer.

Follows:
- SRP: Only handles detection operations
- DIP: Depends on IDetector abstraction (interface)
- OCP: Extends without modifying existing code
"""

import time
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.interfaces.detector_interface import IDetector, Detection
from core.detector import createDetector
from services.interfaces.detection_service_interface import (
    IDetectionService,
    DetectionServiceResult
)
from services.interfaces.base_service_interface import BaseService


class S2DetectionService(IDetectionService, BaseService):
    """
    Step 2: Detection Service Implementation.
    
    Runs YOLO instance segmentation on frames and filters results
    based on confidence, area ratio, and top N selection.
    
    Creates YOLODetector internally with provided parameters.
    """
    
    SERVICE_NAME = "s2_detection"
    
    def __init__(
        self,
        backend: str = "onnx",
        modelPath: str = "",
        inputSize: int = 640,
        isSegmentation: bool = True,
        classNames: Optional[List[str]] = None,
        confidenceThreshold: float = 0.5,
        maxAreaRatio: float = 0.40,
        topNDetections: int = 1,
        boxColor: Tuple[int, int, int] = (0, 255, 0),
        textColor: Tuple[int, int, int] = (0, 0, 0),
        lineThickness: int = 2,
        fontSize: float = 0.8,
        maskOpacity: float = 0.4,
        maskColors: Optional[List[Tuple[int, int, int]]] = None,
        debugBasePath: str = "output/debug",
        debugEnabled: bool = False
    ):
        """
        Initialize S2DetectionService.
        
        Args:
            backend: Backend for inference ("onnx" or "openvino").
            modelPath: Path to model file (.onnx for ONNX, .xml for OpenVINO).
            inputSize: Model input size (default: 640).
            isSegmentation: Whether model is segmentation type.
            classNames: List of class names.
            confidenceThreshold: Minimum confidence for detections.
            maxAreaRatio: Maximum area ratio (filter large objects).
            topNDetections: Number of top detections to keep.
            boxColor: Color for bounding boxes (BGR).
            textColor: Color for text labels (BGR).
            lineThickness: Line thickness for boxes.
            fontSize: Font size for labels.
            maskOpacity: Opacity for segmentation masks.
            maskColors: List of colors for masks.
            debugBasePath: Base path for debug output.
            debugEnabled: Whether to save debug output.
        """
        BaseService.__init__(
            self,
            serviceName=self.SERVICE_NAME,
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        
        # Create core detector implementation using factory
        self._detector: IDetector = createDetector(
            backend=backend,
            modelPath=modelPath,
            inputSize=inputSize,
            classNames=classNames or ["label"],
            isSegmentation=isSegmentation
        )
        
        self._modelPath = modelPath
        self._confidenceThreshold = max(0.0, min(1.0, confidenceThreshold))
        self._maxAreaRatio = max(0.0, min(1.0, maxAreaRatio))
        self._topNDetections = max(1, topNDetections)
        self._enabled = False
        self._modelLoaded = False
        
        # Visualization settings
        self._boxColor = boxColor
        self._textColor = textColor
        self._lineThickness = lineThickness
        self._fontSize = fontSize
        self._maskOpacity = maskOpacity
        self._maskColors = maskColors or [
            (128, 0, 128), (0, 128, 255), (255, 0, 0),
            (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 255, 0), (255, 128, 0), (0, 255, 0)
        ]
        
        # Model already loaded by factory if modelPath provided
        
        self._logger.info(
            f"S2DetectionService initialized "
            f"(backend={backend}, inputSize={inputSize}, isSegmentation={isSegmentation})"
        )
    
    def detect(
        self,
        frame: np.ndarray,
        frameId: str
    ) -> DetectionServiceResult:
        """Run detection on a frame."""
        startTime = time.time()
        
        # Check if detection is enabled
        if not self._enabled:
            return DetectionServiceResult(
                detections=[],
                annotatedFrame=None,
                frameId=frameId,
                success=True,
                processingTimeMs=self._measureTime(startTime)
            )
        
        # Check if model is loaded
        if not self._modelLoaded:
            self._logger.warning(f"[{frameId}] Model not loaded")
            return DetectionServiceResult(
                detections=[],
                annotatedFrame=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        try:
            # Run detection
            detections = self._detector.detect(frame, self._confidenceThreshold)
            
            # Filter by area ratio
            imageArea = frame.shape[0] * frame.shape[1]
            filteredDetections = self._filterByArea(detections, imageArea)
            
            # Sort by confidence and take top N
            filteredDetections.sort(key=lambda d: d.confidence, reverse=True)
            topDetections = filteredDetections[:self._topNDetections]
            
            # Create annotated frame
            annotatedFrame = self._createAnnotatedFrame(frame, topDetections)
            
            processingTimeMs = self._measureTime(startTime)
            
            # Save debug output
            self._saveDebugOutput(frameId, topDetections, annotatedFrame, frame, processingTimeMs)
            
            # Log timing
            self._logTiming(frameId, processingTimeMs)
            self._logger.debug(
                f"[{frameId}] Detected {len(topDetections)} objects "
                f"(filtered from {len(detections)})"
            )
            
            return DetectionServiceResult(
                detections=topDetections,
                annotatedFrame=annotatedFrame,
                frameId=frameId,
                success=True,
                processingTimeMs=processingTimeMs
            )
            
        except Exception as e:
            self._logger.error(f"[{frameId}] Detection failed: {e}")
            return DetectionServiceResult(
                detections=[],
                annotatedFrame=None,
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
    
    def loadModel(self, modelPath: str) -> bool:
        """Load the detection model."""
        success = self._detector.loadModel(modelPath)
        
        if success:
            self._modelLoaded = True
            self._logger.info(f"Model loaded: {modelPath}")
        else:
            self._logger.error(f"Failed to load model: {modelPath}")
        
        return success
    
    def setConfidenceThreshold(self, threshold: float) -> None:
        """Set confidence threshold."""
        self._confidenceThreshold = max(0.0, min(1.0, threshold))
        self._logger.debug(f"Confidence threshold: {self._confidenceThreshold}")
    
    def getConfidenceThreshold(self) -> float:
        """Get current confidence threshold."""
        return self._confidenceThreshold
    
    def setEnabled(self, enabled: bool) -> None:
        """Enable or disable detection."""
        self._enabled = enabled
        self._logger.info(f"Detection {'enabled' if enabled else 'disabled'}")
    
    def isEnabled(self) -> bool:
        """Check if detection is enabled."""
        return self._enabled
    
    def isModelLoaded(self) -> bool:
        """Check if model is loaded."""
        return self._modelLoaded
    
    def _filterByArea(
        self, 
        detections: List[Detection], 
        imageArea: int
    ) -> List[Detection]:
        """Filter detections by area ratio."""
        maxArea = imageArea * self._maxAreaRatio
        
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            detArea = (x2 - x1) * (y2 - y1)
            
            if detArea <= maxArea:
                filtered.append(det)
        
        return filtered
    
    def _createAnnotatedFrame(
        self, 
        frame: np.ndarray, 
        detections: List[Detection]
    ) -> np.ndarray:
        """Create frame with detection overlays."""
        annotated = frame.copy()
        
        for i, det in enumerate(detections):
            color = self._maskColors[i % len(self._maskColors)]
            
            # Draw mask if available
            if det.mask is not None:
                mask = det.mask.astype(bool)
                overlay = annotated.copy()
                overlay[mask] = color
                annotated = cv2.addWeighted(
                    overlay, self._maskOpacity,
                    annotated, 1 - self._maskOpacity, 0
                )
            
            # Draw bounding box
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2),
                self._boxColor, self._lineThickness
            )
            
            # Draw label
            label = f"{det.className}: {det.confidence:.2f}"
            labelSize, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,
                self._fontSize, self._lineThickness
            )
            
            # Background for label
            cv2.rectangle(
                annotated,
                (x1, y1 - labelSize[1] - 10),
                (x1 + labelSize[0], y1),
                self._boxColor, -1
            )
            
            # Label text
            cv2.putText(
                annotated, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._fontSize, self._textColor, self._lineThickness
            )
        
        return annotated
    
    def _saveDebugOutput(
        self, 
        frameId: str, 
        detections: List[Detection],
        annotatedFrame: np.ndarray,
        originalFrame: np.ndarray,
        processingTimeMs: float
    ) -> None:
        """Save debug output for detection step."""
        if not self._debugEnabled:
            return
        
        # Save annotated frame
        self._saveDebugImage(frameId, annotatedFrame, "detection")
        
        # Save detection data as JSON
        detectionData = {
            "frameId": frameId,
            "numDetections": len(detections),
            "processingTimeMs": processingTimeMs,
            "detections": [
                {
                    "className": det.className,
                    "confidence": det.confidence,
                    "bbox": list(det.bbox),
                    "hasMask": det.mask is not None
                }
                for det in detections
            ]
        }
        self._saveDebugJson(frameId, detectionData, "detection")
        
        # Save individual cropped images by mask
        for i, det in enumerate(detections):
            if det.mask is not None:
                # Mask is already uint8 with values 0 or 255 from yolo_detector
                maskImage = det.mask.astype(np.uint8)
                self._saveDebugImage(f"{frameId}_{i}", maskImage, "mask")
                
                # Crop original image using mask (keep only masked region)
                mask3ch = cv2.cvtColor(maskImage, cv2.COLOR_GRAY2BGR)
                croppedByMask = cv2.bitwise_and(originalFrame, mask3ch)
                
                # Crop to bounding box for compact output
                x1, y1, x2, y2 = det.bbox
                croppedBbox = croppedByMask[y1:y2, x1:x2]
                self._saveDebugImage(f"{frameId}_{i}", croppedBbox, "cropped")
