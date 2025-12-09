"""
Image Saver Service

Orchestrates image saving with optional annotation overlay.
Manages output directory and filename generation.
"""

import os
import logging
from datetime import datetime
from typing import Optional
import numpy as np
import cv2

from core.interfaces.writer_interface import IImageWriter
from core.interfaces.detector_interface import Detection


logger = logging.getLogger(__name__)


class ImageSaverService:
    """
    Service for saving captured images.
    
    Handles saving raw frames and annotated frames with detection overlays.
    Supports segmentation mask rendering with configurable opacity.
    
    Follows:
    - SRP: Only handles image saving orchestration
    - DIP: Depends on IImageWriter abstraction
    """
    
    def __init__(
        self,
        imageWriter: IImageWriter,
        captureDirectory: str = "output/captures",
        debugDirectory: str = "output/debug",
        boxColor: tuple[int, int, int] = (0, 255, 0),
        textColor: tuple[int, int, int] = (255, 255, 255),
        lineThickness: int = 2,
        fontSize: float = 0.6,
        maskOpacity: float = 0.4,
        maskColors: list[tuple[int, int, int]] = None
    ):
        """
        Initialize ImageSaverService.
        
        Args:
            imageWriter: Image writer implementation.
            captureDirectory: Directory to save raw captured images.
            debugDirectory: Directory to save debug images with annotations.
            boxColor: Color for detection boxes (BGR).
            textColor: Color for text labels (BGR).
            lineThickness: Thickness of detection boxes.
            fontSize: Font scale for labels.
            maskOpacity: Opacity for segmentation masks (0.0-1.0, default 0.4).
            maskColors: List of colors for masks (BGR format).
        """
        self._imageWriter = imageWriter
        self._captureDirectory = captureDirectory
        self._debugDirectory = debugDirectory
        self._displayDirectory = os.path.join(debugDirectory, "display")
        self._originalDirectory = os.path.join(debugDirectory, "original")
        self._bboxCropDirectory = os.path.join(debugDirectory, "bbox")
        self._maskCropDirectory = os.path.join(debugDirectory, "mask")
        self._txtDirectory = os.path.join(debugDirectory, "txt")
        self._boxColor = boxColor
        self._textColor = textColor
        self._lineThickness = lineThickness
        self._fontSize = fontSize
        self._maskOpacity = max(0.0, min(1.0, maskOpacity))  # Clamp to [0, 1]
        
        # Use provided mask colors or default palette (BGR format)
        if maskColors:
            self._maskColors = [tuple(c) for c in maskColors]
        else:
            self._maskColors = [
                (128, 0, 128),   # Purple
                (0, 128, 255),   # Orange
                (255, 0, 0),     # Blue
                (0, 0, 255),     # Red
                (255, 255, 0),   # Cyan
                (255, 0, 255),   # Magenta
                (0, 255, 255),   # Yellow
                (128, 255, 0),   # Light green
                (255, 128, 0),   # Light blue
                (0, 255, 0),     # Green
            ]
    
    def saveRawCapture(self, frame: np.ndarray) -> Optional[str]:
        """
        Save a raw frame without any annotations.
        Used when user clicks Capture button.
        
        Args:
            frame: Image to save (BGR format).
            
        Returns:
            Saved file path, or None if failed.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(self._captureDirectory, filename)
        
        success = self._imageWriter.save(frame.copy(), filepath)
        
        if success:
            logger.info(f"Raw capture saved: {filepath}")
            return filepath
        else:
            logger.error(f"Failed to save raw capture: {filepath}")
            return None
    
    def saveDebugFrame(
        self,
        frame: np.ndarray,
        detections: list[Detection]
    ) -> Optional[str]:
        """
        Save a frame with detection annotations.
        Also saves cropped images by bbox and mask for each detection.
        Used in debug mode when detections are found.
        
        Args:
            frame: Image to save (BGR format).
            detections: List of detections to overlay.
            
        Returns:
            Saved file path, or None if failed.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Save original frame first
        originalFilename = f"frame_{timestamp}.png"
        originalFilepath = os.path.join(self._originalDirectory, originalFilename)
        self._imageWriter.save(frame.copy(), originalFilepath)
        logger.debug(f"Original frame saved: {originalFilepath}")
        
        # Save annotated debug frame to display directory
        displayFilename = f"debug_{timestamp}.jpg"
        displayFilepath = os.path.join(self._displayDirectory, displayFilename)
        
        # Draw annotations on copy
        imageToSave = self._drawAnnotations(frame.copy(), detections)
        
        success = self._imageWriter.save(imageToSave, displayFilepath)
        
        if success:
            logger.info(f"Debug frame saved: {displayFilepath}")
            
            # Save cropped images for each detection
            self._saveCroppedImages(frame, detections, timestamp)
            
            # Save mask coordinates to txt files
            self._saveMaskCoordinates(detections, timestamp)
            
            return displayFilepath
        else:
            logger.error(f"Failed to save debug frame: {displayFilepath}")
            return None
    
    def _saveCroppedImages(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        timestamp: str
    ) -> None:
        """
        Save cropped images by bounding box and mask for each detection.
        
        Args:
            frame: Original frame (BGR format).
            detections: List of detections.
            timestamp: Timestamp string for filename.
        """
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure valid bbox coordinates
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Save bbox cropped image
            bboxCrop = frame[y1:y2, x1:x2].copy()
            bboxFilename = f"bbox_{timestamp}_{idx}.jpg"
            bboxFilepath = os.path.join(self._bboxCropDirectory, bboxFilename)
            
            if self._imageWriter.save(bboxCrop, bboxFilepath):
                logger.debug(f"Bbox crop saved: {bboxFilepath}")
            
            # Save mask cropped image (if mask available)
            if det.mask is not None:
                maskCrop = self._cropByMask(frame, det.mask, x1, y1, x2, y2)
                if maskCrop is not None:
                    maskFilename = f"mask_{timestamp}_{idx}.png"
                    maskFilepath = os.path.join(self._maskCropDirectory, maskFilename)
                    
                    if self._imageWriter.save(maskCrop, maskFilepath):
                        logger.debug(f"Mask crop saved: {maskFilepath}")
    
    def _saveMaskCoordinates(
        self,
        detections: list[Detection],
        timestamp: str
    ) -> None:
        """
        Save mask coordinates to text files for each detection.
        
        Format: Each line contains x,y coordinates of mask contour points.
        First line is metadata (class, confidence, bbox).
        
        Args:
            detections: List of detections.
            timestamp: Timestamp string for filename.
        """
        for idx, det in enumerate(detections):
            if det.mask is None:
                continue
            
            txtFilename = f"mask_{timestamp}_{idx}.txt"
            txtFilepath = os.path.join(self._txtDirectory, txtFilename)
            
            try:
                # Ensure directory exists
                os.makedirs(self._txtDirectory, exist_ok=True)
                
                with open(txtFilepath, 'w') as f:
                    # Write metadata header
                    x1, y1, x2, y2 = det.bbox
                    f.write(f"# Class: {det.className}\n")
                    f.write(f"# Confidence: {det.confidence:.4f}\n")
                    f.write(f"# BBox: {x1},{y1},{x2},{y2}\n")
                    f.write(f"# Format: x,y coordinates of contour points\n")
                    f.write(f"#\n")
                    
                    # Find contours from mask
                    contours, _ = cv2.findContours(
                        det.mask, 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    # Write contour points
                    for contourIdx, contour in enumerate(contours):
                        f.write(f"# Contour {contourIdx} ({len(contour)} points)\n")
                        for point in contour:
                            x, y = point[0]
                            f.write(f"{x},{y}\n")
                        f.write(f"#\n")  # Separator between contours
                
                logger.debug(f"Mask coordinates saved: {txtFilepath}")
                
            except Exception as e:
                logger.error(f"Failed to save mask coordinates: {e}")
    
    def _cropByMask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        x1: int, y1: int, x2: int, y2: int
    ) -> Optional[np.ndarray]:
        """
        Crop image by mask within bounding box, with transparent background.
        
        Args:
            frame: Original frame (BGR format).
            mask: Binary mask (H x W) with values 0 or 255.
            x1, y1, x2, y2: Bounding box coordinates.
            
        Returns:
            Cropped image with alpha channel (BGRA), or None if failed.
        """
        try:
            h, w = frame.shape[:2]
            
            # Create BGRA image (with alpha channel)
            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            
            # Apply mask to alpha channel
            maskBool = mask > 127
            bgra[:, :, 3] = 0  # Start with fully transparent
            bgra[maskBool, 3] = 255  # Set opaque where mask is active
            
            # Crop to bounding box
            croppedBgra = bgra[y1:y2, x1:x2].copy()
            
            return croppedBgra
            
        except Exception as e:
            logger.error(f"Failed to crop by mask: {e}")
            return None
    
    def _drawAnnotations(
        self,
        image: np.ndarray,
        detections: list[Detection]
    ) -> np.ndarray:
        """
        Draw detection annotations on an image.
        Draws masks first (if available), then bounding boxes and labels on top.
        
        Args:
            image: Image to annotate (will be modified).
            detections: List of detections to draw.
            
        Returns:
            Annotated image.
        """
        # First pass: Draw all masks (so they appear behind boxes and labels)
        for idx, det in enumerate(detections):
            if det.mask is not None:
                image = self._drawMask(image, det.mask, idx)
        
        # Second pass: Draw bounding boxes and labels
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                self._boxColor,
                self._lineThickness
            )
            
            # Prepare label text
            label = f"{det.className}: {det.confidence:.2f}"
            
            # Calculate text size for background
            (textWidth, textHeight), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self._fontSize,
                1
            )
            
            # Draw text background
            cv2.rectangle(
                image,
                (x1, y1 - textHeight - 10),
                (x1 + textWidth + 10, y1),
                self._boxColor,
                -1
            )
            
            # Draw text label
            cv2.putText(
                image,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._fontSize,
                self._textColor,
                1,
                cv2.LINE_AA
            )
        
        return image
    
    def _drawMask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        colorIndex: int = 0
    ) -> np.ndarray:
        """
        Draw a segmentation mask on an image with transparency.
        
        Uses cv2.addWeighted for blending with configurable opacity.
        
        Args:
            image: Image to draw on (will be modified).
            mask: Binary mask (H x W) with values 0 or 255.
            colorIndex: Index into mask colors palette.
            
        Returns:
            Image with mask overlay.
        """
        if mask is None:
            return image
        
        try:
            # Get color for this mask
            color = self._maskColors[colorIndex % len(self._maskColors)]
            
            # Create colored mask overlay
            coloredMask = np.zeros_like(image, dtype=np.uint8)
            
            # Apply color where mask is active
            maskBool = mask > 127  # Binary threshold
            coloredMask[maskBool] = color
            
            # Proper alpha blending: result = image * (1 - alpha) + coloredMask * alpha
            # This gives true transparency effect
            blended = cv2.addWeighted(
                image, 
                1.0 - self._maskOpacity,  # Background weight
                coloredMask, 
                self._maskOpacity,         # Mask weight
                0
            )
            
            # Only apply blending where mask is active
            image[maskBool] = blended[maskBool]
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to draw mask: {e}")
            return image
    
    @property
    def captureDirectory(self) -> str:
        """Get capture directory."""
        return self._captureDirectory
    
    @property
    def debugDirectory(self) -> str:
        """Get debug directory."""
        return self._debugDirectory
