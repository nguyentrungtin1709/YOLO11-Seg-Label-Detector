"""
Orientation Corrector Module

Handles orientation correction for document/label images.
Includes landscape forcing and AI-based 180-degree rotation fix.

Follows SRP: Only handles orientation correction operations.
"""

import logging
import os
from typing import Optional, Tuple
import numpy as np
import cv2


logger = logging.getLogger(__name__)


# Check and import PaddleOCR library
try:
    from paddleocr import DocImgOrientationClassification
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning(
        "PaddleOCR not installed. AI orientation correction will be disabled. "
        "Run: pip install paddleocr paddlepaddle --upgrade"
    )


class OrientationCorrector:
    """
    Handles orientation correction for document/label images.
    
    Provides methods for:
    - Forcing landscape orientation (width >= height)
    - AI-based 180-degree rotation detection and correction
    
    Uses PaddleOCR's DocImgOrientationClassification for AI detection.
    
    Follows SRP: Only responsible for orientation correction.
    """
    
    def __init__(
        self, 
        aiConfidenceThreshold: float = 0.6,
        modelPath: Optional[str] = None,
        cpuThreads: int = 4,
        enableMkldnn: bool = True
    ):
        """
        Initialize OrientationCorrector.
        
        Args:
            aiConfidenceThreshold: Minimum confidence score to apply AI rotation fix.
            modelPath: Path to local model directory. If None, uses default download.
            cpuThreads: Number of CPU threads for orientation classification (default: 4).
            enableMkldnn: Enable MKL-DNN acceleration (default: True).
        """
        self._aiConfidenceThreshold = aiConfidenceThreshold
        self._angleClassifier = None
        self._aiAvailable = False
        self._cpuThreads = cpuThreads
        self._enableMkldnn = enableMkldnn
        
        if PADDLE_AVAILABLE:
            try:
                # Initialize the orientation classifier with CPU optimization
                if modelPath and os.path.exists(modelPath):
                    # Load from local model directory
                    self._angleClassifier = DocImgOrientationClassification(
                        model_dir=modelPath,
                        cpu_threads=cpuThreads,
                        enable_mkldnn=enableMkldnn
                    )
                    logger.info(f"PaddleOCR classifier loaded from local: {modelPath} (threads={cpuThreads}, mkldnn={enableMkldnn})")
                else:
                    # Fallback to downloading model
                    self._angleClassifier = DocImgOrientationClassification(
                        model_name="PP-LCNet_x1_0_doc_ori",
                        cpu_threads=cpuThreads,
                        enable_mkldnn=enableMkldnn
                    )
                    logger.info(f"PaddleOCR classifier loaded from cache/download (threads={cpuThreads}, mkldnn={enableMkldnn})")
                
                self._aiAvailable = True
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR classifier: {e}")
                self._aiAvailable = False
    
    @property
    def isAiAvailable(self) -> bool:
        """Check if AI orientation correction is available."""
        return self._aiAvailable
    
    @staticmethod
    def forceLandscape(image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Ensures the image is in landscape orientation (width >= height).
        
        Args:
            image: The input image (BGR format).
            
        Returns:
            Tuple[np.ndarray, bool]:
                - The image, rotated 90 degrees clockwise if it was portrait.
                - True if rotation was applied, False otherwise.
                
        Logic:
            1. Checks the dimensions (height, width) of the image.
            2. If height > width, rotates the image 90 degrees clockwise.
            3. Otherwise, returns the image unchanged.
        """
        h, w = image.shape[:2]
        if h > w:
            logger.debug("Force Landscape: Rotated 90 degrees clockwise")
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), True
        return image, False
    
    def correctOrientationWithAi(
        self, 
        image: np.ndarray,
        maxWidth: int = 1000
    ) -> Tuple[np.ndarray, str]:
        """
        Corrects the orientation of the image (specifically 180-degree rotation) using AI classifier.
        
        Args:
            image: The input image (BGR format).
            maxWidth: Maximum width for resized image during classification (for speed).
            
        Returns:
            Tuple[np.ndarray, str]:
                - The orientation-corrected image.
                - Status message describing the action taken.
                
        Logic:
            1. Checks if the PaddleOCR classifier is initialized.
            2. Resizes the image to maxWidth for faster inference (preserving aspect ratio).
            3. Runs the classifier to predict the orientation label and confidence score.
            4. If the label indicates '180' degrees and confidence > threshold:
               - Rotates the image 180 degrees.
            5. Returns the (potentially rotated) image and status.
        """
        if not self._aiAvailable or self._angleClassifier is None:
            return image, "AI not available"
        
        try:
            # Resize for faster inference if needed
            h, w = image.shape[:2]
            if w > maxWidth:
                scale = maxWidth / float(w)
                checkImg = cv2.resize(
                    image, 
                    (0, 0), 
                    fx=scale, 
                    fy=scale, 
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                checkImg = image
            
            # Run classification
            results = self._angleClassifier.predict(checkImg)
            
            if not results:
                return image, "No classification result"
            
            res = results[0]
            label = ""
            score = 0.0
            
            # Handle different result formats
            if isinstance(res, dict) and 'label_names' in res:
                label = res['label_names'][0]
                score = res['scores'][0]
            elif hasattr(res, 'label_names'):
                label = res.label_names[0]
                score = res.scores[0]
            
            if not label:
                return image, "No label detected"
            
            logger.debug(f"AI Orientation: Detected {label} (score: {score:.4f})")
            
            # Check if 180-degree rotation is needed
            if '180' in str(label) and score > self._aiConfidenceThreshold:
                logger.info(f"AI Orientation: Rotating 180 degrees (confidence: {score:.4f})")
                return cv2.rotate(image, cv2.ROTATE_180), f"Rotated 180° (conf: {score:.2f})"
            
            return image, f"No rotation needed (detected: {label}, conf: {score:.2f})"
            
        except Exception as e:
            logger.error(f"Error in AI orientation correction: {e}")
            return image, f"Error: {str(e)}"
