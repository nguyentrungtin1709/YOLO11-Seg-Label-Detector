"""
Local Image Writer Implementation

Implements IImageWriter for saving images to local filesystem.
Follows SRP: Only handles image saving operations.
"""

import os
import logging
import numpy as np
import cv2

from core.interfaces.writer_interface import IImageWriter


logger = logging.getLogger(__name__)


class LocalImageWriter(IImageWriter):
    """
    Image writer implementation for local filesystem.
    
    Saves images to the local filesystem using OpenCV.
    Automatically creates directories if they don't exist.
    
    Follows SRP: Only responsible for saving images locally.
    """
    
    def __init__(self, quality: int = 95):
        """
        Initialize LocalImageWriter.
        
        Args:
            quality: JPEG quality (0-100) for saving images.
        """
        self._quality = quality
    
    def save(self, image: np.ndarray, filepath: str) -> bool:
        """
        Save an image to the specified filepath.
        
        Creates parent directories if they don't exist.
        
        Args:
            image: Image as numpy array (BGR format).
            filepath: Destination path for the image file.
            
        Returns:
            bool: True if image saved successfully.
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            # Set encoding parameters based on file extension
            ext = os.path.splitext(filepath)[1].lower()
            
            if ext in ['.jpg', '.jpeg']:
                params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]
            elif ext == '.png':
                params = [cv2.IMWRITE_PNG_COMPRESSION, 3]  # 0-9, lower is faster
            else:
                params = []
            
            # Save the image
            success = cv2.imwrite(filepath, image, params)
            
            if success:
                logger.debug(f"Image saved to {filepath}")
            else:
                logger.error(f"Failed to save image to {filepath}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving image to {filepath}: {e}")
            return False
