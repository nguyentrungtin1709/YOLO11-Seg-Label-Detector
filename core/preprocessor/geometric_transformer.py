"""
Geometric Transformer Module

Handles geometric transformations for document/label image preprocessing.
Includes cropping and rotation based on polygon mask points.

Follows SRP: Only handles geometric transformation operations.
"""

import logging
from typing import Optional, Tuple
import numpy as np
import cv2


logger = logging.getLogger(__name__)


class GeometricTransformer:
    """
    Handles geometric transformations optimized for speed.
    
    Provides methods for:
    - Ordering polygon points consistently
    - Cropping and rotating images based on minimum area rectangle
    
    Uses warpAffine instead of warpPerspective for better performance.
    
    Follows SRP: Only responsible for geometric transformations.
    """
    
    @staticmethod
    def orderPoints(pts: np.ndarray) -> np.ndarray:
        """
        Orders a set of 4 points in a consistent order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        
        Args:
            pts: A numpy array of shape (4, 2) containing the coordinates of the rectangle corners.
            
        Returns:
            np.ndarray: A numpy array of shape (4, 2) with points ordered as [TL, TR, BR, BL].
            
        Logic:
            1. Calculates the centroid (mean) of the points.
            2. Calculates the angle of each point relative to the centroid using arctan2.
            3. Sorts the points based on these angles.
            4. Identifies the Top-Left point as the one with the minimum sum of (x, y) coordinates.
            5. Rolls the sorted array so that the Top-Left point is first.
        """
        center = np.mean(pts, axis=0)
        
        def getAngle(p):
            return np.arctan2(p[1] - center[1], p[0] - center[0])
        
        sortedPts = sorted(pts, key=getAngle)
        sortedPts = np.array(sortedPts, dtype="float32")
        
        # Find top-left by minimum sum of coordinates
        sums = sortedPts.sum(axis=1)
        topLeftIdx = np.argmin(sums)
        ordered = np.roll(sortedPts, -topLeftIdx, axis=0)
        
        return ordered
    
    @staticmethod
    def applyCropAndRotate(
        image: np.ndarray, 
        points: np.ndarray
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Crops and rotates a region of interest from an image based on polygon points using Warp Affine.
        
        Args:
            image: The source image (BGR format).
            points: A numpy array of polygon points defining the region to crop.
            
        Returns:
            Tuple[Optional[np.ndarray], str]: 
                - The cropped and rotated image (upright rectangle), or None if failed.
                - Status message describing the operation.
                
        Logic:
            1. Computes the minimum area rectangle enclosing the points using cv2.minAreaRect.
            2. Converts the rectangle to 4 corner points (cv2.boxPoints).
            3. Orders the corner points (TL, TR, BR, BL).
            4. Calculates the width and height of the new upright image based on edge lengths.
            5. Defines destination points for the upright rectangle.
            6. Computes the Affine Transform matrix mapping source points to destination points.
            7. Applies the affine warp (cv2.warpAffine) to obtain the final result.
        """
        if points is None or len(points) < 3:
            return None, "Insufficient points (need at least 3)"
        
        try:
            # Ensure points are in correct format
            points = np.array(points, dtype=np.int32)
            if len(points.shape) == 1:
                points = points.reshape(-1, 2)
            
            # Get the minimum enclosing rectangle
            rect = cv2.minAreaRect(points)
            (center, (w, h), angle) = rect
            
            # Get bounding box for logging
            x, y, boundingW, boundingH = cv2.boundingRect(points)
            if boundingW > 0 and boundingH > 0:
                areaRatio = (w * h) / (boundingW * boundingH)
            else:
                areaRatio = 0.0
            
            logger.debug(f"Geometric Transform: Area Ratio = {areaRatio:.4f}")
            
            # Get box points and order them
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            srcPts = GeometricTransformer.orderPoints(box)
            (tl, tr, br, bl) = srcPts
            
            # Calculate dimensions of the destination rectangle
            widthTop = np.linalg.norm(tr - tl)
            widthBottom = np.linalg.norm(br - bl)
            maxWidth = int(max(widthTop, widthBottom))
            
            heightLeft = np.linalg.norm(tl - bl)
            heightRight = np.linalg.norm(tr - br)
            maxHeight = int(max(heightLeft, heightRight))
            
            # Validate dimensions
            if maxWidth <= 0 or maxHeight <= 0:
                return None, "Invalid dimensions after transformation"
            
            # Destination points (Upright Rectangle)
            # We only need 3 points for Affine Transform
            dstPts = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
            ], dtype="float32")
            
            # Compute Affine Transform matrix (uses 3 points)
            M = cv2.getAffineTransform(srcPts[:3], dstPts)
            
            # Apply warpAffine (faster than warpPerspective)
            warped = cv2.warpAffine(
                image, 
                M, 
                (maxWidth, maxHeight),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return warped, f"Success: {maxWidth}x{maxHeight}"
            
        except Exception as e:
            logger.error(f"Error in crop and rotate: {e}")
            return None, f"Error: {str(e)}"
