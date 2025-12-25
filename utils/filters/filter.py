"""
S3b Quality Filter Service Implementation.

Step 3b of the pipeline: Quality filtering before enhancement.
Filters images based on 4 criteria (must pass ALL):

============================================================================
PARAMETERS (Thông số lọc):
============================================================================
1. Size (Kích thước):
   - minWidth: 270px  (chiều rộng tối thiểu)
   - minHeight: 180px (chiều cao tối thiểu)
   
2. Contrast (Độ tương phản):
   - minContrast: 30 (std deviation của grayscale)
   - Dùng để kiểm tra ảnh có đủ chi tiết không
   
3. Sharpness (Độ sắc nét):
   - minSharpness: 300 (Laplacian variance)
   - Dùng để kiểm tra ảnh có bị mờ không
   
4. Brightness (Độ sáng):
   - minBrightness: 60 (mean pixel value)
   - maxBrightness: 240
   - Dùng để loại ảnh quá tối hoặc quá sáng

============================================================================
TIMING (Hiệu suất):
============================================================================
Đo trên 100 ảnh (kích thước trung bình ~300x200):
- Size check:        ~0.001 ms (0.3%)
- Grayscale convert: ~0.038 ms (7.7%)
- Contrast (std):    ~0.136 ms (27.7%)
- Brightness (mean): ~0.037 ms (7.4%)
- Sharpness (Laplacian): ~0.280 ms (56.9%)
- TOTAL:             ~0.5 ms/ảnh

============================================================================
USAGE (Cách sử dụng):
============================================================================
Config file: config/application_config.json

{
    "s3b_quality_filter": {
        "enabled": true,        // Bật/tắt bộ lọc
        "minWidth": 270,
        "minHeight": 180,
        "minContrast": 30,
        "minSharpness": 300,
        "minBrightness": 60,
        "maxBrightness": 240
    }
}

============================================================================
PIPELINE:
============================================================================
- Ảnh PASS: Bỏ qua s4 enhancement → Đi thẳng s5-s8 (QR+OCR)
- Ảnh FAIL: Dừng, không xử lý tiếp

Only images that pass ALL criteria will continue to s5-s8.
Images that fail are skipped from further processing.

Follows:
- SRP: Only handles quality filtering
- DIP: Implements IQualityFilterService interface
"""

import time
import logging
from typing import Optional

import numpy as np
import cv2

from services.interfaces.quality_filter_service_interface import (
    IQualityFilterService,
    QualityFilterResult,
    QualityFilterServiceResult
)
from services.interfaces.base_service_interface import BaseService


class S3bQualityFilterService(IQualityFilterService, BaseService):
    """
    Step 3b: Quality Filter Service Implementation.
    
    Filters images based on quality criteria.
    Only high-quality images proceed to the rest of the pipeline.
    """
    
    SERVICE_NAME = "s3b_quality_filter"
    
    def __init__(
        self,
        enabled: bool = False,
        minWidth: int = 270,
        minHeight: int = 180,
        minContrast: float = 50.0,
        minSharpness: float = 300.0,
        minBrightness: float = 60.0,
        maxBrightness: float = 240.0,
        debugBasePath: str = "output/debug",
        debugEnabled: bool = False
    ):
        """
        Initialize S3bQualityFilterService.
        
        Args:
            enabled: Whether quality filter is enabled.
            minWidth: Minimum image width.
            minHeight: Minimum image height.
            minContrast: Minimum contrast (std deviation).
            minSharpness: Minimum sharpness (Laplacian variance).
            minBrightness: Minimum brightness (mean).
            maxBrightness: Maximum brightness (mean).
            debugBasePath: Base path for debug output.
            debugEnabled: Whether to save debug output.
        """
        BaseService.__init__(
            self,
            serviceName=self.SERVICE_NAME,
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        
        self._enabled = enabled
        self._minWidth = minWidth
        self._minHeight = minHeight
        self._minContrast = minContrast
        self._minSharpness = minSharpness
        self._minBrightness = minBrightness
        self._maxBrightness = maxBrightness
        
        self._logger.info(
            f"S3bQualityFilterService initialized "
            f"(enabled={enabled}, size>={minWidth}x{minHeight}, "
            f"contrast>={minContrast}, sharpness>={minSharpness}, "
            f"brightness={minBrightness}-{maxBrightness})"
        )
    
    def _calculateContrast(self, gray: np.ndarray) -> float:
        """Calculate contrast using standard deviation."""
        return float(np.std(gray))
    
    def _calculateSharpness(self, gray: np.ndarray) -> float:
        """Calculate sharpness using Laplacian variance."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    def _calculateBrightness(self, gray: np.ndarray) -> float:
        """Calculate brightness using mean pixel value."""
        return float(np.mean(gray))
    
    def filter(
        self,
        image: np.ndarray,
        frameId: str
    ) -> QualityFilterServiceResult:
        """Filter image based on quality criteria."""
        startTime = time.time()
        
        # If disabled, pass all images
        if not self._enabled:
            return QualityFilterServiceResult(
                filterResult=QualityFilterResult(
                    passed=True,
                    width=0,
                    height=0,
                    contrast=0.0,
                    sharpness=0.0,
                    brightness=0.0,
                    failReason=None
                ),
                frameId=frameId,
                success=True,
                processingTimeMs=self._measureTime(startTime)
            )
        
        if image is None:
            self._logger.warning(f"[{frameId}] No image provided for quality filter")
            return QualityFilterServiceResult(
                filterResult=QualityFilterResult(
                    passed=False,
                    width=0,
                    height=0,
                    contrast=0.0,
                    sharpness=0.0,
                    brightness=0.0,
                    failReason="No image provided"
                ),
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
        
        try:
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            contrast = self._calculateContrast(gray)
            sharpness = self._calculateSharpness(gray)
            brightness = self._calculateBrightness(gray)
            
            # Check criteria
            failReason = None
            passed = True
            
            # Check 1: Size
            if width < self._minWidth or height < self._minHeight:
                passed = False
                failReason = f"Size too small ({width}x{height} < {self._minWidth}x{self._minHeight})"
            
            # Check 2: Contrast
            elif contrast < self._minContrast:
                passed = False
                failReason = f"Contrast too low ({contrast:.1f} < {self._minContrast})"
            
            # Check 3: Sharpness
            elif sharpness < self._minSharpness:
                passed = False
                failReason = f"Image blurry ({sharpness:.0f} < {self._minSharpness})"
            
            # Check 4: Brightness
            elif brightness < self._minBrightness:
                passed = False
                failReason = f"Image too dark ({brightness:.0f} < {self._minBrightness})"
            elif brightness > self._maxBrightness:
                passed = False
                failReason = f"Image too bright ({brightness:.0f} > {self._maxBrightness})"
            
            processingTimeMs = self._measureTime(startTime)
            
            # Create result
            filterResult = QualityFilterResult(
                passed=passed,
                width=width,
                height=height,
                contrast=round(contrast, 2),
                sharpness=round(sharpness, 2),
                brightness=round(brightness, 2),
                failReason=failReason
            )
            
            # Log result
            if passed:
                self._logger.debug(
                    f"[{frameId}] Quality PASSED: size={width}x{height}, "
                    f"contrast={contrast:.1f}, sharpness={sharpness:.0f}, "
                    f"brightness={brightness:.0f}"
                )
            else:
                self._logger.warning(f"[{frameId}] Quality FAILED: {failReason}")
            
            # Save debug output
            self._saveDebugOutput(frameId, filterResult)
            
            # Log timing
            self._logTiming(frameId, processingTimeMs)
            
            return QualityFilterServiceResult(
                filterResult=filterResult,
                frameId=frameId,
                success=True,
                processingTimeMs=processingTimeMs
            )
            
        except Exception as e:
            self._logger.error(f"[{frameId}] Quality filter failed: {e}")
            return QualityFilterServiceResult(
                filterResult=QualityFilterResult(
                    passed=False,
                    width=0,
                    height=0,
                    contrast=0.0,
                    sharpness=0.0,
                    brightness=0.0,
                    failReason=str(e)
                ),
                frameId=frameId,
                success=False,
                processingTimeMs=self._measureTime(startTime)
            )
    
    def setEnabled(self, enabled: bool) -> None:
        """Enable or disable quality filtering."""
        self._enabled = enabled
        self._logger.info(f"Quality filter {'enabled' if enabled else 'disabled'}")
    
    def isEnabled(self) -> bool:
        """Check if quality filtering is enabled."""
        return self._enabled
    
    def _saveDebugOutput(
        self,
        frameId: str,
        filterResult: QualityFilterResult
    ) -> None:
        """Save debug output for quality filter step."""
        if not self._debugEnabled:
            return
        
        # Save filter result as JSON
        data = {
            "frameId": frameId,
            "passed": filterResult.passed,
            "metrics": {
                "width": filterResult.width,
                "height": filterResult.height,
                "contrast": filterResult.contrast,
                "sharpness": filterResult.sharpness,
                "brightness": filterResult.brightness
            },
            "thresholds": {
                "minWidth": self._minWidth,
                "minHeight": self._minHeight,
                "minContrast": self._minContrast,
                "minSharpness": self._minSharpness,
                "minBrightness": self._minBrightness,
                "maxBrightness": self._maxBrightness
            },
            "failReason": filterResult.failReason
        }
        self._saveDebugJson(frameId, data, "quality_filter")
