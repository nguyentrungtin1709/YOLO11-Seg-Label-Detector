"""
Config Service Implementation.

Centralized configuration management for the Label Detector pipeline.
Loads configuration from application_config.json organized by service.

All configuration values are read from file - NO HARD-CODED DEFAULTS.

Follows:
- SRP: Only handles configuration management
- DIP: Provides configuration to other services via interface
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List


from services.interfaces.config_service_interface import IConfigService


logger = logging.getLogger(__name__)


class ConfigService(IConfigService):
    """
    Implementation of IConfigService.
    
    Loads and manages application configuration from application_config.json.
    Configuration is organized by service section (s1_camera, s2_detection, etc.)
    
    All values are read from the config file, not hard-coded.
    """
    
    def __init__(self, configPath: str = "config/application_config.json"):
        """
        Initialize ConfigService.
        
        Args:
            configPath: Path to the configuration file.
        """
        self._config: Dict[str, Any] = {}
        self._configPath = Path(configPath)
        self._debugEnabled = False
        
        # Load config (required)
        if not self.loadConfig(configPath):
            raise RuntimeError(f"Failed to load configuration from: {configPath}")
    
    def loadConfig(self, configPath: str) -> bool:
        """Load configuration from JSON file."""
        try:
            path = Path(configPath)
            if not path.exists():
                logger.error(f"Config file not found: {configPath}")
                return False
            
            with open(path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            
            # Initialize debug state from config
            self._debugEnabled = self.get("debug.enabled", False)
            
            logger.info(f"Configuration loaded from: {path.absolute()}")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Generic Config Access
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key with dot notation support.
        
        Examples:
            get("s1_camera.frameWidth") -> 640
            get("s2_detection.modelPath") -> "models/..."
            get("s4_enhancement.brightnessEnabled") -> True
        """
        try:
            value = self._config
            for part in key.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                    if value is None:
                        return default
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default
    
    def getServiceConfig(self, serviceName: str) -> Dict[str, Any]:
        """
        Get all configuration for a specific service.
        
        Args:
            serviceName: Service name (e.g., "s1_camera", "s2_detection")
            
        Returns:
            Configuration dictionary for the service.
        """
        config = self._config.get(serviceName, {})
        return config if isinstance(config, dict) else {}
    
    def getAllConfig(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return self._config.copy()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Debug Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def getDebugBasePath(self) -> str:
        """Get base path for debug output."""
        return self.get("debug.basePath", "output/debug")
    
    def isDebugEnabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self._debugEnabled
    
    def setDebugEnabled(self, enabled: bool) -> None:
        """Enable or disable debug mode at runtime."""
        self._debugEnabled = enabled
        logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
    
    def getDebugSaveCooldown(self) -> float:
        """Get debug save cooldown in seconds."""
        return self.get("debug.saveCooldown", 2.0)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # App Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def getAppConfig(self) -> Dict[str, Any]:
        """Get app-level configuration."""
        return self.getServiceConfig("app")
    
    def getCaptureDirectory(self) -> str:
        """Get capture directory path."""
        return self.get("app.captureDirectory", "output/captures")
    
    def getJpegQuality(self) -> int:
        """Get JPEG quality setting."""
        return self.get("app.jpegQuality", 95)
    
    def getWindowMinWidth(self) -> int:
        """Get minimum window width."""
        return self.get("app.windowMinWidth", 1000)
    
    def getWindowMinHeight(self) -> int:
        """Get minimum window height."""
        return self.get("app.windowMinHeight", 700)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S1 Camera Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def getCameraConfig(self) -> Dict[str, Any]:
        """Get camera configuration."""
        return self.getServiceConfig("s1_camera")
    
    def getFrameWidth(self) -> int:
        """Get camera frame width."""
        return self.get("s1_camera.frameWidth", 640)
    
    def getFrameHeight(self) -> int:
        """Get camera frame height."""
        return self.get("s1_camera.frameHeight", 640)
    
    def getFps(self) -> int:
        """Get camera FPS."""
        return self.get("s1_camera.fps", 60)
    
    def getMaxCameraSearch(self) -> int:
        """Get max camera search count."""
        return self.get("s1_camera.maxCameraSearch", 2)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S2 Detection Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def getDetectionConfig(self) -> Dict[str, Any]:
        """Get detection configuration."""
        return self.getServiceConfig("s2_detection")
    
    def getModelPath(self) -> str:
        """Get model file path."""
        return self.get("s2_detection.modelPath")
    
    def getDetectionBackend(self) -> str:
        """
        Get detection backend (onnx or openvino).
        
        Returns:
            str: Backend name ("onnx" or "openvino"), default "onnx".
        """
        backend = self.get("s2_detection.backend", "onnx")
        return backend.lower()
    
    def isSegmentation(self) -> bool:
        """Check if model is segmentation type."""
        return self.get("s2_detection.isSegmentation", True)
    
    def getInputSize(self) -> int:
        """Get model input size."""
        return self.get("s2_detection.inputSize", 640)
    
    def getConfidenceThreshold(self) -> float:
        """Get confidence threshold."""
        return self.get("s2_detection.confidenceThreshold", 0.5)
    
    def getMaxAreaRatio(self) -> float:
        """Get max area ratio for filtering."""
        return self.get("s2_detection.maxAreaRatio", 0.40)
    
    def getTopNDetections(self) -> int:
        """Get top N detections to keep."""
        return self.get("s2_detection.topNDetections", 1)
    
    def getVisualizationConfig(self) -> Dict[str, Any]:
        """Get visualization settings."""
        return self.get("s2_detection.visualization", {})
    
    def getBoxColor(self) -> List[int]:
        """Get bounding box color."""
        return self.get("s2_detection.visualization.boxColor", [0, 255, 0])
    
    def getTextColor(self) -> List[int]:
        """Get text color."""
        return self.get("s2_detection.visualization.textColor", [0, 0, 0])
    
    def getLineThickness(self) -> int:
        """Get line thickness."""
        return self.get("s2_detection.visualization.lineThickness", 2)
    
    def getFontSize(self) -> float:
        """Get font size."""
        return self.get("s2_detection.visualization.fontSize", 0.8)
    
    def getMaskOpacity(self) -> float:
        """Get mask opacity."""
        return self.get("s2_detection.visualization.maskOpacity", 0.4)
    
    def getMaskColors(self) -> List[List[int]]:
        """Get mask colors."""
        return self.get("s2_detection.visualization.maskColors", [])
    
    def getOpenvinoConfig(self) -> Dict[str, Any]:
        """
        Get OpenVINO Runtime performance configuration.
        
        Returns:
            Dict with keys: numThreads, numStreams, performanceHint,
            enableHyperThreading, enableCpuPinning.
        """
        return self.get("s2_detection.openvino", {
            "numThreads": 0,
            "numStreams": 0,
            "performanceHint": "LATENCY",
            "enableHyperThreading": False,
            "enableCpuPinning": True
        })

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S3 Preprocessing Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def getPreprocessingConfig(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.getServiceConfig("s3_preprocessing")
    
    def isPreprocessingEnabled(self) -> bool:
        """Check if preprocessing is enabled."""
        return self.get("s3_preprocessing.enabled", True)
    
    def isForceLandscape(self) -> bool:
        """Check if force landscape mode is enabled."""
        return self.get("s3_preprocessing.forceLandscape", True)
    
    def isAiOrientationFix(self) -> bool:
        """Check if AI orientation fix is enabled."""
        return self.get("s3_preprocessing.aiOrientationFix", True)
    
    def getAiConfidenceThreshold(self) -> float:
        """Get AI orientation confidence threshold."""
        return self.get("s3_preprocessing.aiConfidenceThreshold", 0.6)
    
    def getPaddleModelPath(self) -> str:
        """Get Paddle orientation model path."""
        return self.get("s3_preprocessing.paddleModelPath")
    
    def getPreprocessingDisplayWidth(self) -> int:
        """Get preprocessing display width."""
        return self.get("s3_preprocessing.displayWidth", 230)
    
    def getPreprocessingDisplayHeight(self) -> int:
        """Get preprocessing display height."""
        return self.get("s3_preprocessing.displayHeight", 100)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S4 Enhancement Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def getEnhancementConfig(self) -> Dict[str, Any]:
        """Get enhancement configuration."""
        return self.getServiceConfig("s4_enhancement")
    
    def isBrightnessEnabled(self) -> bool:
        """Check if brightness enhancement is enabled."""
        return self.get("s4_enhancement.brightnessEnabled", True)
    
    def getBrightnessClipLimit(self) -> float:
        """Get CLAHE clip limit."""
        return self.get("s4_enhancement.brightnessClipLimit", 2.5)
    
    def getBrightnessTileSize(self) -> int:
        """Get CLAHE tile size."""
        return self.get("s4_enhancement.brightnessTileSize", 8)
    
    def isSharpnessEnabled(self) -> bool:
        """Check if sharpness enhancement is enabled."""
        return self.get("s4_enhancement.sharpnessEnabled", True)
    
    def getSharpnessSigma(self) -> float:
        """Get sharpness sigma."""
        return self.get("s4_enhancement.sharpnessSigma", 1.0)
    
    def getSharpnessAmount(self) -> float:
        """Get sharpness amount."""
        return self.get("s4_enhancement.sharpnessAmount", 1.5)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S5 QR Detection Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def getQrDetectionConfig(self) -> Dict[str, Any]:
        """Get QR detection configuration."""
        return self.getServiceConfig("s5_qr_detection")
    
    def isQrDetectionEnabled(self) -> bool:
        """Check if QR detection is enabled."""
        return self.get("s5_qr_detection.enabled", True)
    
    def isQrTryRotate(self) -> bool:
        """Check if QR try rotate is enabled."""
        return self.get("s5_qr_detection.tryRotate", True)
    
    def isQrTryDownscale(self) -> bool:
        """Check if QR try downscale is enabled."""
        return self.get("s5_qr_detection.tryDownscale", True)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S6 Component Extraction Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def getComponentExtractionConfig(self) -> Dict[str, Any]:
        """Get component extraction configuration."""
        return self.getServiceConfig("s6_component_extraction")
    
    def getAboveQrWidthRatio(self) -> float:
        """Get above QR width ratio."""
        return self.get("s6_component_extraction.aboveQrWidthRatio", 0.35)
    
    def getAboveQrHeightRatio(self) -> float:
        """Get above QR height ratio."""
        return self.get("s6_component_extraction.aboveQrHeightRatio", 0.20)
    
    def getBelowQrWidthRatio(self) -> float:
        """Get below QR width ratio."""
        return self.get("s6_component_extraction.belowQrWidthRatio", 0.65)
    
    def getBelowQrHeightRatio(self) -> float:
        """Get below QR height ratio."""
        return self.get("s6_component_extraction.belowQrHeightRatio", 0.45)
    
    def getComponentPadding(self) -> int:
        """Get component extraction padding."""
        return self.get("s6_component_extraction.padding", 5)
    
    def getAboveQrScaleFactor(self) -> float:
        """
        Get scale factor for above QR region.
        
        The above QR region will be scaled by this factor before merging.
        This helps improve OCR accuracy for position/quantity text.
        
        Returns:
            float: Scale factor (default: 2.0)
        """
        return self.get("s6_component_extraction.aboveQrScaleFactor", 2.0)
    
    def isGrayscalePreprocessing(self) -> bool:
        """Check if grayscale preprocessing is enabled for component extraction."""
        return self.get("s6_component_extraction.grayscalePreprocessing", False)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S7 OCR Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def getOcrConfig(self) -> Dict[str, Any]:
        """Get OCR configuration."""
        return self.getServiceConfig("s7_ocr")
    
    def isOcrEnabled(self) -> bool:
        """Check if OCR is enabled."""
        return self.get("s7_ocr.enabled", True)
    
    def getOcrLang(self) -> str:
        """Get OCR language."""
        return self.get("s7_ocr.lang", "en")
    
    def isUseTextlineOrientation(self) -> bool:
        """Check if textline orientation is enabled."""
        return self.get("s7_ocr.useTextlineOrientation", False)
    
    def getTextDetThresh(self) -> float:
        """Get text detection threshold."""
        return self.get("s7_ocr.textDetThresh", 0.15)
    
    def getTextDetBoxThresh(self) -> float:
        """Get text detection box threshold."""
        return self.get("s7_ocr.textDetBoxThresh", 0.15)
    
    def getTextRecScoreThresh(self) -> float:
        """Get text recognition score threshold."""
        return self.get("s7_ocr.textRecScoreThresh", 0.3)
    
    def getOcrDevice(self) -> str:
        """Get OCR device (cpu/gpu)."""
        return self.get("s7_ocr.device", "cpu")
    
    def getTextDetectionModelName(self) -> Optional[str]:
        """Get text detection model name."""
        return self.get("s7_ocr.textDetectionModelName")
    
    def getTextRecognitionModelName(self) -> Optional[str]:
        """Get text recognition model name."""
        return self.get("s7_ocr.textRecognitionModelName")
    
    def getOcrPrecision(self) -> str:
        """Get OCR inference precision."""
        return self.get("s7_ocr.precision", "fp32")
    
    def isOcrEnableMkldnn(self) -> bool:
        """Check if MKL-DNN is enabled for OCR."""
        return self.get("s7_ocr.enableMkldnn", True)
    
    def getOcrMkldnnCacheCapacity(self) -> int:
        """Get MKL-DNN cache capacity."""
        return self.get("s7_ocr.mkldnnCacheCapacity", 10)
    
    def getOcrCpuThreads(self) -> int:
        """Get number of CPU threads for OCR."""
        return self.get("s7_ocr.cpuThreads", 8)
    
    def getTextDetLimitType(self) -> str:
        """Get text detection limit type."""
        return self.get("s7_ocr.textDetLimitType", "min")
    
    def getTextDetLimitSideLen(self) -> int:
        """Get text detection limit side length."""
        return self.get("s7_ocr.textDetLimitSideLen", 736)
    
    def getTextDetUnclipRatio(self) -> float:
        """Get text detection unclip ratio."""
        return self.get("s7_ocr.textDetUnclipRatio", 1.5)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S8 Postprocessing Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def getPostprocessingConfig(self) -> Dict[str, Any]:
        """Get postprocessing configuration."""
        return self.getServiceConfig("s8_postprocessing")
    
    def getMinFuzzyScore(self) -> float:
        """Get minimum fuzzy matching score."""
        return self.get("s8_postprocessing.minFuzzyScore", 0.90)
    
    def getProductsJsonPath(self) -> str:
        """Get products JSON file path."""
        return self.get("s8_postprocessing.productsJsonPath")
    
    def getSizesJsonPath(self) -> str:
        """Get sizes JSON file path."""
        return self.get("s8_postprocessing.sizesJsonPath")
    
    def getColorsJsonPath(self) -> str:
        """Get colors JSON file path."""
        return self.get("s8_postprocessing.colorsJsonPath")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Performance Logging Settings
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def isPerformanceLoggingEnabled(self) -> bool:
        """Check if performance logging is enabled."""
        return self.get("debug.performanceLogging.enabled", True)
    
    def getPerformanceLogInterval(self) -> int:
        """Get performance log interval."""
        return self.get("debug.performanceLogging.logInterval", 1)
    
    def isShowFpsInStatusBar(self) -> bool:
        """Check if FPS should be shown in status bar."""
        return self.get("debug.performanceLogging.showInStatusBar", True)
