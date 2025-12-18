"""
Pipeline Orchestrator Module.

Orchestrates the 8-step label detection pipeline.
Creates ConfigService and initializes all services with proper parameters.

Pipeline Steps:
1. S1 Camera: Capture frames from camera
2. S2 Detection: Detect labels using YOLO
3. S3 Preprocessing: Crop, rotate, fix orientation
4. S4 Enhancement: Enhance brightness and sharpness
5. S5 QR Detection: Detect and decode QR codes
6. S6 Component Extraction: Extract text regions
7. S7 OCR: Extract text from regions
8. S8 Postprocessing: Fuzzy match and validate

Follows:
- SRP: Only handles pipeline orchestration
- DIP: Services receive parameters, not dependencies
- OCP: Easy to add new services
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from services.impl.config_service import ConfigService
from services.impl.s1_camera_service import S1CameraService
from services.impl.s2_detection_service import S2DetectionService
from services.impl.s3_preprocessing_service import S3PreprocessingService
from services.impl.s4_enhancement_service import S4EnhancementService
from services.impl.s5_qr_detection_service import S5QrDetectionService
from services.impl.s6_component_extraction_service import S6ComponentExtractionService
from services.impl.s7_ocr_service import S7OcrService
from services.impl.s8_postprocessing_service import S8PostprocessingService


class PipelineOrchestrator:
    """
    Orchestrates the complete label detection pipeline.
    
    Responsibilities:
    - Initialize ConfigService
    - Create all pipeline services with parameters from config
    - Manage pipeline execution flow
    - Provide access to individual services
    """
    
    def __init__(self, configPath: str = "config/application_config.json"):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            configPath: Path to the application configuration file.
        """
        self._logger = logging.getLogger(__name__)
        
        # Step 1: Initialize ConfigService (reads from JSON)
        self._configService = ConfigService(configPath)
        self._logger.info("ConfigService initialized")
        
        # Get common debug settings
        debugBasePath = self._configService.getDebugBasePath()
        debugEnabled = self._configService.isDebugEnabled()
        classNames = self._configService.get("app.classNames", ["label"])
        
        # Step 2: Initialize all services with parameters from config
        self._initializeServices(debugBasePath, debugEnabled, classNames)
        
        self._logger.info("PipelineOrchestrator initialized successfully")
    
    def _initializeServices(
        self,
        debugBasePath: str,
        debugEnabled: bool,
        classNames: List[str]
    ) -> None:
        """
        Initialize all pipeline services with parameters from config.
        
        Following DIP: Services receive parameters, not IConfigService.
        Each service creates its core components internally.
        """
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # S1 Camera Service
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self._s1CameraService = S1CameraService(
            frameWidth=self._configService.getFrameWidth(),
            frameHeight=self._configService.getFrameHeight(),
            maxCameraSearch=self._configService.getMaxCameraSearch(),
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        self._logger.info("S1CameraService initialized")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # S2 Detection Service
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self._s2DetectionService = S2DetectionService(
            modelPath=self._configService.getModelPath(),
            inputSize=self._configService.getInputSize(),
            isSegmentation=self._configService.isSegmentation(),
            classNames=classNames,
            confidenceThreshold=self._configService.getConfidenceThreshold(),
            maxAreaRatio=self._configService.getMaxAreaRatio(),
            topNDetections=self._configService.getTopNDetections(),
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        self._logger.info("S2DetectionService initialized")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # S3 Preprocessing Service
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self._s3PreprocessingService = S3PreprocessingService(
            enabled=self._configService.isPreprocessingEnabled(),
            forceLandscape=self._configService.isForceLandscape(),
            aiOrientationFix=self._configService.isAiOrientationFix(),
            aiConfidenceThreshold=self._configService.getAiConfidenceThreshold(),
            paddleModelPath=self._configService.getPaddleModelPath(),
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        self._logger.info("S3PreprocessingService initialized")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # S4 Enhancement Service
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self._s4EnhancementService = S4EnhancementService(
            enabled=True,  # Master switch - enable if any enhancer is enabled
            brightnessEnabled=self._configService.isBrightnessEnabled(),
            brightnessClipLimit=self._configService.getBrightnessClipLimit(),
            brightnessTileSize=self._configService.getBrightnessTileSize(),
            sharpnessEnabled=self._configService.isSharpnessEnabled(),
            sharpnessSigma=self._configService.getSharpnessSigma(),
            sharpnessAmount=self._configService.getSharpnessAmount(),
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        self._logger.info("S4EnhancementService initialized")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # S5 QR Detection Service
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self._s5QrDetectionService = S5QrDetectionService(
            enabled=self._configService.isQrDetectionEnabled(),
            tryRotate=self._configService.isQrTryRotate(),
            tryDownscale=self._configService.isQrTryDownscale(),
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        self._logger.info("S5QrDetectionService initialized")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # S6 Component Extraction Service
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self._s6ComponentExtractionService = S6ComponentExtractionService(
            enabled=True,  # Component extraction is always enabled
            aboveQrWidthRatio=self._configService.getAboveQrWidthRatio(),
            aboveQrHeightRatio=self._configService.getAboveQrHeightRatio(),
            belowQrWidthRatio=self._configService.getBelowQrWidthRatio(),
            belowQrHeightRatio=self._configService.getBelowQrHeightRatio(),
            padding=self._configService.getComponentPadding(),
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        self._logger.info("S6ComponentExtractionService initialized")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # S7 OCR Service
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self._s7OcrService = S7OcrService(
            enabled=self._configService.isOcrEnabled(),
            lang=self._configService.getOcrLang(),
            useTextlineOrientation=self._configService.isUseTextlineOrientation(),
            textDetThresh=self._configService.getTextDetThresh(),
            textDetBoxThresh=self._configService.getTextDetBoxThresh(),
            textRecScoreThresh=self._configService.getTextRecScoreThresh(),
            device=self._configService.getOcrDevice(),
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        self._logger.info("S7OcrService initialized")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # S8 Postprocessing Service
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self._s8PostprocessingService = S8PostprocessingService(
            enabled=True,  # Postprocessing is always enabled
            minFuzzyScore=self._configService.getMinFuzzyScore(),
            productsJsonPath=self._configService.getProductsJsonPath(),
            sizesJsonPath=self._configService.getSizesJsonPath(),
            colorsJsonPath=self._configService.getColorsJsonPath(),
            debugBasePath=debugBasePath,
            debugEnabled=debugEnabled
        )
        self._logger.info("S8PostprocessingService initialized")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Service Getters (For UI/External Access)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    @property
    def configService(self) -> ConfigService:
        """Get the configuration service."""
        return self._configService
    
    @property
    def cameraService(self) -> S1CameraService:
        """Get Step 1: Camera service."""
        return self._s1CameraService
    
    @property
    def detectionService(self) -> S2DetectionService:
        """Get Step 2: Detection service."""
        return self._s2DetectionService
    
    @property
    def preprocessingService(self) -> S3PreprocessingService:
        """Get Step 3: Preprocessing service."""
        return self._s3PreprocessingService
    
    @property
    def enhancementService(self) -> S4EnhancementService:
        """Get Step 4: Enhancement service."""
        return self._s4EnhancementService
    
    @property
    def qrDetectionService(self) -> S5QrDetectionService:
        """Get Step 5: QR detection service."""
        return self._s5QrDetectionService
    
    @property
    def componentExtractionService(self) -> S6ComponentExtractionService:
        """Get Step 6: Component extraction service."""
        return self._s6ComponentExtractionService
    
    @property
    def ocrService(self) -> S7OcrService:
        """Get Step 7: OCR service."""
        return self._s7OcrService
    
    @property
    def postprocessingService(self) -> S8PostprocessingService:
        """Get Step 8: Postprocessing service."""
        return self._s8PostprocessingService
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Debug Control
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def setDebugEnabled(self, enabled: bool) -> None:
        """
        Enable or disable debug mode for all services.
        
        Args:
            enabled: True to enable debug output for all services.
        """
        self._configService.setDebugEnabled(enabled)
        
        # Update all services
        self._s1CameraService.setDebugEnabled(enabled)
        self._s2DetectionService.setDebugEnabled(enabled)
        self._s3PreprocessingService.setDebugEnabled(enabled)
        self._s4EnhancementService.setDebugEnabled(enabled)
        self._s5QrDetectionService.setDebugEnabled(enabled)
        self._s6ComponentExtractionService.setDebugEnabled(enabled)
        self._s7OcrService.setDebugEnabled(enabled)
        self._s8PostprocessingService.setDebugEnabled(enabled)
        
        self._logger.info(f"Debug mode {'enabled' if enabled else 'disabled'} for all services")
    
    def isDebugEnabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self._configService.isDebugEnabled()
    
    def getDebugBasePath(self) -> str:
        """Get the debug base path from config."""
        return self._configService.getDebugBasePath()
    
    def savePipelineTiming(
        self,
        frameId: str,
        timing: Dict[str, float]
    ) -> Optional[str]:
        """
        Save pipeline timing information to JSON file.
        
        Only saves when debug is enabled. Timing is saved to
        output/debug/timing/timing_{frameId}.json with same naming
        convention as other debug outputs.
        
        Args:
            frameId: Frame identifier (same as other debug outputs).
            timing: Dictionary with step names and their timing in ms.
            
        Returns:
            Saved file path, or None if debug disabled or failed.
        """
        if not self.isDebugEnabled():
            return None
        
        try:
            # Create timing directory
            timingPath = Path(self.getDebugBasePath()) / "timing"
            timingPath.mkdir(parents=True, exist_ok=True)
            
            # Build timing data structure
            timingData = {
                "frameId": frameId,
                "timestamp": datetime.now().isoformat(),
                "timing_ms": timing,
                "summary": {
                    "total_ms": timing.get("total_pipeline", 0),
                    "fps": round(1000 / timing.get("total_pipeline", 1), 2) 
                           if timing.get("total_pipeline", 0) > 0 else 0
                }
            }
            
            # Save to file
            filepath = timingPath / f"timing_{frameId}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(timingData, f, indent=2, ensure_ascii=False)
            
            self._logger.debug(f"Pipeline timing saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self._logger.error(f"Failed to save pipeline timing: {e}")
            return None
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Lifecycle Management
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def shutdown(self) -> None:
        """
        Shutdown all services and release resources.
        
        Call this when the application is closing.
        """
        self._logger.info("Shutting down PipelineOrchestrator...")
        
        # Release camera resources
        if hasattr(self._s1CameraService, 'release'):
            self._s1CameraService.release()
        
        self._logger.info("PipelineOrchestrator shutdown complete")
