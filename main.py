"""
Label Detector Application

Main entry point for the Label Detector desktop application.
Uses Dependency Injection to wire up all components following SOLID principles.
"""

import sys
import os
import json
import logging

from PySide6.QtWidgets import QApplication

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.camera.opencv_camera import OpenCVCamera
from core.detector.yolo_detector import YOLODetector
from core.writer.local_writer import LocalImageWriter
from core.preprocessor.geometric_transformer import GeometricTransformer
from core.preprocessor.orientation_corrector import OrientationCorrector
from core.preprocessor.document_preprocessor import DocumentPreprocessor
from core.enhancer.brightness_enhancer import BrightnessEnhancer
from core.enhancer.sharpness_enhancer import SharpnessEnhancer
from core.enhancer.image_enhancer import ImageEnhancer
from core.qr.zxing_qr_detector import ZxingQrDetector
from core.extractor.label_component_extractor import LabelComponentExtractor
from core.ocr.paddle_ocr_extractor import PaddleOcrExtractor
from core.processor.label_text_processor import LabelTextProcessor
from services.camera_service import CameraService
from services.detection_service import DetectionService
from services.image_saver_service import ImageSaverService
from services.preprocessing_service import PreprocessingService
from services.ocr_pipeline_service import OcrPipelineService
from services.performance_logger import PerformanceLogger
from ui.main_window import MainWindow


def setupLogging(debugMode: bool = False) -> None:
    """
    Setup application logging.
    
    Args:
        debugMode: If True, set log level to DEBUG.
    """
    level = logging.DEBUG if debugMode else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def loadConfig(configPath: str = "config/app_config.json") -> dict:
    """
    Load application configuration from JSON file.
    
    Args:
        configPath: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    defaultConfig = {
        "modelPath": "models/yolo11n_best.onnx",
        "confidenceThreshold": 0.5,
        "frameWidth": 640,
        "frameHeight": 640,
        "fps": 30,
        "outputDirectory": "output",
        "boxColor": [0, 255, 0],
        "textColor": [255, 255, 255],
        "lineThickness": 2,
        "fontSize": 0.6,
        "jpegQuality": 95
    }
    
    try:
        with open(configPath, 'r') as f:
            userConfig = json.load(f)
            # Merge user config with defaults
            defaultConfig.update(userConfig)
    except FileNotFoundError:
        logging.warning(f"Config file not found: {configPath}, using defaults")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing config file: {e}, using defaults")
    
    return defaultConfig


def createApplication(config: dict) -> tuple:
    """
    Create and wire up all application components using Dependency Injection.
    
    Args:
        config: Application configuration.
        
    Returns:
        Tuple of (QApplication, MainWindow).
    """
    # Create Qt Application
    app = QApplication(sys.argv)
    app.setApplicationName("Label Detector")
    app.setApplicationVersion("1.0.0")
    
    # Create Core Layer components
    camera = OpenCVCamera(maxCameraSearch=config.get("maxCameraSearch", 10))
    detector = YOLODetector(
        inputSize=config.get("inputSize", 640),
        isSegmentation=config.get("isSegmentation", False)
    )
    imageWriter = LocalImageWriter(quality=config.get("jpegQuality", 95))
    
    # Create Service Layer components
    cameraService = CameraService(camera)
    
    # Create performance logger if enabled
    perfLoggingConfig = config.get("performanceLogging", {})
    performanceLogger = None
    if perfLoggingConfig.get("enabled", False):
        performanceLogger = PerformanceLogger(
            logInterval=perfLoggingConfig.get("logInterval", 30)
        )
    
    # Get filter settings from config
    filterSettings = config.get("filterSettings", {})
    detectionService = DetectionService(
        detector,
        maxAreaRatio=filterSettings.get("maxAreaRatio", 0.15),
        topNDetections=filterSettings.get("topNDetections", 3),
        performanceLogger=performanceLogger
    )
    
    imageSaverService = ImageSaverService(
        imageWriter=imageWriter,
        captureDirectory=config.get("captureDirectory", "output/captures"),
        debugDirectory=config.get("debugDirectory", "output/debug"),
        boxColor=tuple(config.get("boxColor", [0, 255, 0])),
        textColor=tuple(config.get("textColor", [255, 255, 255])),
        lineThickness=config.get("lineThickness", 2),
        fontSize=config.get("fontSize", 0.6),
        maskOpacity=config.get("maskOpacity", 0.4),
        maskColors=config.get("maskColors", None)
    )
    
    # Create Preprocessing Service
    preprocessingConfig = config.get("preprocessing", {})
    geometricTransformer = GeometricTransformer()
    
    # Resolve paddle model path (relative to project root or absolute)
    paddleModelPath = preprocessingConfig.get("paddleModelPath")
    if paddleModelPath and not os.path.isabs(paddleModelPath):
        paddleModelPath = os.path.join(PROJECT_ROOT, paddleModelPath)
    
    orientationCorrector = OrientationCorrector(
        aiConfidenceThreshold=preprocessingConfig.get("aiConfidenceThreshold", 0.6),
        modelPath=paddleModelPath
    )
    documentPreprocessor = DocumentPreprocessor(
        geometricTransformer=geometricTransformer,
        orientationCorrector=orientationCorrector
    )
    
    # Create Image Enhancer
    enhancementConfig = preprocessingConfig.get("enhancement", {})
    brightnessEnhancer = BrightnessEnhancer(
        clipLimit=enhancementConfig.get("brightnessClipLimit", 2.5),
        tileGridSize=(
            enhancementConfig.get("brightnessTileSize", 8),
            enhancementConfig.get("brightnessTileSize", 8)
        )
    )
    sharpnessEnhancer = SharpnessEnhancer(
        sigma=enhancementConfig.get("sharpnessSigma", 1.0),
        amount=enhancementConfig.get("sharpnessAmount", 1.5)
    )
    imageEnhancer = ImageEnhancer(
        brightnessEnhancer=brightnessEnhancer,
        sharpnessEnhancer=sharpnessEnhancer
    )
    
    preprocessingService = PreprocessingService(
        preprocessor=documentPreprocessor,
        enhancer=imageEnhancer,
        enabled=preprocessingConfig.get("enabled", True),
        forceLandscape=preprocessingConfig.get("forceLandscape", True),
        aiOrientationFix=preprocessingConfig.get("aiOrientationFix", True),
        brightnessEnabled=enhancementConfig.get("brightnessEnabled", True),
        sharpnessEnabled=enhancementConfig.get("sharpnessEnabled", True)
    )
    
    # Create OCR Pipeline Service
    ocrPipelineConfig = config.get("ocrPipeline", {})
    ocrPipelineService = None
    
    if ocrPipelineConfig.get("enabled", False):
        # Create OCR pipeline components
        qrConfig = ocrPipelineConfig.get("qrDetector", {})
        qrDetector = ZxingQrDetector(
            tryRotate=qrConfig.get("tryRotate", True),
            tryDownscale=qrConfig.get("tryDownscale", True)
        )
        
        componentConfig = ocrPipelineConfig.get("componentExtractor", {})
        componentExtractor = LabelComponentExtractor(
            aboveQrWidthRatio=componentConfig.get("aboveQrWidthRatio", 0.35),
            aboveQrHeightRatio=componentConfig.get("aboveQrHeightRatio", 0.20),
            belowQrWidthRatio=componentConfig.get("belowQrWidthRatio", 0.65),
            belowQrHeightRatio=componentConfig.get("belowQrHeightRatio", 0.45),
            padding=componentConfig.get("padding", 5)
        )
        
        ocrConfig = ocrPipelineConfig.get("ocr", {})
        ocrExtractor = PaddleOcrExtractor(
            lang=ocrConfig.get("lang", "en"),
            useTextlineOrientation=ocrConfig.get("useTextlineOrientation", False),
            textDetThresh=ocrConfig.get("textDetThresh", 0.3),
            textDetBoxThresh=ocrConfig.get("textDetBoxThresh", 0.5),
            textRecScoreThresh=ocrConfig.get("textRecScoreThresh", 0.5),
            device=ocrConfig.get("device", "cpu")
        )
        
        textProcessorConfig = ocrPipelineConfig.get("textProcessor", {})
        textProcessor = LabelTextProcessor(
            productsJsonPath=textProcessorConfig.get("productsJsonPath"),
            sizesJsonPath=textProcessorConfig.get("sizesJsonPath"),
            colorsJsonPath=textProcessorConfig.get("colorsJsonPath"),
            minFuzzyScore=textProcessorConfig.get("minFuzzyScore", 0.75)
        )
        
        ocrPipelineService = OcrPipelineService(
            qrDetector=qrDetector,
            componentExtractor=componentExtractor,
            ocrExtractor=ocrExtractor,
            textProcessor=textProcessor,
            debugEnabled=False,  # Controlled by UI debug toggle
            debugBasePath=ocrPipelineConfig.get("debugBasePath", "output/debug")
        )
    
    # Create Main Window (UI Layer)
    mainWindow = MainWindow(
        cameraService=cameraService,
        detectionService=detectionService,
        imageSaverService=imageSaverService,
        preprocessingService=preprocessingService,
        ocrPipelineService=ocrPipelineService,
        config=config
    )
    
    return app, mainWindow


def main():
    """Main entry point."""
    # Load configuration
    config = loadConfig()
    
    # Setup logging
    setupLogging(debugMode=os.environ.get("DEBUG", "").lower() == "true")
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Label Detector application")
    
    # Create application
    app, mainWindow = createApplication(config)
    
    # Show main window
    mainWindow.show()
    
    # Run application
    exitCode = app.exec()
    
    logger.info("Application terminated")
    sys.exit(exitCode)


if __name__ == "__main__":
    main()
