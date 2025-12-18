"""
Services Interfaces Package.

Exports all service interfaces for the Label Detector pipeline.
"""

from services.interfaces.base_service_interface import (
    ServiceResult,
    TimingInfo,
    IBaseService,
    BaseService
)

from services.interfaces.config_service_interface import IConfigService

from services.interfaces.camera_service_interface import (
    CameraFrame,
    ICameraService
)

from services.interfaces.detection_service_interface import (
    DetectionServiceResult,
    IDetectionService
)

from services.interfaces.preprocessing_service_interface import (
    PreprocessingServiceResult,
    IPreprocessingService
)

from services.interfaces.enhancement_service_interface import (
    EnhancementServiceResult,
    IEnhancementService
)

from services.interfaces.qr_detection_service_interface import (
    QrDetectionServiceResult,
    IQrDetectionService
)

from services.interfaces.component_extraction_service_interface import (
    ComponentExtractionServiceResult,
    IComponentExtractionService
)

from services.interfaces.ocr_service_interface import (
    OcrServiceResult,
    IOcrService
)

from services.interfaces.postprocessing_service_interface import (
    PostprocessingServiceResult,
    IPostprocessingService
)


__all__ = [
    # Base
    "ServiceResult",
    "TimingInfo",
    "IBaseService",
    "BaseService",
    # Config
    "IConfigService",
    # Step 1: Camera
    "CameraFrame",
    "ICameraService",
    # Step 2: Detection
    "DetectionServiceResult",
    "IDetectionService",
    # Step 3: Preprocessing
    "PreprocessingServiceResult",
    "IPreprocessingService",
    # Step 4: Enhancement
    "EnhancementServiceResult",
    "IEnhancementService",
    # Step 5: QR Detection
    "QrDetectionServiceResult",
    "IQrDetectionService",
    # Step 6: Component Extraction
    "ComponentExtractionServiceResult",
    "IComponentExtractionService",
    # Step 7: OCR
    "OcrServiceResult",
    "IOcrService",
    # Step 8: Postprocessing
    "PostprocessingServiceResult",
    "IPostprocessingService",
]
