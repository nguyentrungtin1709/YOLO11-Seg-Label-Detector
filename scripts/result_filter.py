"""
Result Filter Script

This script analyzes the ML pipeline output to identify three types of errors:
1. OCR Errors: Text recognition failures in component extraction
2. QR Detection Errors: QR code detection failures  
3. YOLO Detection Errors: Label detection failures

The script follows SOLID principles:
- Single Responsibility: Each class handles one specific error type
- Open/Closed: Easy to extend with new error types
- Interface Segregation: Clear interfaces for each error detector
- Dependency Inversion: All detectors follow a common interface

Date: 2025-12-23
"""

import json
import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ErrorDetectionResult:
    """Result of error detection operation"""
    errorType: str
    totalFrames: int
    errorCount: int
    errorFrames: List[str]
    savedFilePaths: List[Path]


class IErrorDetector(ABC):
    """Interface for error detection strategies"""
    
    @abstractmethod
    def detectErrors(self) -> ErrorDetectionResult:
        """Detect errors and return results"""
        pass
    
    @abstractmethod
    def saveErrorArtifacts(self, errorFrames: Set[str]) -> List[Path]:
        """Save error artifacts to errors folder"""
        pass


class OCRErrorDetector(IErrorDetector):
    """
    Detects OCR errors by comparing postprocessing results with ground truth templates.
    
    Algorithm:
    1. Load ground truth templates
    2. Compare with s8_postprocessing results
    3. Identify incorrect predictions
    4. Save corresponding component images from s6
    """
    
    def __init__(self, templateDir: Path, s8Dir: Path, s6Dir: Path):
        self._templateDir = templateDir
        self._s8Dir = s8Dir
        self._s6Dir = s6Dir
        self._errorDir = s8Dir / "errors"
        self._groundTruthData: Dict[str, dict] = {}
        
    def _loadGroundTruth(self) -> None:
        """Load ground truth data from template files"""
        logger.info(f"Loading ground truth templates from {self._templateDir}")
        
        for templateFile in self._templateDir.glob("*.json"):
            try:
                with open(templateFile, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    labelData = data.get("labelData", {})
                    fullOrderCode = labelData.get("fullOrderCode", "")
                    
                    if fullOrderCode:
                        self._groundTruthData[fullOrderCode] = labelData
                        logger.debug(f"Loaded template: {fullOrderCode}")
            except Exception as e:
                logger.error(f"Failed to load template {templateFile}: {e}")
        
        logger.info(f"Loaded {len(self._groundTruthData)} ground truth templates")
    
    def _compareResults(self) -> Set[str]:
        """Compare s8 results with ground truth and return error frames"""
        errorFrames: Set[str] = set()
        
        logger.info(f"Comparing results in {self._s8Dir}")
        
        for resultFile in self._s8Dir.glob("result_frame_*.json"):
            try:
                with open(resultFile, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    frameId = data.get("frameId", "")
                    labelData = data.get("labelData", {})
                    fullOrderCode = labelData.get("fullOrderCode", "")
                    
                    if fullOrderCode in self._groundTruthData:
                        groundTruth = self._groundTruthData[fullOrderCode]
                        
                        # Compare key fields
                        if not self._isMatchingGroundTruth(labelData, groundTruth):
                            errorFrames.add(frameId)
                            logger.warning(
                                f"OCR error detected in {frameId}: "
                                f"predicted={fullOrderCode}, mismatch in fields"
                            )
            except Exception as e:
                logger.error(f"Failed to process result file {resultFile}: {e}")
        
        return errorFrames
    
    def _isMatchingGroundTruth(self, predicted: dict, groundTruth: dict) -> bool:
        """Check if predicted data matches ground truth"""
        keyFields = [
            "fullOrderCode", "dateCode", "facility", "orderType",
            "orderNumber", "qrPosition", "positionQuantity", "ocrPosition",
            "quantity", "productCode", "size", "color"
        ]
        
        for field in keyFields:
            predictedValue = str(predicted.get(field, "")).strip()
            truthValue = str(groundTruth.get(field, "")).strip()
            
            if predictedValue != truthValue:
                logger.debug(
                    f"Field mismatch: {field} - "
                    f"predicted='{predictedValue}', truth='{truthValue}'"
                )
                return False
        
        return True
    
    def detectErrors(self) -> ErrorDetectionResult:
        """Detect OCR errors"""
        logger.info("Starting OCR error detection")
        
        self._loadGroundTruth()
        errorFrames = self._compareResults()
        savedPaths = self.saveErrorArtifacts(errorFrames)
        
        totalFrames = len(list(self._s8Dir.glob("result_frame_*.json")))
        
        return ErrorDetectionResult(
            errorType="OCR",
            totalFrames=totalFrames,
            errorCount=len(errorFrames),
            errorFrames=sorted(errorFrames),
            savedFilePaths=savedPaths
        )
    
    def saveErrorArtifacts(self, errorFrames: Set[str]) -> List[Path]:
        """Save component images for error frames"""
        self._errorDir.mkdir(parents=True, exist_ok=True)
        savedPaths: List[Path] = []
        
        logger.info(f"Saving OCR error artifacts to {self._errorDir}")
        
        for frameId in errorFrames:
            # Only save components PNG images (not above_qr, below_qr, or JSON)
            sourcePath = self._s6Dir / f"components_{frameId}.png"
            if sourcePath.exists():
                destPath = self._errorDir / sourcePath.name
                try:
                    shutil.copy2(sourcePath, destPath)
                    savedPaths.append(destPath)
                    logger.debug(f"Copied {sourcePath.name} to errors")
                except Exception as e:
                    logger.error(f"Failed to copy {sourcePath}: {e}")
        
        logger.info(f"Saved {len(savedPaths)} OCR error artifacts")
        return savedPaths


class QRDetectionErrorDetector(IErrorDetector):
    """
    Detects QR detection errors by comparing enhancement and QR detection outputs.
    
    Algorithm:
    1. List all frames in s4_enhancement
    2. Check which frames have QR detection results in s5_qr_detection
    3. Frames without QR results are errors
    4. Save corresponding enhancement images
    """
    
    def __init__(self, s4Dir: Path, s5Dir: Path):
        self._s4Dir = s4Dir
        self._s5Dir = s5Dir
        self._errorDir = s5Dir / "errors"
    
    def _extractFrameId(self, filename: str) -> str:
        """Extract frame ID from filename"""
        # enhancement_frame_20251223_131024_283.png -> frame_20251223_131024_283
        # qr_frame_20251223_131024_283.json -> frame_20251223_131024_283
        if filename.startswith("enhancement_"):
            return filename.replace("enhancement_", "").replace(".png", "").replace(".json", "")
        elif filename.startswith("qr_"):
            return filename.replace("qr_", "").replace(".json", "")
        return filename
    
    def _getEnhancementFrames(self) -> Set[str]:
        """Get all frame IDs from enhancement directory"""
        frames: Set[str] = set()
        
        for enhancementFile in self._s4Dir.glob("enhancement_frame_*.png"):
            frameId = self._extractFrameId(enhancementFile.name)
            frames.add(frameId)
        
        logger.info(f"Found {len(frames)} enhancement frames")
        return frames
    
    def _getQRDetectionFrames(self) -> Set[str]:
        """Get all frame IDs that have QR detection results"""
        frames: Set[str] = set()
        
        for qrFile in self._s5Dir.glob("qr_frame_*.json"):
            frameId = self._extractFrameId(qrFile.name)
            frames.add(frameId)
        
        logger.info(f"Found {len(frames)} QR detection frames")
        return frames
    
    def detectErrors(self) -> ErrorDetectionResult:
        """Detect QR detection errors"""
        logger.info("Starting QR detection error detection")
        
        enhancementFrames = self._getEnhancementFrames()
        qrDetectionFrames = self._getQRDetectionFrames()
        
        # Frames in s4 but not in s5 are QR detection failures
        errorFrames = enhancementFrames - qrDetectionFrames
        
        logger.info(
            f"QR detection errors: {len(errorFrames)} out of "
            f"{len(enhancementFrames)} frames"
        )
        
        savedPaths = self.saveErrorArtifacts(errorFrames)
        
        return ErrorDetectionResult(
            errorType="QR_DETECTION",
            totalFrames=len(enhancementFrames),
            errorCount=len(errorFrames),
            errorFrames=sorted(errorFrames),
            savedFilePaths=savedPaths
        )
    
    def saveErrorArtifacts(self, errorFrames: Set[str]) -> List[Path]:
        """Save enhancement images for error frames"""
        self._errorDir.mkdir(parents=True, exist_ok=True)
        savedPaths: List[Path] = []
        
        logger.info(f"Saving QR detection error artifacts to {self._errorDir}")
        
        for frameId in errorFrames:
            # Only save enhancement PNG images (not JSON)
            sourcePath = self._s4Dir / f"enhancement_{frameId}.png"
            if sourcePath.exists():
                destPath = self._errorDir / sourcePath.name
                try:
                    shutil.copy2(sourcePath, destPath)
                    savedPaths.append(destPath)
                    logger.debug(f"Copied {sourcePath.name} to errors")
                except Exception as e:
                    logger.error(f"Failed to copy {sourcePath}: {e}")
        
        logger.info(f"Saved {len(savedPaths)} QR detection error artifacts")
        return savedPaths


class YOLODetectionErrorDetector(IErrorDetector):
    """
    Detects YOLO detection errors by comparing camera and detection outputs.
    
    Algorithm:
    1. List all frames in s1_camera
    2. Check which frames have detection results in s2_detection
    3. Frames without detection results are YOLO errors
    4. Save corresponding camera images
    """
    
    def __init__(self, s1Dir: Path, s2Dir: Path):
        self._s1Dir = s1Dir
        self._s2Dir = s2Dir
        self._errorDir = s2Dir / "errors"
    
    def _extractFrameId(self, filename: str) -> str:
        """Extract frame ID from filename"""
        # frame_20251223_131024_283.png -> frame_20251223_131024_283
        return filename.replace(".png", "").replace(".json", "")
    
    def _getCameraFrames(self) -> Set[str]:
        """Get all frame IDs from camera directory"""
        frames: Set[str] = set()
        
        for cameraFile in self._s1Dir.glob("frame_*.png"):
            frameId = self._extractFrameId(cameraFile.name)
            frames.add(frameId)
        
        logger.info(f"Found {len(frames)} camera frames")
        return frames
    
    def _getDetectionFrames(self) -> Set[str]:
        """Get all frame IDs that have successful detection (cropped images)"""
        frames: Set[str] = set()
        
        # Only count frames that have cropped images (actual detections)
        for croppedFile in self._s2Dir.glob("cropped_frame_*.png"):
            # cropped_frame_20251223_131024_283_0.png -> frame_20251223_131024_283
            fileName = croppedFile.name.replace("cropped_", "")
            # Remove _0.png, _1.png etc suffix
            frameId = fileName.rsplit("_", 1)[0]
            frames.add(frameId)
        
        logger.info(f"Found {len(frames)} frames with successful detections")
        return frames
    
    def detectErrors(self) -> ErrorDetectionResult:
        """Detect YOLO detection errors"""
        logger.info("Starting YOLO detection error detection")
        
        cameraFrames = self._getCameraFrames()
        detectionFrames = self._getDetectionFrames()
        
        # Frames in s1 but not in s2 are YOLO detection failures
        errorFrames = cameraFrames - detectionFrames
        
        logger.info(
            f"YOLO detection errors: {len(errorFrames)} out of "
            f"{len(cameraFrames)} frames"
        )
        
        savedPaths = self.saveErrorArtifacts(errorFrames)
        
        return ErrorDetectionResult(
            errorType="YOLO_DETECTION",
            totalFrames=len(cameraFrames),
            errorCount=len(errorFrames),
            errorFrames=sorted(errorFrames),
            savedFilePaths=savedPaths
        )
    
    def saveErrorArtifacts(self, errorFrames: Set[str]) -> List[Path]:
        """Save camera images for error frames"""
        self._errorDir.mkdir(parents=True, exist_ok=True)
        savedPaths: List[Path] = []
        
        logger.info(f"Saving YOLO detection error artifacts to {self._errorDir}")
        
        for frameId in errorFrames:
            sourcePath = self._s1Dir / f"{frameId}.png"
            if sourcePath.exists():
                destPath = self._errorDir / sourcePath.name
                try:
                    shutil.copy2(sourcePath, destPath)
                    savedPaths.append(destPath)
                    logger.debug(f"Copied {sourcePath.name} to errors")
                except Exception as e:
                    logger.error(f"Failed to copy {sourcePath}: {e}")
        
        logger.info(f"Saved {len(savedPaths)} YOLO detection error artifacts")
        return savedPaths


class ResultFilterOrchestrator:
    """
    Orchestrates the error detection process.
    
    Follows the Facade pattern to provide a simple interface
    for running all error detectors.
    """
    
    def __init__(self, debugDir: Path, templateDir: Path):
        self._debugDir = debugDir
        self._templateDir = templateDir
        self._detectors: List[IErrorDetector] = []
        self._initializeDetectors()
    
    def _initializeDetectors(self) -> None:
        """Initialize all error detectors"""
        logger.info("Initializing error detectors")
        
        # OCR Error Detector
        s8Dir = self._debugDir / "s8_postprocessing"
        s6Dir = self._debugDir / "s6_component_extraction"
        if s8Dir.exists() and s6Dir.exists():
            self._detectors.append(
                OCRErrorDetector(self._templateDir, s8Dir, s6Dir)
            )
            logger.info("OCR error detector initialized")
        else:
            logger.warning(
                f"Skipping OCR detector: missing directories "
                f"(s8: {s8Dir.exists()}, s6: {s6Dir.exists()})"
            )
        
        # QR Detection Error Detector
        s4Dir = self._debugDir / "s4_enhancement"
        s5Dir = self._debugDir / "s5_qr_detection"
        if s4Dir.exists() and s5Dir.exists():
            self._detectors.append(
                QRDetectionErrorDetector(s4Dir, s5Dir)
            )
            logger.info("QR detection error detector initialized")
        else:
            logger.warning(
                f"Skipping QR detector: missing directories "
                f"(s4: {s4Dir.exists()}, s5: {s5Dir.exists()})"
            )
        
        # YOLO Detection Error Detector
        s1Dir = self._debugDir / "s1_camera"
        s2Dir = self._debugDir / "s2_detection"
        if s1Dir.exists() and s2Dir.exists():
            self._detectors.append(
                YOLODetectionErrorDetector(s1Dir, s2Dir)
            )
            logger.info("YOLO detection error detector initialized")
        else:
            logger.warning(
                f"Skipping YOLO detector: missing directories "
                f"(s1: {s1Dir.exists()}, s2: {s2Dir.exists()})"
            )
    
    def runAllDetectors(self) -> List[ErrorDetectionResult]:
        """Run all error detectors and return results"""
        logger.info(f"Running {len(self._detectors)} error detectors")
        results: List[ErrorDetectionResult] = []
        
        for detector in self._detectors:
            try:
                result = detector.detectErrors()
                results.append(result)
                logger.info(
                    f"{result.errorType}: {result.errorCount}/{result.totalFrames} "
                    f"errors ({result.errorCount/result.totalFrames*100:.1f}%)"
                )
            except Exception as e:
                logger.error(f"Error running detector: {e}", exc_info=True)
        
        return results
    
    def generateReport(self, results: List[ErrorDetectionResult]) -> str:
        """Generate a summary report"""
        report = "=" * 80 + "\n"
        report += "ERROR DETECTION SUMMARY REPORT\n"
        report += "=" * 80 + "\n\n"
        
        for result in results:
            report += f"Error Type: {result.errorType}\n"
            report += f"Total Frames: {result.totalFrames}\n"
            report += f"Error Count: {result.errorCount}\n"
            
            if result.totalFrames > 0:
                errorRate = (result.errorCount / result.totalFrames) * 100
                report += f"Error Rate: {errorRate:.2f}%\n"
            
            report += f"Artifacts Saved: {len(result.savedFilePaths)}\n"
            
            if result.errorFrames:
                report += f"Error Frames (first 10): {result.errorFrames[:10]}\n"
            
            report += "\n" + "-" * 80 + "\n\n"
        
        return report
    
    def saveReport(self, results: List[ErrorDetectionResult], outputPath: Path) -> None:
        """Save report to file"""
        report = self.generateReport(results)
        
        try:
            with open(outputPath, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to {outputPath}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")


def main():
    """Main entry point"""
    # Configuration
    projectRoot = Path(__file__).parent.parent
    debugDir = projectRoot / "output" / "debug"
    templateDir = projectRoot / "scripts" / "template"
    reportPath = debugDir / "error_detection_report.txt"
    
    logger.info("Starting Result Filter Script")
    logger.info(f"Debug directory: {debugDir}")
    logger.info(f"Template directory: {templateDir}")
    
    # Validate paths
    if not debugDir.exists():
        logger.error(f"Debug directory not found: {debugDir}")
        return
    
    if not templateDir.exists():
        logger.error(f"Template directory not found: {templateDir}")
        return
    
    # Run error detection
    orchestrator = ResultFilterOrchestrator(debugDir, templateDir)
    results = orchestrator.runAllDetectors()
    
    # Generate and save report
    report = orchestrator.generateReport(results)
    print(report)
    
    orchestrator.saveReport(results, reportPath)
    
    logger.info("Result Filter Script completed successfully")


if __name__ == "__main__":
    main()
