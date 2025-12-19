#!/usr/bin/env python3
"""
Batch Detection Script.

Process images from a directory through the label detection pipeline.
Replaces real-time camera capture with batch image processing.

Usage:
    python scripts/detection.py
    python scripts/detection.py --input samples/ --debug
    python scripts/detection.py --debug --limit 10

Pipeline Steps (S2-S8):
    S2: Detection       - Detect label using YOLO
    S3: Preprocessing   - Crop, rotate, fix orientation
    S4: Enhancement     - Brightness, sharpness
    S5: QR Detection    - Detect and decode QR code
    S6: Component Ext.  - Extract text regions
    S7: OCR             - Text recognition
    S8: Postprocessing  - Fuzzy match and validate

Output:
    When --debug is enabled, results are automatically saved to output/debug/
    by the existing service debug mechanisms.

Follows:
    - SRP: Single responsibility for batch processing
    - DIP: Depends on abstractions via PipelineOrchestrator
    - Additive Only: No modifications to existing code
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ui.pipeline_orchestrator import PipelineOrchestrator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Image Loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def loadImages(inputDir: str) -> List[Path]:
    """
    Load list of image files from directory.
    
    Args:
        inputDir: Path to input directory containing images.
        
    Returns:
        List of image file paths, sorted by name.
    """
    logger = logging.getLogger(__name__)
    inputPath = Path(inputDir)
    
    if not inputPath.exists():
        logger.error(f"Input directory not found: {inputDir}")
        return []
    
    if not inputPath.is_dir():
        logger.error(f"Input path is not a directory: {inputDir}")
        return []
    
    # Find all supported image files
    imageFiles = []
    for ext in SUPPORTED_EXTENSIONS:
        imageFiles.extend(inputPath.glob(f"*{ext}"))
        imageFiles.extend(inputPath.glob(f"*{ext.upper()}"))
    
    # Sort by filename and remove duplicates
    imageFiles = sorted(set(imageFiles), key=lambda p: p.name.lower())
    
    logger.info(f"Found {len(imageFiles)} images in {inputDir}")
    return imageFiles


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Image Processing (Reuse logic from main_window._updateFrame)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def processImage(orchestrator: PipelineOrchestrator, image, frameId: str) -> dict:
    """
    Process a single image through pipeline S2-S8.
    
    Logic follows main_window._updateFrame() but without UI updates.
    Debug output is automatically saved by services when debug mode is enabled.
    
    Args:
        orchestrator: Pipeline orchestrator with all services.
        image: OpenCV image (BGR format).
        frameId: Unique frame identifier (used for debug output naming).
        
    Returns:
        Dictionary with processing results and timing.
    """
    logger = logging.getLogger(__name__)
    startTime = time.time()
    timing = {}
    
    result = {
        "frameId": frameId,
        "success": False,
        "detection": False,
        "qrCode": None,
        "ocrResult": None,
        "error": None,
        "timing": timing
    }
    
    # Get services from orchestrator
    detectionService = orchestrator.detectionService
    preprocessingService = orchestrator.preprocessingService
    enhancementService = orchestrator.enhancementService
    qrDetectionService = orchestrator.qrDetectionService
    componentExtractionService = orchestrator.componentExtractionService
    ocrService = orchestrator.ocrService
    postprocessingService = orchestrator.postprocessingService
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S2: Detection
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    detectionResult = detectionService.detect(image, frameId)
    timing["s2_detection"] = detectionResult.processingTimeMs
    
    if not detectionResult.success or not detectionResult.detections:
        result["error"] = "No label detected"
        result["timing"]["total"] = (time.time() - startTime) * 1000
        return result
    
    result["detection"] = True
    firstDetection = detectionResult.detections[0]
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S3: Preprocessing
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if not preprocessingService.isEnabled():
        result["error"] = "Preprocessing disabled"
        result["timing"]["total"] = (time.time() - startTime) * 1000
        return result
    
    preprocessResult = preprocessingService.preprocess(image, firstDetection, frameId)
    timing["s3_preprocessing"] = preprocessResult.processingTimeMs
    
    if not preprocessResult.success or preprocessResult.croppedImage is None:
        result["error"] = "Preprocessing failed"
        result["timing"]["total"] = (time.time() - startTime) * 1000
        return result
    
    processedImage = preprocessResult.croppedImage
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S4: Enhancement
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if enhancementService.isEnabled():
        enhanceResult = enhancementService.enhance(processedImage, frameId)
        timing["s4_enhancement"] = enhanceResult.processingTimeMs
        
        if enhanceResult.success and enhanceResult.enhancedImage is not None:
            processedImage = enhanceResult.enhancedImage
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S5: QR Detection
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    qrResult = qrDetectionService.detectQr(processedImage, frameId)
    timing["s5_qr_detection"] = qrResult.processingTimeMs
    
    if not qrResult.success or qrResult.qrData is None:
        result["error"] = "No QR code detected"
        result["timing"]["total"] = (time.time() - startTime) * 1000
        return result
    
    result["qrCode"] = qrResult.qrData.text
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S6: Component Extraction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    componentResult = componentExtractionService.extractComponents(
        processedImage,
        qrResult.qrData.polygon,
        frameId
    )
    timing["s6_component_extraction"] = componentResult.processingTimeMs
    
    if not componentResult.success or componentResult.mergedImage is None:
        result["error"] = "Component extraction failed"
        result["timing"]["total"] = (time.time() - startTime) * 1000
        return result
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S7: OCR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ocrResult = ocrService.extractText(componentResult.mergedImage, frameId)
    timing["s7_ocr"] = ocrResult.processingTimeMs
    
    textBlocks = []
    if ocrResult.success and ocrResult.ocrData:
        textBlocks = ocrResult.ocrData.textBlocks
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # S8: Postprocessing
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    postResult = postprocessingService.process(textBlocks, qrResult.qrData, frameId)
    timing["s8_postprocessing"] = postResult.processingTimeMs
    
    if postResult.success and postResult.labelData:
        result["ocrResult"] = {
            "productCode": postResult.labelData.productCode,
            "size": postResult.labelData.size,
            "color": postResult.labelData.color
        }
        result["success"] = True
    else:
        result["error"] = "Postprocessing failed"
    
    # Calculate total time
    result["timing"]["total"] = (time.time() - startTime) * 1000
    
    return result


def processAll(
    orchestrator: PipelineOrchestrator,
    inputDir: str,
    limit: Optional[int] = None
) -> List[dict]:
    """
    Process all images in a directory.
    
    Args:
        orchestrator: Pipeline orchestrator with all services.
        inputDir: Input directory containing images.
        limit: Maximum number of images to process (None = all).
        
    Returns:
        List of result dictionaries for each processed image.
    """
    logger = logging.getLogger(__name__)
    
    # Load image list
    imageFiles = loadImages(inputDir)
    
    if not imageFiles:
        logger.warning("No images found to process")
        return []
    
    # Apply limit
    if limit is not None and limit > 0:
        imageFiles = imageFiles[:limit]
        logger.info(f"Processing limited to {limit} images")
    
    # Process each image
    results = []
    totalCount = len(imageFiles)
    successCount = 0
    
    logger.info(f"Starting batch processing of {totalCount} images...")
    batchStartTime = time.time()
    
    for idx, imagePath in enumerate(imageFiles, 1):
        logger.info(f"[{idx}/{totalCount}] Processing: {imagePath.name}")
        
        # Read image
        image = cv2.imread(str(imagePath))
        if image is None:
            logger.error(f"  ✗ Failed to read image")
            results.append({
                "frameId": imagePath.stem,
                "success": False,
                "error": "Failed to read image"
            })
            continue
        
        # Use filename (without extension) as frameId
        frameId = imagePath.stem
        
        # Process image
        result = processImage(orchestrator, image, frameId)
        results.append(result)
        
        # Log result
        if result["success"]:
            successCount += 1
            ocrData = result.get("ocrResult", {})
            logger.info(
                f"  ✓ ProductCode: {ocrData.get('productCode', 'N/A')}, "
                f"Size: {ocrData.get('size', 'N/A')}, "
                f"Color: {ocrData.get('color', 'N/A')}"
            )
        else:
            logger.warning(f"  ✗ {result.get('error', 'Unknown error')}")
        
        # Log timing
        timing = result.get("timing", {})
        logger.info(f"  Time: {timing.get('total', 0):.1f}ms")
    
    # Summary
    batchTotalTime = (time.time() - batchStartTime) * 1000
    logger.info("=" * 60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total images:  {totalCount}")
    logger.info(f"Success:       {successCount} ({successCount/totalCount*100:.1f}%)")
    logger.info(f"Failed:        {totalCount - successCount}")
    logger.info(f"Total time:    {batchTotalTime:.1f}ms")
    logger.info(f"Avg time:      {batchTotalTime/totalCount:.1f}ms per image")
    
    if orchestrator.isDebugEnabled():
        logger.info(f"Debug output:  {orchestrator.getDebugBasePath()}")
    
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI Interface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def setupLogging(debugMode: bool = False) -> None:
    """Setup application logging."""
    level = logging.DEBUG if debugMode else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parseArgs() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch detection script - process images from directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/detection.py
  python scripts/detection.py --input samples/ --debug
  python scripts/detection.py --debug --limit 10

Output:
  When --debug is enabled, results are saved to output/debug/
  by the existing service debug mechanisms.
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="samples",
        help="Input directory containing images (default: samples)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/application_config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Maximum number of images to process"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode (saves output to output/debug/)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parseArgs()
    
    # Setup logging
    setupLogging(debugMode=args.debug)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("BATCH DETECTION SCRIPT")
    logger.info("=" * 60)
    logger.info(f"Input:  {args.input}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Debug:  {args.debug}")
    if args.limit:
        logger.info(f"Limit:  {args.limit}")
    logger.info("=" * 60)
    
    # Check input directory
    inputPath = Path(args.input)
    if not inputPath.exists():
        logger.error(f"Input directory not found: {args.input}")
        logger.info(f"Please create '{args.input}' directory and add images.")
        sys.exit(1)
    
    try:
        # Create orchestrator
        logger.info("Initializing pipeline...")
        orchestrator = PipelineOrchestrator(args.config)
        
        # Enable debug mode if requested
        if args.debug:
            orchestrator.setDebugEnabled(True)
            logger.info(f"Debug output will be saved to: {orchestrator.getDebugBasePath()}")
        
        # Load model (already loaded during init, but verify)
        modelPath = orchestrator.configService.getModelPath()
        if not orchestrator.detectionService.isModelLoaded():
            if not orchestrator.detectionService.loadModel(modelPath):
                logger.error(f"Failed to load model: {modelPath}")
                sys.exit(1)
        
        # IMPORTANT: Enable detection service (disabled by default)
        orchestrator.detectionService.setEnabled(True)
        
        logger.info("Pipeline initialized successfully")
        logger.info("=" * 60)
        
        # Process all images
        results = processAll(
            orchestrator=orchestrator,
            inputDir=args.input,
            limit=args.limit
        )
        
        # Shutdown
        orchestrator.shutdown()
        
        # Exit code based on results
        if not results:
            sys.exit(1)
        
        successCount = sum(1 for r in results if r.get("success", False))
        sys.exit(0 if successCount > 0 else 1)
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
