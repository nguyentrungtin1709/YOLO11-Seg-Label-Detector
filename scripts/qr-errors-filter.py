"""
QR Detection Error Filter Script

This script identifies images where QR detection failed.
It compares S4 Enhancement outputs with S5 QR Detection results
to find frames that have enhanced images but no QR detection results.

Usage:
    python scripts/qr-errors-filter.py
    python scripts/qr-errors-filter.py --debug-dir output/debug
    python scripts/qr-errors-filter.py --copy-to output/qr_errors

Follows Single Responsibility Principle (SRP) from SOLID.

Date: 2025-12-26
"""

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Set, List


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QRErrorFilterResult:
    """Result of QR error filtering operation."""
    totalEnhancedFrames: int
    totalQrDetectedFrames: int
    errorCount: int
    errorFrameIds: List[str]
    copiedFiles: List[Path]


class QRErrorFilter:
    """
    Filters images where QR detection failed.
    
    Compares S4 Enhancement outputs with S5 QR Detection results
    to identify frames that failed QR detection.
    """
    
    def __init__(self, debugDir: Path):
        """
        Initialize QRErrorFilter.
        
        Args:
            debugDir: Path to debug output directory.
        """
        self._debugDir = debugDir
        self._s4Dir = debugDir / "s4_enhancement"
        self._s5Dir = debugDir / "s5_qr_detection"
        
        # Validate directories exist
        if not self._s4Dir.exists():
            raise FileNotFoundError(f"S4 Enhancement directory not found: {self._s4Dir}")
        if not self._s5Dir.exists():
            raise FileNotFoundError(f"S5 QR Detection directory not found: {self._s5Dir}")
        
        logger.info(f"S4 Enhancement dir: {self._s4Dir}")
        logger.info(f"S5 QR Detection dir: {self._s5Dir}")
    
    def _extractFrameId(self, filename: str, prefix: str) -> str:
        """
        Extract frame ID from filename.
        
        Args:
            filename: File name (e.g., enhancement_frame_20251224_141647_032.png)
            prefix: Prefix to remove (e.g., "enhancement_")
            
        Returns:
            Frame ID (e.g., frame_20251224_141647_032)
        """
        return filename.replace(prefix, "").replace(".png", "").replace(".json", "")
    
    def _getEnhancementFrames(self) -> Set[str]:
        """Get all frame IDs from S4 enhancement directory."""
        frames: Set[str] = set()
        
        for enhancementFile in self._s4Dir.glob("enhancement_frame_*.png"):
            frameId = self._extractFrameId(enhancementFile.name, "enhancement_")
            frames.add(frameId)
        
        logger.info(f"Found {len(frames)} enhanced frames in S4")
        return frames
    
    def _getQRDetectionFrames(self) -> Set[str]:
        """Get all frame IDs that have QR detection results in S5."""
        frames: Set[str] = set()
        
        for qrFile in self._s5Dir.glob("qr_frame_*.json"):
            frameId = self._extractFrameId(qrFile.name, "qr_")
            frames.add(frameId)
        
        logger.info(f"Found {len(frames)} QR detected frames in S5")
        return frames
    
    def findErrors(self) -> QRErrorFilterResult:
        """
        Find frames where QR detection failed.
        
        Returns:
            QRErrorFilterResult with error information.
        """
        logger.info("Scanning for QR detection errors...")
        
        enhancementFrames = self._getEnhancementFrames()
        qrDetectionFrames = self._getQRDetectionFrames()
        
        # Frames in S4 but not in S5 are QR detection failures
        errorFrameIds = enhancementFrames - qrDetectionFrames
        
        logger.info(f"QR detection errors: {len(errorFrameIds)} / {len(enhancementFrames)} frames")
        
        if len(enhancementFrames) > 0:
            errorRate = (len(errorFrameIds) / len(enhancementFrames)) * 100
            logger.info(f"Error rate: {errorRate:.2f}%")
        
        return QRErrorFilterResult(
            totalEnhancedFrames=len(enhancementFrames),
            totalQrDetectedFrames=len(qrDetectionFrames),
            errorCount=len(errorFrameIds),
            errorFrameIds=sorted(errorFrameIds),
            copiedFiles=[]
        )
    
    def copyErrorImages(self, errorFrameIds: List[str], outputDir: Path) -> List[Path]:
        """
        Copy enhancement images of error frames to output directory.
        
        Args:
            errorFrameIds: List of frame IDs that failed QR detection.
            outputDir: Directory to copy error images to.
            
        Returns:
            List of copied file paths.
        """
        outputDir.mkdir(parents=True, exist_ok=True)
        copiedFiles: List[Path] = []
        
        logger.info(f"Copying {len(errorFrameIds)} error images to {outputDir}")
        
        for frameId in errorFrameIds:
            sourcePath = self._s4Dir / f"enhancement_{frameId}.png"
            
            if sourcePath.exists():
                destPath = outputDir / sourcePath.name
                try:
                    shutil.copy2(sourcePath, destPath)
                    copiedFiles.append(destPath)
                    logger.debug(f"Copied: {sourcePath.name}")
                except Exception as e:
                    logger.error(f"Failed to copy {sourcePath}: {e}")
            else:
                logger.warning(f"Source file not found: {sourcePath}")
        
        logger.info(f"Successfully copied {len(copiedFiles)} files")
        return copiedFiles


def printReport(result: QRErrorFilterResult) -> None:
    """Print a summary report to console."""
    print("\n" + "=" * 60)
    print("QR DETECTION ERROR REPORT")
    print("=" * 60)
    print(f"Total Enhanced Frames (S4):    {result.totalEnhancedFrames}")
    print(f"Total QR Detected Frames (S5): {result.totalQrDetectedFrames}")
    print(f"QR Detection Errors:           {result.errorCount}")
    
    if result.totalEnhancedFrames > 0:
        errorRate = (result.errorCount / result.totalEnhancedFrames) * 100
        successRate = 100 - errorRate
        print(f"Success Rate:                  {successRate:.2f}%")
        print(f"Error Rate:                    {errorRate:.2f}%")
    
    if result.copiedFiles:
        print(f"Files Copied:                  {len(result.copiedFiles)}")
    
    print("=" * 60)
    
    if result.errorFrameIds:
        print("\nError Frame IDs:")
        print("-" * 40)
        for i, frameId in enumerate(result.errorFrameIds, 1):
            print(f"  {i:3d}. {frameId}")
    else:
        print("\nNo QR detection errors found!")
    
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Filter images where QR detection failed"
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("output/debug"),
        help="Path to debug output directory (default: output/debug)"
    )
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="Copy error images to this directory (optional)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting QR Detection Error Filter")
    
    try:
        # Create filter
        qrFilter = QRErrorFilter(args.debug_dir)
        
        # Find errors
        result = qrFilter.findErrors()
        
        # Copy error images if output directory specified
        if args.copy_to and result.errorFrameIds:
            result.copiedFiles = qrFilter.copyErrorImages(
                result.errorFrameIds,
                args.copy_to
            )
        elif result.errorFrameIds:
            # Default: copy to s5_qr_detection/errors
            defaultErrorDir = args.debug_dir / "s5_qr_detection" / "errors"
            result.copiedFiles = qrFilter.copyErrorImages(
                result.errorFrameIds,
                defaultErrorDir
            )
        
        # Print report
        printReport(result)
        
        logger.info("QR Detection Error Filter completed")
        
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        print(f"\nError: {e}")
        print("Make sure you run 'python scripts/detection.py --debug' first!")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
