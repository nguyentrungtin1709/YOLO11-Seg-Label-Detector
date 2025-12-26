"""
QR Detection Error Filter Script

This script identifies images where QR detection failed by analyzing batch summary logs.
It parses 'batch_summary_*.json' files in the timing directory, finds frames with
"No QR code detected" error, and copies the corresponding enhanced images to
batch-specific error directories.

Usage:
    python scripts/qr-errors-filter.py
    python scripts/qr-errors-filter.py --debug-dir output/debug

Follows Single Responsibility Principle (SRP) from SOLID.

Date: 2025-12-26
"""

import argparse
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BatchErrorResult:
    """Result of error filtering for a single batch."""
    batchName: str
    totalFrames: int
    errorCount: int
    errorFrameIds: List[str]
    copiedFiles: List[Path]


@dataclass
class GlobalFilterResult:
    """Aggregated result of all batches."""
    batches: List[BatchErrorResult] = field(default_factory=list)
    
    @property
    def totalFrames(self) -> int:
        return sum(b.totalFrames for b in self.batches)
    
    @property
    def totalErrors(self) -> int:
        return sum(b.errorCount for b in self.batches)

    @property
    def totalCopied(self) -> int:
        return sum(len(b.copiedFiles) for b in self.batches)


class QRErrorFilter:
    """
    Filters images where QR detection failed based on batch summaries.
    """
    
    def __init__(self, debugDir: Path):
        """
        Initialize QRErrorFilter.
        
        Args:
            debugDir: Path to debug output directory.
        """
        self._debugDir = debugDir
        self._timingDir = debugDir / "timing"
        self._s4Dir = debugDir / "s4_enhancement"
        self._s5ErrorsBaseDir = debugDir / "s5_qr_detection" / "errors"
        
        # Validate directories exist
        if not self._timingDir.exists():
            raise FileNotFoundError(f"Timing directory not found: {self._timingDir}")
        if not self._s4Dir.exists():
            raise FileNotFoundError(f"S4 Enhancement directory not found: {self._s4Dir}")
            
        logger.info(f"Timing dir: {self._timingDir}")
        logger.info(f"S4 Enhancement dir: {self._s4Dir}")
        logger.info(f"Errors output base dir: {self._s5ErrorsBaseDir}")

    def processAllBatches(self) -> GlobalFilterResult:
        """
        Process all batch summary files in the timing directory.
        """
        globalResult = GlobalFilterResult()
        batchFiles = sorted(list(self._timingDir.glob("batch_summary_*.json")))
        
        if not batchFiles:
            logger.warning("No batch summary files found.")
            return globalResult

        logger.info(f"Found {len(batchFiles)} batch summary files.")
        
        for batchFile in batchFiles:
            try:
                batchResult = self._processSingleBatch(batchFile)
                globalResult.batches.append(batchResult)
            except Exception as e:
                logger.error(f"Failed to process batch {batchFile.name}: {e}")
                
        return globalResult

    def _processSingleBatch(self, batchFile: Path) -> BatchErrorResult:
        """
        Process a single batch summary file.
        """
        batchName = batchFile.stem  # e.g., batch_summary_20251226_223159
        logger.info(f"Processing batch: {batchName}")
        
        with open(batchFile, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        individualResults = data.get("individual_results", [])
        totalFrames = len(individualResults)
        
        errorFrameIds = []
        for item in individualResults:
            if item.get("error") == "No QR code detected":
                errorFrameIds.append(item.get("frameId"))
        
        copiedFiles = []
        if errorFrameIds:
            # Create batch specific error directory
            batchErrorDir = self._s5ErrorsBaseDir / batchName
            copiedFiles = self._copyErrorImages(errorFrameIds, batchErrorDir)
            
        return BatchErrorResult(
            batchName=batchName,
            totalFrames=totalFrames,
            errorCount=len(errorFrameIds),
            errorFrameIds=errorFrameIds,
            copiedFiles=copiedFiles
        )
    
    def _copyErrorImages(self, errorFrameIds: List[str], outputDir: Path) -> List[Path]:
        """
        Copy enhancement images of error frames to output directory.
        """
        outputDir.mkdir(parents=True, exist_ok=True)
        copiedFiles: List[Path] = []
        
        for frameId in errorFrameIds:
            # Construct filename. Assuming standard format: enhancement_{frameId}.png
            sourceFilename = f"enhancement_{frameId}.png"
            sourcePath = self._s4Dir / sourceFilename
            
            if not sourcePath.exists():
                logger.warning(f"Source file not found: {sourcePath}")
                continue
                
            destPath = outputDir / sourceFilename
            try:
                shutil.copy2(sourcePath, destPath)
                copiedFiles.append(destPath)
            except Exception as e:
                logger.error(f"Failed to copy {sourcePath}: {e}")
                
        return copiedFiles


def printReport(result: GlobalFilterResult) -> None:
    """Print a summary report to console."""
    print("\\n" + "=" * 80)
    print("QR DETECTION ERROR REPORT (BATCH PROCESSING)")
    print("=" * 80)
    
    print(f"{'Batch Name':<40} | {'Total':<8} | {'Errors':<8} | {'Rate':<8}")
    print("-" * 80)
    
    for batch in result.batches:
        errorRate = (batch.errorCount / batch.totalFrames * 100) if batch.totalFrames > 0 else 0
        print(f"{batch.batchName:<40} | {batch.totalFrames:<8} | {batch.errorCount:<8} | {errorRate:.1f}%")
        
    print("-" * 80)
    print(f"{'TOTAL':<40} | {result.totalFrames:<8} | {result.totalErrors:<8} | {(result.totalErrors/result.totalFrames*100 if result.totalFrames else 0):.1f}%")
    print("=" * 80)
    print(f"Total files copied to error directories: {result.totalCopied}")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Filter images where QR detection failed based on batch summaries"
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("output/debug"),
        help="Path to debug output directory (default: output/debug)"
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
        qrFilter = QRErrorFilter(args.debug_dir)
        result = qrFilter.processAllBatches()
        printReport(result)
        logger.info("Completed successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        print(f"\\nError: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
