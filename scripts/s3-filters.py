#!/usr/bin/env python3
"""
S3 Preprocessing Filter Script

This script filters preprocessing images (s3) into pass/fail categories
by comparing s8 postprocessing results against template ground truth data.

Usage:
    python scripts/s3-filters.py
"""

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for the S3 filter process."""
    template_dir: Path
    s3_dir: Path
    s8_dir: Path
    pass_dir: Path
    fail_dir: Path
    
    # Fields to compare between template and s8 result
    comparison_fields: tuple = (
        "fullOrderCode",
        "dateCode",
        "facility",
        "orderType",
        "orderNumber",
        "qrPosition",
        "qrRevisionCount",
        "positionQuantity",
        "ocrPosition",
        "quantity",
        "productCode",
        "size",
        "color"
    )


@dataclass
class FilterResult:
    """Result of the filtering process."""
    total_templates: int = 0
    total_s8_results: int = 0
    total_s3_files: int = 0
    pass_count: int = 0
    fail_count: int = 0
    pass_by_template: dict = None
    
    def __post_init__(self):
        if self.pass_by_template is None:
            self.pass_by_template = {}


class TemplateLoader:
    """Loads and manages template ground truth data."""
    
    def __init__(self, template_dir: Path):
        self._template_dir = template_dir
        self._templates: dict[str, dict] = {}
    
    def load(self) -> dict[str, dict]:
        """Load all template files into a dictionary keyed by fullOrderCode."""
        self._templates.clear()
        
        for template_file in self._template_dir.glob("*.json"):
            # Skip pass/fail directories
            if template_file.parent.name in ("pass", "fail"):
                continue
                
            try:
                with open(template_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                label_data = data.get("labelData", {})
                full_order_code = label_data.get("fullOrderCode")
                
                if full_order_code:
                    self._templates[full_order_code] = label_data
                    logger.debug(f"Loaded template: {full_order_code}")
                else:
                    logger.warning(f"Template missing fullOrderCode: {template_file}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse template {template_file}: {e}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
        
        logger.info(f"Loaded {len(self._templates)} templates")
        return self._templates
    
    def get(self, full_order_code: str) -> Optional[dict]:
        """Get template data by fullOrderCode."""
        return self._templates.get(full_order_code)
    
    @property
    def count(self) -> int:
        """Return the number of loaded templates."""
        return len(self._templates)


class ResultComparator:
    """Compares s8 results against template data."""
    
    def __init__(self, comparison_fields: tuple):
        self._comparison_fields = comparison_fields
    
    def compare(self, template_data: dict, result_data: dict) -> tuple[bool, list[str]]:
        """
        Compare template data with result data.
        
        Returns:
            Tuple of (is_match, list of mismatched fields)
        """
        mismatches = []
        
        for field in self._comparison_fields:
            template_value = template_data.get(field)
            result_value = result_data.get(field)
            
            if template_value != result_value:
                mismatches.append(field)
                logger.debug(
                    f"Field mismatch - {field}: "
                    f"template={template_value}, result={result_value}"
                )
        
        is_match = len(mismatches) == 0
        return is_match, mismatches


class S3FileManager:
    """Manages s3 preprocessing files."""
    
    def __init__(self, s3_dir: Path, pass_dir: Path, fail_dir: Path):
        self._s3_dir = s3_dir
        self._pass_dir = pass_dir
        self._fail_dir = fail_dir
        self._all_frame_ids: set[str] = set()
    
    def initialize(self) -> None:
        """Initialize output directories and scan s3 files."""
        # Create output directories
        self._pass_dir.mkdir(parents=True, exist_ok=True)
        self._fail_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan all frame IDs from s3
        self._all_frame_ids.clear()
        for s3_file in self._s3_dir.glob("preprocessing_*.png"):
            frame_id = self._extract_frame_id(s3_file.stem)
            if frame_id:
                self._all_frame_ids.add(frame_id)
        
        logger.info(f"Found {len(self._all_frame_ids)} preprocessing frames in s3")
    
    def _extract_frame_id(self, filename: str) -> Optional[str]:
        """Extract frame ID from filename (e.g., preprocessing_frame_xxx -> frame_xxx)."""
        prefix = "preprocessing_"
        if filename.startswith(prefix):
            return filename[len(prefix):]
        return None
    
    def copy_to_pass(self, frame_id: str) -> bool:
        """Copy s3 files (png and json) to pass directory."""
        return self._copy_files(frame_id, self._pass_dir)
    
    def copy_to_fail(self, frame_id: str) -> bool:
        """Copy s3 files (png and json) to fail directory."""
        return self._copy_files(frame_id, self._fail_dir)
    
    def _copy_files(self, frame_id: str, target_dir: Path) -> bool:
        """Copy png file for a frame to target directory."""
        base_name = f"preprocessing_{frame_id}"
        source = self._s3_dir / f"{base_name}.png"
        
        if source.exists():
            target = target_dir / f"{base_name}.png"
            try:
                shutil.copy2(source, target)
                return True
            except Exception as e:
                logger.error(f"Failed to copy {source} to {target}: {e}")
        
        return False
    
    @property
    def all_frame_ids(self) -> set[str]:
        """Return all frame IDs found in s3."""
        return self._all_frame_ids
    
    @property
    def count(self) -> int:
        """Return the number of s3 frames."""
        return len(self._all_frame_ids)


class S8ResultProcessor:
    """Processes s8 postprocessing results."""
    
    def __init__(self, s8_dir: Path):
        self._s8_dir = s8_dir
    
    def process_all(self) -> list[tuple[str, dict]]:
        """
        Process all s8 result files.
        
        Returns:
            List of tuples (frame_id, label_data)
        """
        results = []
        
        for result_file in self._s8_dir.glob("result_*.json"):
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                frame_id = data.get("frameId")
                label_data = data.get("labelData", {})
                
                if frame_id and label_data:
                    results.append((frame_id, label_data))
                else:
                    logger.warning(f"Invalid s8 result file: {result_file}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse s8 result {result_file}: {e}")
            except Exception as e:
                logger.error(f"Error processing s8 result {result_file}: {e}")
        
        logger.info(f"Processed {len(results)} s8 results")
        return results
    
    def get_result_count(self) -> int:
        """Return the count of result files."""
        return len(list(self._s8_dir.glob("result_*.json")))


class S3FilterService:
    """Main service that orchestrates the filtering process."""
    
    def __init__(self, config: FilterConfig):
        self._config = config
        self._template_loader = TemplateLoader(config.template_dir)
        self._comparator = ResultComparator(config.comparison_fields)
        self._s3_manager = S3FileManager(
            config.s3_dir, config.pass_dir, config.fail_dir
        )
        self._s8_processor = S8ResultProcessor(config.s8_dir)
    
    def run(self) -> FilterResult:
        """Execute the filtering process."""
        logger.info("Starting S3 preprocessing filter...")
        
        # Initialize
        templates = self._template_loader.load()
        self._s3_manager.initialize()
        s8_results = self._s8_processor.process_all()
        
        # Track pass frame IDs
        pass_frame_ids: set[str] = set()
        pass_by_template: dict[str, int] = {}
        
        # Process each s8 result
        for frame_id, label_data in s8_results:
            full_order_code = label_data.get("fullOrderCode")
            
            # Check if template exists
            template_data = self._template_loader.get(full_order_code)
            if template_data is None:
                logger.debug(f"No template for {full_order_code}, frame: {frame_id}")
                continue
            
            # Compare with template
            is_match, mismatches = self._comparator.compare(template_data, label_data)
            
            if is_match:
                pass_frame_ids.add(frame_id)
                pass_by_template[full_order_code] = pass_by_template.get(full_order_code, 0) + 1
                logger.debug(f"PASS: {frame_id} matches {full_order_code}")
            else:
                logger.debug(
                    f"FAIL: {frame_id} - mismatched fields: {mismatches}"
                )
        
        # Copy files to pass/fail directories
        pass_count = 0
        fail_count = 0
        
        for frame_id in self._s3_manager.all_frame_ids:
            if frame_id in pass_frame_ids:
                if self._s3_manager.copy_to_pass(frame_id):
                    pass_count += 1
            else:
                if self._s3_manager.copy_to_fail(frame_id):
                    fail_count += 1
        
        # Build result
        result = FilterResult(
            total_templates=self._template_loader.count,
            total_s8_results=len(s8_results),
            total_s3_files=self._s3_manager.count,
            pass_count=pass_count,
            fail_count=fail_count,
            pass_by_template=pass_by_template
        )
        
        self._print_report(result)
        return result
    
    def _print_report(self, result: FilterResult) -> None:
        """Print the filtering report."""
        total = result.pass_count + result.fail_count
        pass_pct = (result.pass_count / total * 100) if total > 0 else 0
        fail_pct = (result.fail_count / total * 100) if total > 0 else 0
        
        report = f"""
{'=' * 50}
        S3 PREPROCESSING FILTER REPORT
{'=' * 50}

Input Statistics:
  - Total templates:   {result.total_templates}
  - Total s8 results:  {result.total_s8_results}
  - Total s3 files:    {result.total_s3_files}

Filter Results:
  - PASS: {result.pass_count:4d} files ({pass_pct:.1f}%)
  - FAIL: {result.fail_count:4d} files ({fail_pct:.1f}%)

Pass by Template:"""
        
        for order_code, count in sorted(result.pass_by_template.items()):
            report += f"\n  - {order_code}: {count} files"
        
        if not result.pass_by_template:
            report += "\n  (no matching results)"
        
        report += f"""

Output Directories:
  - Pass: {self._config.pass_dir}
  - Fail: {self._config.fail_dir}

{'=' * 50}
"""
        print(report)
        logger.info(f"Filter complete: {result.pass_count} pass, {result.fail_count} fail")


def get_default_config() -> FilterConfig:
    """Get default configuration based on project structure."""
    # Get project root (assuming script is in scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    return FilterConfig(
        template_dir=script_dir / "template",
        s3_dir=project_root / "output" / "debug" / "s3_preprocessing",
        s8_dir=project_root / "output" / "debug" / "s8_postprocessing",
        pass_dir=script_dir / "template" / "pass",
        fail_dir=script_dir / "template" / "fail"
    )


def main():
    """Main entry point."""
    config = get_default_config()
    
    # Validate directories
    if not config.template_dir.exists():
        logger.error(f"Template directory not found: {config.template_dir}")
        return
    
    if not config.s3_dir.exists():
        logger.error(f"S3 directory not found: {config.s3_dir}")
        return
    
    if not config.s8_dir.exists():
        logger.error(f"S8 directory not found: {config.s8_dir}")
        return
    
    # Run filter
    service = S3FilterService(config)
    service.run()


if __name__ == "__main__":
    main()
