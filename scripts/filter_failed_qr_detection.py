"""
Filter out images that failed QR code detection.

This script compares input files from s4_enhancement with output files
from s5_qr_detection and moves failed detections to an errors folder.
"""

import os
import shutil
from pathlib import Path


def extract_frame_id(filename: str) -> str:
    """
    Extract frame ID from filename.
    
    Args:
        filename: Filename like 'enhancement_frame_20251223_094754_940.json'
        
    Returns:
        Frame ID like 'frame_20251223_094754_940'
    """
    parts = filename.replace('enhancement_', '').replace('qr_', '').replace('.json', '')
    return parts


def filter_failed_qr_detections(
    input_dir: str,
    output_dir: str,
    error_dir: str
) -> None:
    """
    Filter and move files that failed QR detection.
    
    Args:
        input_dir: Path to s4_enhancement folder
        output_dir: Path to s5_qr_detection folder
        error_dir: Path to errors folder
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    error_path = Path(error_dir)
    
    if not input_path.exists():
        print(f"Input directory does not exist: {input_dir}")
        return
    
    if not output_path.exists():
        print(f"Output directory does not exist: {output_dir}")
        return
    
    error_path.mkdir(parents=True, exist_ok=True)
    
    input_files = {f.name: f for f in input_path.glob("enhancement_*.json")}
    output_files = {f.name: f for f in output_path.glob("qr_*.json")}
    
    output_frame_ids = {
        extract_frame_id(name) for name in output_files.keys()
    }
    
    failed_count = 0
    success_count = 0
    
    for input_filename, input_file in input_files.items():
        frame_id = extract_frame_id(input_filename)
        
        if frame_id not in output_frame_ids:
            image_filename = input_filename.replace('.json', '.png')
            image_file = input_path / image_filename
            if image_file.exists():
                dest_image = error_path / image_filename
                shutil.copy2(image_file, dest_image)
                print(f"Failed QR detection: {image_filename}")
                failed_count += 1
        else:
            success_count += 1
    
    print(f"\nSummary:")
    print(f"  Total input files: {len(input_files)}")
    print(f"  Successful QR detections: {success_count}")
    print(f"  Failed QR detections: {failed_count}")
    print(f"  Error files saved to: {error_dir}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    
    input_directory = base_dir / "output/debug/s4_enhancement"
    output_directory = base_dir / "output/debug/s5_qr_detection"
    error_directory = base_dir / "output/debug/s5_qr_detection/errors"
    
    filter_failed_qr_detections(
        str(input_directory),
        str(output_directory),
        str(error_directory)
    )
