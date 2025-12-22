"""
Backend Performance Comparison Script.

Reads detection JSON files from OpenVINO and ONNX output folders,
compares average processing times, and generates a report with visualization.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np


def readJsonFiles(folderPath: str) -> List[Dict]:
    """
    Read all JSON files from a folder.
    
    Args:
        folderPath: Path to folder containing JSON files
        
    Returns:
        List of parsed JSON data
    """
    jsonData = []
    folderPathObj = Path(folderPath)
    
    if not folderPathObj.exists():
        print(f"Warning: Folder does not exist: {folderPath}")
        return jsonData
    
    for jsonFile in folderPathObj.glob("detection_*.json"):
        try:
            with open(jsonFile, 'r', encoding='utf-8') as f:
                data = json.load(f)
                jsonData.append(data)
        except Exception as e:
            print(f"Error reading {jsonFile.name}: {e}")
    
    return jsonData


def calculateAverageTime(jsonDataList: List[Dict]) -> Tuple[float, int, List[float]]:
    """
    Calculate average processing time from JSON data list.
    
    Args:
        jsonDataList: List of JSON data containing processingTimeMs
        
    Returns:
        Tuple of (average_time, count, all_times)
    """
    times = []
    
    for data in jsonDataList:
        if "processingTimeMs" in data:
            times.append(data["processingTimeMs"])
    
    if not times:
        return 0.0, 0, []
    
    averageTime = sum(times) / len(times)
    return averageTime, len(times), times


def generateReport(
    openvinoStats: Tuple[float, int, List[float]],
    onnxStats: Tuple[float, int, List[float]],
    outputPath: str
) -> None:
    """
    Generate text report comparing backend performance.
    
    Args:
        openvinoStats: (avg_time, count, times) for OpenVINO
        onnxStats: (avg_time, count, times) for ONNX
        outputPath: Path to save report file
    """
    openvinoAvg, openvinoCount, openvinoTimes = openvinoStats
    onnxAvg, onnxCount, onnxTimes = onnxStats
    
    # Calculate additional statistics
    openvinoMin = min(openvinoTimes) if openvinoTimes else 0
    openvinoMax = max(openvinoTimes) if openvinoTimes else 0
    openvinoMedian = np.median(openvinoTimes) if openvinoTimes else 0
    openvinoStd = np.std(openvinoTimes) if openvinoTimes else 0
    
    onnxMin = min(onnxTimes) if onnxTimes else 0
    onnxMax = max(onnxTimes) if onnxTimes else 0
    onnxMedian = np.median(onnxTimes) if onnxTimes else 0
    onnxStd = np.std(onnxTimes) if onnxTimes else 0
    
    # Calculate performance difference
    if onnxAvg > 0:
        speedup = ((onnxAvg - openvinoAvg) / onnxAvg) * 100
        fasterBackend = "OpenVINO" if speedup > 0 else "ONNX"
        speedupAbs = abs(speedup)
    else:
        speedup = 0
        fasterBackend = "N/A"
        speedupAbs = 0
    
    # Generate report content
    reportLines = [
        "=" * 80,
        "YOLO11 BACKEND PERFORMANCE COMPARISON REPORT",
        "=" * 80,
        "",
        "OPENVINO BACKEND",
        "-" * 80,
        f"  Total samples processed: {openvinoCount}",
        f"  Average processing time: {openvinoAvg:.2f} ms",
        f"  Median processing time:  {openvinoMedian:.2f} ms",
        f"  Min processing time:     {openvinoMin:.2f} ms",
        f"  Max processing time:     {openvinoMax:.2f} ms",
        f"  Standard deviation:      {openvinoStd:.2f} ms",
        "",
        "ONNX BACKEND",
        "-" * 80,
        f"  Total samples processed: {onnxCount}",
        f"  Average processing time: {onnxAvg:.2f} ms",
        f"  Median processing time:  {onnxMedian:.2f} ms",
        f"  Min processing time:     {onnxMin:.2f} ms",
        f"  Max processing time:     {onnxMax:.2f} ms",
        f"  Standard deviation:      {onnxStd:.2f} ms",
        "",
        "COMPARISON",
        "-" * 80,
        f"  Faster backend:          {fasterBackend}",
        f"  Performance difference:  {speedupAbs:.2f}% faster",
        f"  Time difference:         {abs(onnxAvg - openvinoAvg):.2f} ms",
        "",
        "PERFORMANCE METRICS",
        "-" * 80,
        f"  OpenVINO FPS (avg):      {1000.0 / openvinoAvg:.2f} fps" if openvinoAvg > 0 else "  OpenVINO FPS (avg):      N/A",
        f"  ONNX FPS (avg):          {1000.0 / onnxAvg:.2f} fps" if onnxAvg > 0 else "  ONNX FPS (avg):          N/A",
        "",
        "=" * 80,
    ]
    
    # Write report to file
    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(reportLines))
    
    print(f"Report saved to: {outputPath}")
    
    # Print to console
    print('\n'.join(reportLines))


def generateBarChart(
    openvinoAvg: float,
    onnxAvg: float,
    outputPath: str
) -> None:
    """
    Generate bar chart comparing backend performance.
    
    Args:
        openvinoAvg: Average processing time for OpenVINO
        onnxAvg: Average processing time for ONNX
        outputPath: Path to save chart image
    """
    backends = ['OpenVINO', 'ONNX']
    times = [openvinoAvg, onnxAvg]
    colors = ['#1f77b4', '#ff7f0e']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(backends, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{time:.2f} ms',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )
    
    # Chart styling
    ax.set_ylabel('Average Processing Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('YOLO11 Backend Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(times) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add performance difference annotation
    if onnxAvg > 0:
        speedup = ((onnxAvg - openvinoAvg) / onnxAvg) * 100
        fasterBackend = "OpenVINO" if speedup > 0 else "ONNX"
        speedupText = f'{fasterBackend} is {abs(speedup):.1f}% faster'
        
        ax.text(
            0.5, 0.95,
            speedupText,
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    plt.savefig(outputPath, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {outputPath}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("YOLO11 Backend Performance Comparison")
    print("=" * 80)
    print()
    
    # Define paths
    baseDir = Path("output/debug/s2_detection")
    openvinoDir = baseDir / "OpenVINO"
    onnxDir = baseDir / "ONNX"
    
    reportPath = baseDir / "backend_comparison_report.txt"
    chartPath = baseDir / "backend_comparison_chart.png"
    
    # Read JSON files
    print(f"Reading OpenVINO data from: {openvinoDir}")
    openvinoData = readJsonFiles(str(openvinoDir))
    print(f"  Found {len(openvinoData)} files")
    
    print(f"Reading ONNX data from: {onnxDir}")
    onnxData = readJsonFiles(str(onnxDir))
    print(f"  Found {len(onnxData)} files")
    print()
    
    # Calculate statistics
    openvinoStats = calculateAverageTime(openvinoData)
    onnxStats = calculateAverageTime(onnxData)
    
    openvinoAvg, openvinoCount, _ = openvinoStats
    onnxAvg, onnxCount, _ = onnxStats
    
    if openvinoCount == 0 or onnxCount == 0:
        print("Error: Insufficient data for comparison")
        if openvinoCount == 0:
            print(f"  No valid OpenVINO data found in {openvinoDir}")
        if onnxCount == 0:
            print(f"  No valid ONNX data found in {onnxDir}")
        return
    
    # Generate outputs
    print("Generating report...")
    generateReport(openvinoStats, onnxStats, str(reportPath))
    print()
    
    print("Generating chart...")
    generateBarChart(openvinoAvg, onnxAvg, str(chartPath))
    print()
    
    print("=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
