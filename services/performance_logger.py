"""
Performance Logger Service

Tracks and logs performance metrics for the detection pipeline.
Measures preprocessing, inference, postprocessing times and calculates FPS.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable


logger = logging.getLogger(__name__)


@dataclass
class TimingInfo:
    """
    Stores timing information for a single detection cycle.
    All times are in milliseconds.
    """
    preprocessMs: float = 0.0
    inferenceMs: float = 0.0
    postprocessMs: float = 0.0
    filterMs: float = 0.0
    totalMs: float = 0.0
    fps: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"preprocess={self.preprocessMs:.1f}ms, "
            f"inference={self.inferenceMs:.1f}ms, "
            f"postprocess={self.postprocessMs:.1f}ms, "
            f"filter={self.filterMs:.1f}ms | "
            f"Total={self.totalMs:.1f}ms | "
            f"FPS={self.fps:.1f}"
        )


class PerformanceLogger:
    """
    Performance logging service for detection pipeline.
    
    Features:
    - Tracks timing for each pipeline stage
    - Calculates rolling average FPS
    - Logs performance metrics at configurable intervals
    - Supports callback for UI updates (e.g., status bar)
    
    Follows SRP: Only handles performance measurement and logging.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        logInterval: int = 30,
        rollingWindowSize: int = 30,
        onUpdate: Optional[Callable[[TimingInfo], None]] = None
    ):
        """
        Initialize PerformanceLogger.
        
        Args:
            enabled: Enable/disable performance logging.
            logInterval: Log every N frames (0 = don't log to console).
            rollingWindowSize: Number of frames for rolling average FPS.
            onUpdate: Callback function called with TimingInfo after each frame.
        """
        self._enabled = enabled
        self._logInterval = logInterval
        self._rollingWindowSize = rollingWindowSize
        self._onUpdate = onUpdate
        
        # Rolling window for FPS calculation
        self._recentTimes: deque[float] = deque(maxlen=rollingWindowSize)
        
        # Frame counter for logging interval
        self._frameCount = 0
        
        # Current timing accumulator
        self._currentTiming = TimingInfo()
        self._stageStartTime: float = 0.0
        self._cycleStartTime: float = 0.0
    
    @property
    def enabled(self) -> bool:
        """Check if performance logging is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable performance logging."""
        self._enabled = value
    
    def startCycle(self) -> None:
        """
        Start a new detection cycle.
        Call this when a new frame is received.
        """
        if not self._enabled:
            return
        
        self._cycleStartTime = time.perf_counter()
        self._currentTiming = TimingInfo()
    
    def startStage(self) -> None:
        """
        Start timing a pipeline stage.
        Call this before each stage (preprocess, inference, etc.).
        """
        if not self._enabled:
            return
        
        self._stageStartTime = time.perf_counter()
    
    def endPreprocess(self) -> float:
        """
        End preprocessing stage and record time.
        
        Returns:
            Preprocessing time in milliseconds.
        """
        if not self._enabled:
            return 0.0
        
        elapsed = (time.perf_counter() - self._stageStartTime) * 1000
        self._currentTiming.preprocessMs = elapsed
        return elapsed
    
    def endInference(self) -> float:
        """
        End inference stage and record time.
        
        Returns:
            Inference time in milliseconds.
        """
        if not self._enabled:
            return 0.0
        
        elapsed = (time.perf_counter() - self._stageStartTime) * 1000
        self._currentTiming.inferenceMs = elapsed
        return elapsed
    
    def endPostprocess(self) -> float:
        """
        End postprocessing stage and record time.
        
        Returns:
            Postprocessing time in milliseconds.
        """
        if not self._enabled:
            return 0.0
        
        elapsed = (time.perf_counter() - self._stageStartTime) * 1000
        self._currentTiming.postprocessMs = elapsed
        return elapsed
    
    def endFilter(self) -> float:
        """
        End filtering stage and record time.
        
        Returns:
            Filter time in milliseconds.
        """
        if not self._enabled:
            return 0.0
        
        elapsed = (time.perf_counter() - self._stageStartTime) * 1000
        self._currentTiming.filterMs = elapsed
        return elapsed
    
    def endCycle(self) -> TimingInfo:
        """
        End the detection cycle, calculate totals and FPS.
        
        Returns:
            TimingInfo with all metrics for this cycle.
        """
        if not self._enabled:
            return TimingInfo()
        
        # Calculate total time
        totalMs = (time.perf_counter() - self._cycleStartTime) * 1000
        self._currentTiming.totalMs = totalMs
        
        # Add to rolling window for FPS calculation
        self._recentTimes.append(totalMs)
        
        # Calculate rolling average FPS
        if len(self._recentTimes) > 0:
            avgMs = sum(self._recentTimes) / len(self._recentTimes)
            self._currentTiming.fps = 1000.0 / avgMs if avgMs > 0 else 0.0
        
        # Increment frame counter
        self._frameCount += 1
        
        # Log at interval
        if self._logInterval > 0 and self._frameCount % self._logInterval == 0:
            logger.info(f"Performance: {self._currentTiming}")
        
        # Call update callback
        if self._onUpdate:
            self._onUpdate(self._currentTiming)
        
        return self._currentTiming
    
    def getAverageFps(self) -> float:
        """
        Get the rolling average FPS.
        
        Returns:
            Average FPS over the rolling window.
        """
        if len(self._recentTimes) == 0:
            return 0.0
        
        avgMs = sum(self._recentTimes) / len(self._recentTimes)
        return 1000.0 / avgMs if avgMs > 0 else 0.0
    
    # Alias for consistency with MainWindow
    def getAverageFPS(self) -> float:
        """Alias for getAverageFps()."""
        return self.getAverageFps()
    
    def getTotalTime(self) -> float:
        """
        Get the average total time per frame in milliseconds.
        
        Returns:
            Average total time over the rolling window.
        """
        if len(self._recentTimes) == 0:
            return 0.0
        return sum(self._recentTimes) / len(self._recentTimes)
    
    def getLastTiming(self) -> TimingInfo:
        """
        Get the timing info from the last cycle.
        
        Returns:
            TimingInfo from the most recent detection cycle.
        """
        return self._currentTiming
    
    def reset(self) -> None:
        """Reset all counters and timing data."""
        self._recentTimes.clear()
        self._frameCount = 0
        self._currentTiming = TimingInfo()
    
    def recordTiming(self, timing: dict) -> TimingInfo:
        """
        Record timing from a dictionary (alternative to stage-based API).
        
        This method is called by DetectionService with timing dict from YOLODetector.
        Expected keys: preprocess_ms, inference_ms, postprocess_ms, filter_ms
        
        Args:
            timing: Dictionary with timing values in milliseconds.
            
        Returns:
            TimingInfo with all recorded metrics.
        """
        if not self._enabled:
            return TimingInfo()
        
        # Create timing info from dict
        # Support both key formats: 'preprocess' and 'preprocess_ms'
        self._currentTiming = TimingInfo(
            preprocessMs=timing.get("preprocess_ms", timing.get("preprocess", 0.0)),
            inferenceMs=timing.get("inference_ms", timing.get("inference", 0.0)),
            postprocessMs=timing.get("postprocess_ms", timing.get("postprocess", 0.0)),
            filterMs=timing.get("filter_ms", timing.get("filter", 0.0))
        )
        
        # Calculate total
        totalMs = (
            self._currentTiming.preprocessMs +
            self._currentTiming.inferenceMs +
            self._currentTiming.postprocessMs +
            self._currentTiming.filterMs
        )
        self._currentTiming.totalMs = totalMs
        
        # Add to rolling window for FPS calculation
        self._recentTimes.append(totalMs)
        
        # Calculate rolling average FPS
        if len(self._recentTimes) > 0:
            avgMs = sum(self._recentTimes) / len(self._recentTimes)
            self._currentTiming.fps = 1000.0 / avgMs if avgMs > 0 else 0.0
        
        # Increment frame counter
        self._frameCount += 1
        
        # Log at interval
        if self._logInterval > 0 and self._frameCount % self._logInterval == 0:
            logger.info(f"Performance: {self._currentTiming}")
        
        # Call update callback
        if self._onUpdate:
            self._onUpdate(self._currentTiming)
        
        return self._currentTiming
