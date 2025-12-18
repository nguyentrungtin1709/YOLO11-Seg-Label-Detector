"""
Base Service Interface Module.

Defines the base interface and common data classes for all pipeline services.
All services should inherit from IBaseService to ensure consistent behavior.

Follows:
- ISP (Interface Segregation Principle): Minimal base interface
- DIP (Dependency Inversion Principle): High-level modules depend on abstractions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from pathlib import Path
import logging
import json
import time


@dataclass
class ServiceResult:
    """
    Base result class for all services.
    
    Attributes:
        success: Whether the operation succeeded.
        data: The result data (type varies by service).
        errorMessage: Error description if success is False.
        processingTimeMs: Processing time in milliseconds.
        frameId: Unique identifier for the frame being processed.
    """
    success: bool
    data: Optional[Any] = None
    errorMessage: str = ""
    processingTimeMs: float = 0.0
    frameId: str = ""


@dataclass
class TimingInfo:
    """
    Timing information for a service.
    
    Attributes:
        serviceName: Name of the service.
        processingTimeMs: Processing time in milliseconds.
        timestamp: When the processing occurred.
    """
    serviceName: str
    processingTimeMs: float
    timestamp: str = ""


class IBaseService(ABC):
    """
    Base interface for all pipeline services.
    
    Provides common functionality for:
    - Service identification
    - Debug output management
    - Logging configuration
    - Timing measurement
    """
    
    @abstractmethod
    def getServiceName(self) -> str:
        """
        Get the service name for logging and debug output.
        
        Returns:
            str: Service name (e.g., "s1_camera", "s2_detection")
        """
        pass
    
    @abstractmethod
    def setDebugEnabled(self, enabled: bool) -> None:
        """
        Enable or disable debug output.
        
        Args:
            enabled: True to enable debug output, False to disable.
        """
        pass
    
    @abstractmethod
    def isDebugEnabled(self) -> bool:
        """
        Check if debug output is enabled.
        
        Returns:
            bool: True if debug is enabled.
        """
        pass


class BaseService(IBaseService):
    """
    Base implementation for all pipeline services.
    
    Provides common functionality that can be inherited by concrete services.
    This is not an interface but a helper base class.
    """
    
    def __init__(
        self,
        serviceName: str,
        debugBasePath: str = "output/debug",
        debugEnabled: bool = False
    ):
        """
        Initialize BaseService.
        
        Args:
            serviceName: Name of the service (e.g., "s1_camera").
            debugBasePath: Base path for debug output.
            debugEnabled: Whether debug output is enabled.
        """
        self._serviceName = serviceName
        self._debugBasePath = Path(debugBasePath) / serviceName
        self._debugEnabled = debugEnabled
        self._logger = logging.getLogger(serviceName)
        
        # Ensure debug directory exists if enabled
        if debugEnabled:
            self._ensureDebugDirectory()
    
    def getServiceName(self) -> str:
        """Get the service name."""
        return self._serviceName
    
    def setDebugEnabled(self, enabled: bool) -> None:
        """Enable or disable debug output."""
        self._debugEnabled = enabled
        if enabled:
            self._ensureDebugDirectory()
        self._logger.info(f"Debug {'enabled' if enabled else 'disabled'}")
    
    def isDebugEnabled(self) -> bool:
        """Check if debug is enabled."""
        return self._debugEnabled
    
    def _ensureDebugDirectory(self) -> None:
        """Create debug directory if it doesn't exist."""
        self._debugBasePath.mkdir(parents=True, exist_ok=True)
    
    def _saveDebugImage(self, frameId: str, image: Any, prefix: str = "") -> Optional[str]:
        """
        Save debug image with consistent naming.
        
        Args:
            frameId: Frame identifier for naming.
            image: Image to save (numpy array).
            prefix: Optional prefix for filename.
            
        Returns:
            Saved file path, or None if debug is disabled or failed.
        """
        if not self._debugEnabled or image is None:
            return None
        
        try:
            import cv2
            filename = f"{prefix}_{frameId}.png" if prefix else f"{frameId}.png"
            filepath = self._debugBasePath / filename
            cv2.imwrite(str(filepath), image)
            self._logger.debug(f"Saved debug image: {filepath}")
            return str(filepath)
        except Exception as e:
            self._logger.warning(f"Failed to save debug image: {e}")
            return None
    
    def _saveDebugJson(self, frameId: str, data: Dict, prefix: str = "") -> Optional[str]:
        """
        Save debug JSON with consistent naming.
        
        Args:
            frameId: Frame identifier for naming.
            data: Data to save as JSON.
            prefix: Optional prefix for filename.
            
        Returns:
            Saved file path, or None if debug is disabled or failed.
        """
        if not self._debugEnabled:
            return None
        
        try:
            filename = f"{prefix}_{frameId}.json" if prefix else f"{frameId}.json"
            filepath = self._debugBasePath / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            self._logger.debug(f"Saved debug JSON: {filepath}")
            return str(filepath)
        except Exception as e:
            self._logger.warning(f"Failed to save debug JSON: {e}")
            return None
    
    def _logTiming(self, frameId: str, processingTimeMs: float) -> None:
        """
        Log processing time to console.
        
        Args:
            frameId: Frame identifier.
            processingTimeMs: Processing time in milliseconds.
        """
        self._logger.info(f"[{frameId}] Processing time: {processingTimeMs:.2f}ms")
    
    def _measureTime(self, startTime: float) -> float:
        """
        Calculate elapsed time in milliseconds.
        
        Args:
            startTime: Start time from time.time().
            
        Returns:
            Elapsed time in milliseconds.
        """
        return (time.time() - startTime) * 1000
