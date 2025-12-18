"""
Config Service Interface Module.

Defines the interface for centralized configuration management.
The config service loads and provides access to all application settings.

Follows:
- SRP: Only handles configuration management
- DIP: Other services depend on this abstraction
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class IConfigService(ABC):
    """
    Interface for configuration management.
    
    Provides centralized access to all application configuration.
    Supports dot notation for nested config access.
    """
    
    @abstractmethod
    def loadConfig(self, configPath: str) -> bool:
        """
        Load configuration from a JSON file.
        
        Args:
            configPath: Path to the configuration file.
            
        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Supports dot notation for nested access:
        - "modelPath" -> config["modelPath"]
        - "preprocessing.forceLandscape" -> config["preprocessing"]["forceLandscape"]
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value or default.
        """
        pass
    
    @abstractmethod
    def getServiceConfig(self, serviceName: str) -> Dict[str, Any]:
        """
        Get all configuration for a specific service.
        
        Args:
            serviceName: Name of the service (e.g., "camera", "detection").
            
        Returns:
            Dictionary with service-specific configuration.
        """
        pass
    
    @abstractmethod
    def getDebugBasePath(self) -> str:
        """
        Get the base path for debug output.
        
        Returns:
            str: Base path for debug files.
        """
        pass
    
    @abstractmethod
    def isDebugEnabled(self) -> bool:
        """
        Check if debug mode is enabled globally.
        
        Returns:
            bool: True if debug is enabled.
        """
        pass
    
    @abstractmethod
    def setDebugEnabled(self, enabled: bool) -> None:
        """
        Enable or disable debug mode globally.
        
        Args:
            enabled: True to enable debug mode.
        """
        pass
    
    @abstractmethod
    def getAllConfig(self) -> Dict[str, Any]:
        """
        Get the complete configuration dictionary.
        
        Returns:
            Dict: Complete configuration.
        """
        pass
