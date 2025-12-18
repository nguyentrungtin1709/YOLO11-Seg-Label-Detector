"""
Label Detector Application

Main entry point for the Label Detector desktop application.
Uses PipelineOrchestrator to initialize all services following SOLID principles.

Architecture:
- PipelineOrchestrator: Reads config and creates all services with parameters
- Services: Receive parameters, create core components internally
- MainWindow: Receives orchestrator to access all services
"""

import sys
import os
import logging

from PySide6.QtWidgets import QApplication

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from ui.pipeline_orchestrator import PipelineOrchestrator
from ui.main_window import MainWindow


def setupLogging(debugMode: bool = False) -> None:
    """
    Setup application logging.
    
    Args:
        debugMode: If True, set log level to DEBUG.
    """
    level = logging.DEBUG if debugMode else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def createApplication(configPath: str = "config/application_config.json") -> tuple:
    """
    Create and wire up all application components using PipelineOrchestrator.
    
    Args:
        configPath: Path to the application configuration file.
        
    Returns:
        Tuple of (QApplication, MainWindow, PipelineOrchestrator).
    """
    # Create Qt Application
    app = QApplication(sys.argv)
    app.setApplicationName("Label Detector")
    app.setApplicationVersion("2.0.0")
    
    # Create Pipeline Orchestrator (initializes all services)
    orchestrator = PipelineOrchestrator(configPath)
    
    # Create Main Window with orchestrator
    mainWindow = MainWindow(orchestrator)
    
    return app, mainWindow, orchestrator


def main():
    """Main entry point."""
    # Setup logging first
    setupLogging(debugMode=os.environ.get("DEBUG", "").lower() == "true")
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Label Detector application v2.0.0")
    
    try:
        # Create application with orchestrator
        app, mainWindow, orchestrator = createApplication()
        
        # Show main window
        mainWindow.show()
        
        # Run application
        exitCode = app.exec()
        
        # Shutdown orchestrator
        orchestrator.shutdown()
        
        logger.info("Application terminated")
        sys.exit(exitCode)
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
