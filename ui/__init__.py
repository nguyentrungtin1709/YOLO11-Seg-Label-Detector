# UI module for Label Detector
# Contains PySide6 widgets and windows

from ui.main_window import MainWindow
from ui.widgets.toggle_switch import ToggleSwitch
from ui.widgets.camera_widget import CameraWidget
from ui.widgets.config_panel import ConfigPanel

__all__ = [
    "MainWindow",
    "ToggleSwitch",
    "CameraWidget",
    "ConfigPanel",
]