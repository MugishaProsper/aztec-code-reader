"""Configuration settings for Aztec Code Reader"""

from dataclasses import dataclass
from typing import Tuple, List
import json
from pathlib import Path

@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters"""
    clahe_clip_limit: float = 4.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    min_data_length: int = 3
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [
                "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif",
                "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.TIFF", "*.TIF"
            ]
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'ProcessingConfig':
        """Load configuration from JSON file"""
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file"""
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

@dataclass
class OutputConfig:
    """Configuration for output settings"""
    csv_fieldnames: List[str] = None
    draw_colors: List[Tuple[int, int, int]] = None
    overlay_alpha: float = 0.2
    font_scale: float = 0.5
    line_thickness: int = 2
    
    def __post_init__(self):
        if self.csv_fieldnames is None:
            self.csv_fieldnames = [
                'filename', 'aztec_index', 'data', 'x', 'y', 'width', 'height',
                'confidence', 'processing_method', 'image_width', 'image_height', 
                'file_size_kb', 'processing_time_ms'
            ]
        
        if self.draw_colors is None:
            self.draw_colors = [
                (0, 165, 255),  # Orange
                (0, 255, 0),    # Green
                (255, 0, 255),  # Magenta
                (255, 255, 0),  # Cyan
                (255, 0, 0),    # Blue
                (0, 255, 255),  # Yellow
            ]

# Default configurations
DEFAULT_PROCESSING_CONFIG = ProcessingConfig()
DEFAULT_OUTPUT_CONFIG = OutputConfig()