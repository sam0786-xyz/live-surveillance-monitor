"""
Configuration Loader
Loads and validates YAML configuration
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DetectionConfig:
    """Detection configuration"""
    model: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "mps"
    vehicle_classes: list = field(default_factory=lambda: [2, 5, 7])


@dataclass
class PlateConfig:
    """Plate recognition configuration"""
    enabled: bool = True
    model: str = "yolov8n.pt"
    ocr_languages: list = field(default_factory=lambda: ["en"])
    confidence_threshold: float = 0.4
    plate_pattern: str = r"^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}$"


@dataclass
class TrackingConfig:
    """Tracking configuration"""
    enabled: bool = True
    algorithm: str = "deepsort"
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3


@dataclass
class VideoConfig:
    """Video input configuration"""
    source: str = "0"
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    show_fps: bool = True
    show_track_id: bool = True
    show_confidence: bool = True
    bbox_thickness: int = 2
    colors: dict = field(default_factory=lambda: {
        'vehicle': [0, 255, 0],
        'selected': [0, 165, 255],
        'plate': [255, 0, 0]
    })


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    save_detections: bool = False
    output_dir: str = "logs"


@dataclass
class AppConfig:
    """Complete application configuration"""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    plate_recognition: PlateConfig = field(default_factory=PlateConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class ConfigLoader:
    """Load and manage application configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[AppConfig] = None
        self._raw_config: Dict[str, Any] = {}
    
    def load(self) -> AppConfig:
        """
        Load configuration from file.
        
        Returns:
            AppConfig object
        """
        if self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self._raw_config = yaml.safe_load(f) or {}
        else:
            self._raw_config = {}
        
        self._config = self._parse_config(self._raw_config)
        return self._config
    
    def _parse_config(self, raw: Dict[str, Any]) -> AppConfig:
        """Parse raw config dict into AppConfig"""
        
        # Detection config
        det_raw = raw.get('detection', {})
        detection = DetectionConfig(
            model=det_raw.get('model', 'yolov8n.pt'),
            confidence_threshold=det_raw.get('confidence_threshold', 0.5),
            iou_threshold=det_raw.get('iou_threshold', 0.45),
            device=det_raw.get('device', 'mps'),
            vehicle_classes=det_raw.get('vehicle_classes', [2, 5, 7])
        )
        
        # Plate recognition config
        plate_raw = raw.get('plate_recognition', {})
        plate = PlateConfig(
            enabled=plate_raw.get('enabled', True),
            model=plate_raw.get('model', 'yolov8n.pt'),
            ocr_languages=plate_raw.get('ocr_languages', ['en']),
            confidence_threshold=plate_raw.get('confidence_threshold', 0.4),
            plate_pattern=plate_raw.get('plate_pattern', r'^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}$')
        )
        
        # Tracking config
        track_raw = raw.get('tracking', {})
        tracking = TrackingConfig(
            enabled=track_raw.get('enabled', True),
            algorithm=track_raw.get('algorithm', 'deepsort'),
            max_age=track_raw.get('max_age', 30),
            min_hits=track_raw.get('min_hits', 3),
            iou_threshold=track_raw.get('iou_threshold', 0.3)
        )
        
        # Video config
        vid_raw = raw.get('video', {})
        video = VideoConfig(
            source=str(vid_raw.get('source', '0')),
            width=vid_raw.get('width', 1280),
            height=vid_raw.get('height', 720),
            fps=vid_raw.get('fps', 30)
        )
        
        # Server config
        srv_raw = raw.get('server', {})
        server = ServerConfig(
            host=srv_raw.get('host', '0.0.0.0'),
            port=srv_raw.get('port', 8000),
            debug=srv_raw.get('debug', False)
        )
        
        # Visualization config
        vis_raw = raw.get('visualization', {})
        visualization = VisualizationConfig(
            show_fps=vis_raw.get('show_fps', True),
            show_track_id=vis_raw.get('show_track_id', True),
            show_confidence=vis_raw.get('show_confidence', True),
            bbox_thickness=vis_raw.get('bbox_thickness', 2),
            colors=vis_raw.get('colors', {
                'vehicle': [0, 255, 0],
                'selected': [0, 165, 255],
                'plate': [255, 0, 0]
            })
        )
        
        # Logging config
        log_raw = raw.get('logging', {})
        logging_config = LoggingConfig(
            level=log_raw.get('level', 'INFO'),
            save_detections=log_raw.get('save_detections', False),
            output_dir=log_raw.get('output_dir', 'logs')
        )
        
        return AppConfig(
            detection=detection,
            plate_recognition=plate,
            tracking=tracking,
            video=video,
            server=server,
            visualization=visualization,
            logging=logging_config
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by key path (e.g., 'detection.model')"""
        keys = key.split('.')
        value = self._raw_config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def save(self, path: Optional[str] = None):
        """Save current config to file"""
        save_path = Path(path) if path else self.config_path
        
        if save_path is None:
            raise ValueError("No config path specified")
        
        with open(save_path, 'w') as f:
            yaml.dump(self._raw_config, f, default_flow_style=False)


def load_config(path: str = "config.yaml") -> AppConfig:
    """
    Convenience function to load configuration.
    
    Args:
        path: Path to config file
        
    Returns:
        AppConfig object
    """
    loader = ConfigLoader(path)
    return loader.load()
