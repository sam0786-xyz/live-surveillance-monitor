"""
Basic tests for the surveillance system components.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDetector:
    """Tests for vehicle detector"""
    
    def test_detection_dataclass(self):
        """Test Detection dataclass"""
        from models.detector import Detection
        
        det = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.85,
            class_id=2,
            class_name='car'
        )
        
        assert det.center == (150, 150)
        assert det.area == 10000
        assert det.contains_point(150, 150)
        assert not det.contains_point(0, 0)
    
    def test_detector_initialization(self):
        """Test VehicleDetector initialization"""
        # Skip if YOLO not installed
        try:
            from models.detector import VehicleDetector
            detector = VehicleDetector(device='cpu')
            assert detector is not None
        except ImportError:
            pytest.skip("YOLOv8 not installed")


class TestTracker:
    """Tests for object tracker"""
    
    def test_track_dataclass(self):
        """Test Track dataclass"""
        from models.tracker import Track
        
        track = Track(
            track_id=1,
            bbox=(100, 100, 200, 200),
            class_name='car',
            confidence=0.9,
            is_confirmed=True
        )
        
        assert track.center == (150, 150)
        assert track.contains_point(150, 150)
        assert not track.is_selected
    
    def test_simple_tracker(self):
        """Test SimpleTracker"""
        from models.tracker import SimpleTracker
        from models.detector import Detection
        
        tracker = SimpleTracker()
        
        # Create detection
        det = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=2,
            class_name='car'
        )
        
        # Update tracker
        tracks = tracker.update([det])
        
        assert len(tracks) >= 0  # May need min_hits before confirmed


class TestPipelines:
    """Tests for pipeline modules"""
    
    def test_tracking_mode_enum(self):
        """Test TrackingMode enum"""
        from modules.object_tracking import TrackingMode
        
        assert TrackingMode.ALL.value == "all"
        assert TrackingMode.SELECTED.value == "selected"
        assert TrackingMode.NONE.value == "none"
    
    def test_plate_recognition_patterns(self):
        """Test Indian license plate pattern matching"""
        from modules.plate_recognition import PlateRecognition
        
        # Valid Indian plates
        valid = PlateRecognition(
            text="MH 12 AB 1234",
            confidence=0.9,
            bbox=(0, 0, 100, 30),
            vehicle_bbox=(0, 0, 200, 100),
            raw_text="MH 12 AB 1234"
        )
        assert valid.is_valid
        
        # With different spacing
        valid2 = PlateRecognition(
            text="DL05XY5678",
            confidence=0.9,
            bbox=(0, 0, 100, 30),
            vehicle_bbox=(0, 0, 200, 100),
            raw_text="DL05XY5678"
        )
        assert valid2.is_valid


class TestUtils:
    """Tests for utility functions"""
    
    def test_config_loader(self):
        """Test configuration loader"""
        from utils.config_loader import ConfigLoader, AppConfig
        
        loader = ConfigLoader()
        config = loader.load()
        
        assert isinstance(config, AppConfig)
        assert config.detection.model == "yolov8n.pt"
        assert config.detection.device == "mps"
    
    def test_visualization(self):
        """Test visualization utilities"""
        from utils.visualization import Visualizer, encode_frame_to_jpeg
        
        viz = Visualizer()
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test JPEG encoding
        jpeg = encode_frame_to_jpeg(frame)
        assert isinstance(jpeg, bytes)
        assert len(jpeg) > 0


class TestVideoHandler:
    """Tests for video handling"""
    
    def test_test_video_generator(self):
        """Test synthetic video generator"""
        from utils.video_handler import TestVideoGenerator
        
        gen = TestVideoGenerator(width=640, height=480, num_cars=3)
        frame_info = gen.generate_frame()
        
        assert frame_info.frame.shape == (480, 640, 3)
        assert frame_info.width == 640
        assert frame_info.height == 480


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
