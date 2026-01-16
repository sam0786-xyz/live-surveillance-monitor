"""
Car Detection Pipeline Module
Integrates YOLOv8 detection with preprocessing and postprocessing
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from models.detector import VehicleDetector, Detection


@dataclass
class CarDetectionResult:
    """Result of car detection with additional metadata"""
    detections: List[Detection]
    frame_id: int
    processing_time_ms: float
    
    @property
    def count(self) -> int:
        return len(self.detections)


class CarDetectionPipeline:
    """
    Complete car detection pipeline for drone footage.
    Handles preprocessing, detection, and postprocessing.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "mps",
        input_size: Tuple[int, int] = (640, 640),
        enable_preprocessing: bool = True
    ):
        """
        Initialize the car detection pipeline.
        
        Args:
            model_path: Path to YOLOv8 model
            confidence_threshold: Detection confidence threshold
            device: Inference device (mps/cpu/cuda)
            input_size: Model input size
            enable_preprocessing: Enable image preprocessing
        """
        self.detector = VehicleDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )
        self.input_size = input_size
        self.enable_preprocessing = enable_preprocessing
        self.frame_count = 0
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better detection.
        Optimized for drone footage characteristics.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Preprocessed frame
        """
        if not self.enable_preprocessing:
            return frame
        
        # Apply CLAHE for better contrast (helps with shadows/highlights)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Slight denoising for aerial footage
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)
        
        return enhanced
    
    def detect(
        self,
        frame: np.ndarray,
        preprocess: bool = True
    ) -> CarDetectionResult:
        """
        Run car detection on a frame.
        
        Args:
            frame: Input BGR frame
            preprocess: Whether to apply preprocessing
            
        Returns:
            CarDetectionResult with detections and metadata
        """
        import time
        start_time = time.time()
        
        self.frame_count += 1
        
        # Preprocess if enabled
        if preprocess and self.enable_preprocessing:
            processed_frame = self.preprocess(frame)
        else:
            processed_frame = frame
        
        # Run detection
        detections = self.detector.detect(processed_frame)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return CarDetectionResult(
            detections=detections,
            frame_id=self.frame_count,
            processing_time_ms=processing_time
        )
    
    def detect_with_crops(
        self,
        frame: np.ndarray
    ) -> List[Tuple[Detection, np.ndarray]]:
        """
        Detect cars and return cropped images for each detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of (Detection, crop) tuples
        """
        return self.detector.detect_with_crops(frame)
    
    def filter_by_size(
        self,
        detections: List[Detection],
        min_area: int = 1000,
        max_area: int = 500000
    ) -> List[Detection]:
        """
        Filter detections by bounding box area.
        
        Args:
            detections: List of detections
            min_area: Minimum bounding box area
            max_area: Maximum bounding box area
            
        Returns:
            Filtered list of detections
        """
        return [
            det for det in detections
            if min_area <= det.area <= max_area
        ]
    
    def filter_by_region(
        self,
        detections: List[Detection],
        region: Tuple[int, int, int, int]
    ) -> List[Detection]:
        """
        Filter detections to only include those within a region.
        
        Args:
            detections: List of detections
            region: (x1, y1, x2, y2) region of interest
            
        Returns:
            Filtered list of detections
        """
        rx1, ry1, rx2, ry2 = region
        
        def in_region(det: Detection) -> bool:
            cx, cy = det.center
            return rx1 <= cx <= rx2 and ry1 <= cy <= ry2
        
        return [det for det in detections if in_region(det)]
    
    def get_statistics(self, result: CarDetectionResult) -> dict:
        """
        Get statistics from detection result.
        
        Args:
            result: Detection result
            
        Returns:
            Dictionary of statistics
        """
        if not result.detections:
            return {
                'total': 0,
                'cars': 0,
                'trucks': 0,
                'buses': 0,
                'avg_confidence': 0,
                'fps': 1000 / result.processing_time_ms if result.processing_time_ms > 0 else 0
            }
        
        cars = sum(1 for d in result.detections if d.class_name == 'car')
        trucks = sum(1 for d in result.detections if d.class_name == 'truck')
        buses = sum(1 for d in result.detections if d.class_name == 'bus')
        avg_conf = sum(d.confidence for d in result.detections) / len(result.detections)
        
        return {
            'total': len(result.detections),
            'cars': cars,
            'trucks': trucks,
            'buses': buses,
            'avg_confidence': round(avg_conf, 2),
            'fps': round(1000 / result.processing_time_ms, 1) if result.processing_time_ms > 0 else 0
        }
