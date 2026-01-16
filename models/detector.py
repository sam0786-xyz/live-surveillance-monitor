"""
YOLOv8-based Vehicle and License Plate Detection Module
Optimized for Apple Silicon (M4) with MPS acceleration
"""

import numpy as np
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from ultralytics import YOLO
import cv2


@dataclass
class Detection:
    """Represents a single detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        """Get area of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside bounding box"""
        x1, y1, x2, y2 = self.bbox
        return x1 <= x <= x2 and y1 <= y <= y2


class VehicleDetector:
    """
    YOLOv8-based vehicle detector
    Detects cars, buses, trucks from drone footage
    """
    
    # COCO class IDs for vehicles
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "mps",
        classes: Optional[List[int]] = None
    ):
        """
        Initialize the vehicle detector.
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run inference on (mps/cpu/cuda)
            classes: List of class IDs to detect (default: cars, buses, trucks)
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes or [2, 5, 7]  # car, bus, truck
        
        # Warm up the model
        self._warmup()
    
    def _warmup(self):
        """Warm up model with dummy inference"""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, device=self.device, verbose=False)
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect vehicles in a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of Detection objects for vehicles found
        """
        results = self.model(
            frame,
            device=self.device,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False
        )[0]
        
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.VEHICLE_CLASSES.get(class_id, 'vehicle')
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                ))
        
        return detections
    
    def detect_with_crops(
        self,
        frame: np.ndarray
    ) -> List[Tuple[Detection, np.ndarray]]:
        """
        Detect vehicles and return crops for each detection.
        Useful for license plate detection pipeline.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of (Detection, cropped_image) tuples
        """
        detections = self.detect(frame)
        results = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # Add padding for better plate detection
            pad = 10
            h, w = frame.shape[:2]
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            crop = frame[y1:y2, x1:x2].copy()
            results.append((det, crop))
        
        return results


class PlateDetector:
    """
    License plate detector using YOLOv8
    Can use pre-trained plate detection model or fine-tuned weights
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        device: str = "mps"
    ):
        """
        Initialize plate detector.
        
        Args:
            model_path: Path to plate detection model
            confidence_threshold: Minimum confidence threshold
            device: Inference device
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # For general YOLO, we'll detect plates as objects
        # If using custom plate model, adjust accordingly
        self.use_custom_model = 'plate' in model_path.lower()
    
    def detect_plate(
        self,
        vehicle_crop: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect license plate in a vehicle crop.
        
        Args:
            vehicle_crop: Cropped image of vehicle
            
        Returns:
            Tuple of (bbox, confidence) or None if no plate found
        """
        if self.use_custom_model:
            # Use custom plate detection model
            results = self.model(
                vehicle_crop,
                device=self.device,
                conf=self.confidence_threshold,
                verbose=False
            )[0]
            
            if results.boxes is not None and len(results.boxes) > 0:
                box = results.boxes[0]  # Take highest confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                return ((x1, y1, x2, y2), confidence)
        else:
            # Fallback: Use image processing to find plate region
            return self._detect_plate_cv(vehicle_crop)
        
        return None
    
    def _detect_plate_cv(
        self,
        image: np.ndarray
    ) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect plate using traditional CV methods (fallback).
        Uses edge detection and contour analysis.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Edge detection
        edges = cv2.Canny(gray, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by aspect ratio (plates are typically 2:1 to 5:1)
        candidates = []
        h, w = image.shape[:2]
        
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / ch if ch > 0 else 0
            area_ratio = (cw * ch) / (w * h)
            
            # Plate-like aspect ratio and reasonable size
            if 1.5 <= aspect_ratio <= 6 and 0.01 <= area_ratio <= 0.3:
                candidates.append((x, y, x + cw, y + ch, area_ratio))
        
        if candidates:
            # Return largest candidate
            candidates.sort(key=lambda x: x[4], reverse=True)
            x1, y1, x2, y2, _ = candidates[0]
            return ((x1, y1, x2, y2), 0.7)  # Fixed confidence for CV method
        
        return None
    
    def extract_plate_crop(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract and preprocess plate region for OCR.
        
        Args:
            image: Source image
            bbox: Plate bounding box
            
        Returns:
            Preprocessed plate image
        """
        x1, y1, x2, y2 = bbox
        plate = image[y1:y2, x1:x2].copy()
        
        # Resize for consistent OCR input
        plate = cv2.resize(plate, (300, 100))
        
        return plate
