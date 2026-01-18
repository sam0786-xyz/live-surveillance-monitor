"""
YOLOv11-based Vehicle and Person Detection Module
Optimized for high-altitude drone surveillance (200-300m)
With image preprocessing for low-quality/small object detection
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from ultralytics import YOLO
import cv2


class ImagePreprocessor:
    """
    Image preprocessing for high-altitude drone footage.
    Enhances contrast, sharpens edges, and upscales small images.
    """
    
    def __init__(
        self,
        clahe_enabled: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        sharpening_enabled: bool = True,
        sharpening_amount: float = 1.0,
        upscaling_enabled: bool = True,
        min_dimension: int = 640,
        target_dimension: int = 1280
    ):
        self.clahe_enabled = clahe_enabled
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_size, clahe_tile_size)
        )
        self.sharpening_enabled = sharpening_enabled
        self.sharpening_amount = sharpening_amount
        self.upscaling_enabled = upscaling_enabled
        self.min_dimension = min_dimension
        self.target_dimension = target_dimension
        
        # Sharpening kernel
        self.sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
    
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply preprocessing pipeline to enhance image for detection.
        
        Returns:
            Tuple of (preprocessed_frame, scale_factor)
        """
        scale_factor = 1.0
        processed = frame.copy()
        
        # 1. Upscale small images
        if self.upscaling_enabled:
            h, w = processed.shape[:2]
            min_dim = min(h, w)
            if min_dim < self.min_dimension:
                scale = self.target_dimension / min_dim
                new_w = int(w * scale)
                new_h = int(h * scale)
                processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                scale_factor = scale
        
        # 2. Apply CLAHE contrast enhancement
        if self.clahe_enabled:
            # Convert to LAB color space
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            # Apply CLAHE to L channel
            l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 3. Apply sharpening
        if self.sharpening_enabled:
            # Blend original with sharpened
            sharpened = cv2.filter2D(processed, -1, self.sharpen_kernel)
            processed = cv2.addWeighted(
                processed, 1 - self.sharpening_amount * 0.3,
                sharpened, self.sharpening_amount * 0.3,
                0
            )
        
        return processed, scale_factor
    
    def scale_detections(
        self,
        detections: List[Any],
        scale_factor: float
    ) -> List[Any]:
        """Scale detection coordinates back to original image size."""
        if scale_factor == 1.0:
            return detections
        
        scaled = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            scaled_bbox = (
                int(x1 / scale_factor),
                int(y1 / scale_factor),
                int(x2 / scale_factor),
                int(y2 / scale_factor)
            )
            # Create new detection with scaled bbox
            from dataclasses import replace
            scaled.append(replace(det, bbox=scaled_bbox))
        return scaled


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
    YOLOv11-based vehicle and person detector
    Optimized for high-altitude drone footage (200-300m)
    """
    
    # COCO class IDs for vehicles and people
    COCO_CLASSES = {
        0: 'person',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(
        self,
        model_path: str = "yolo11l.pt",
        confidence_threshold: float = 0.15,
        iou_threshold: float = 0.45,
        device: str = "mps",
        classes: Optional[List[int]] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the detector with YOLOv11 and preprocessing.
        
        Args:
            model_path: Path to YOLOv11 model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run inference on (mps/cpu/cuda)
            classes: List of class IDs to detect
            preprocessing_config: Optional preprocessing settings
        """
        print(f"🚀 Loading YOLOv11 model: {model_path}")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes or [0, 2, 5, 7]  # person, car, bus, truck
        
        # Initialize preprocessor
        if preprocessing_config and preprocessing_config.get('enabled', False):
            clahe_cfg = preprocessing_config.get('clahe', {})
            sharp_cfg = preprocessing_config.get('sharpening', {})
            upscale_cfg = preprocessing_config.get('upscaling', {})
            
            self.preprocessor = ImagePreprocessor(
                clahe_enabled=clahe_cfg.get('enabled', True),
                clahe_clip_limit=clahe_cfg.get('clip_limit', 2.0),
                clahe_tile_size=clahe_cfg.get('tile_grid_size', 8),
                sharpening_enabled=sharp_cfg.get('enabled', True),
                sharpening_amount=sharp_cfg.get('amount', 1.0),
                upscaling_enabled=upscale_cfg.get('enabled', True),
                min_dimension=upscale_cfg.get('min_dimension', 640),
                target_dimension=upscale_cfg.get('target_dimension', 1280)
            )
            print("✅ Image preprocessing enabled (CLAHE, sharpening, upscaling)")
        else:
            self.preprocessor = None
        
        # Warm up the model
        self._warmup()
    
    def _warmup(self):
        """Warm up model with dummy inference"""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, device=self.device, verbose=False)
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect vehicles and people in a frame.
        Applies preprocessing if enabled for better small object detection.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of Detection objects found
        """
        # Apply preprocessing if enabled
        scale_factor = 1.0
        if self.preprocessor is not None:
            processed_frame, scale_factor = self.preprocessor.preprocess(frame)
        else:
            processed_frame = frame
        
        results = self.model(
            processed_frame,
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
                class_name = self.COCO_CLASSES.get(class_id, 'unknown')
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                ))
        
        # Scale detections back to original image coordinates if upscaled
        if scale_factor != 1.0 and self.preprocessor is not None:
            detections = self.preprocessor.scale_detections(detections, scale_factor)
        
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


class SAHIDetector:
    """
    SAHI (Slicing Aided Hyper Inference) wrapper for improved small object detection.
    Divides large images into overlapping tiles for better detection of tiny objects.
    Essential for high-altitude drone surveillance (200-300m).
    """
    
    def __init__(
        self,
        model_path: str = "yolo11l.pt",
        confidence_threshold: float = 0.15,
        device: str = "mps",
        classes: Optional[List[int]] = None,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        postprocess_type: str = "NMS",
        postprocess_match_threshold: float = 0.5
    ):
        """
        Initialize SAHI detector.
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Minimum confidence
            device: Inference device
            classes: Class IDs to detect
            slice_height: Height of each slice
            slice_width: Width of each slice
            overlap_height_ratio: Vertical overlap ratio
            overlap_width_ratio: Horizontal overlap ratio
            postprocess_type: NMS or GREEDYNMM
            postprocess_match_threshold: IoU threshold for merging
        """
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            self.get_sliced_prediction = get_sliced_prediction
            self.sahi_available = True
        except ImportError:
            print("⚠️ SAHI not installed. Using standard detection.")
            self.sahi_available = False
            
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.classes = classes or [0, 2, 5, 7]
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.postprocess_type = postprocess_type
        self.postprocess_match_threshold = postprocess_match_threshold
        
        # COCO class mapping
        self.COCO_CLASSES = {
            0: 'person',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        if self.sahi_available:
            print(f"🚀 Loading SAHI detector with {model_path}")
            from sahi import AutoDetectionModel
            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=device
            )
            print(f"✅ SAHI ready (slices: {slice_width}x{slice_height}, overlap: {overlap_width_ratio:.0%})")
        else:
            # Fallback to standard detector
            self.fallback_detector = VehicleDetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=device,
                classes=classes
            )
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects using SAHI sliced inference.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of Detection objects
        """
        if not self.sahi_available:
            return self.fallback_detector.detect(frame)
        
        # Run SAHI sliced prediction
        result = self.get_sliced_prediction(
            frame,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            postprocess_type=self.postprocess_type,
            postprocess_match_threshold=self.postprocess_match_threshold,
            verbose=0
        )
        
        detections = []
        
        for pred in result.object_prediction_list:
            class_id = pred.category.id
            
            # Filter by allowed classes
            if class_id not in self.classes:
                continue
            
            bbox = pred.bbox
            x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
            
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=pred.score.value,
                class_id=class_id,
                class_name=self.COCO_CLASSES.get(class_id, 'unknown')
            ))
        
        return detections
    
    def detect_with_crops(
        self,
        frame: np.ndarray
    ) -> List[Tuple[Detection, np.ndarray]]:
        """Detect and return crops for each detection."""
        detections = self.detect(frame)
        results = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # Add padding
            pad = 10
            h, w = frame.shape[:2]
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            crop = frame[y1:y2, x1:x2].copy()
            results.append((det, crop))
        
        return results

