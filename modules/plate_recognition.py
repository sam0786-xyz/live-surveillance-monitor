"""
License Plate Recognition Pipeline
Two-stage: Plate Detection + OCR (EasyOCR)
Optimized for Indian License Plates
"""

import cv2
import numpy as np
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass
import easyocr

from models.detector import Detection, PlateDetector


@dataclass
class PlateRecognition:
    """Represents a recognized license plate"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # Plate bbox within vehicle crop
    vehicle_bbox: Tuple[int, int, int, int]  # Original vehicle bbox
    raw_text: str  # Unformatted OCR output
    
    @property
    def is_valid(self) -> bool:
        """Check if plate text matches expected Indian format"""
        # Indian plate format: XX 00 XX 0000 or XX00XX0000
        pattern = r'^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{4}$'
        return bool(re.match(pattern, self.text.replace(' ', '')))


class PlateRecognitionPipeline:
    """
    Complete license plate recognition pipeline.
    
    Pipeline stages:
    1. Receive vehicle crops from car detection
    2. Detect license plate region
    3. Preprocess plate image
    4. Run OCR
    5. Post-process and validate text
    """
    
    # Common OCR corrections for Indian plates
    CHAR_CORRECTIONS = {
        '0': 'O', 'O': '0',  # Context-dependent
        '1': 'I', 'I': '1',
        '5': 'S', 'S': '5',
        '8': 'B', 'B': '8',
        '6': 'G', 'G': '6',
    }
    
    # Indian state codes for validation
    INDIAN_STATE_CODES = [
        'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'GA',
        'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH',
        'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK',
        'TN', 'TS', 'TR', 'UK', 'UP', 'WB'
    ]
    
    def __init__(
        self,
        plate_model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
        device: str = "mps",
        ocr_languages: List[str] = ['en']
    ):
        """
        Initialize the plate recognition pipeline.
        
        Args:
            plate_model_path: Path to plate detection model (optional)
            confidence_threshold: Detection confidence threshold
            device: Inference device
            ocr_languages: Languages for EasyOCR
        """
        # Initialize plate detector
        self.plate_detector = PlateDetector(
            model_path=plate_model_path or "yolov8n.pt",
            confidence_threshold=confidence_threshold,
            device=device
        )
        
        # Initialize OCR reader
        # GPU=False for MPS compatibility (EasyOCR doesn't support MPS)
        self.ocr = easyocr.Reader(ocr_languages, gpu=False)
        
        self.confidence_threshold = confidence_threshold
        
        # Cache for recognized plates (avoid re-processing)
        self.plate_cache: dict = {}
    
    def recognize(
        self,
        vehicle_crop: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int]
    ) -> Optional[PlateRecognition]:
        """
        Recognize license plate from a vehicle crop.
        
        Args:
            vehicle_crop: Cropped image of vehicle
            vehicle_bbox: Original bounding box of vehicle in frame
            
        Returns:
            PlateRecognition object or None if no plate found
        """
        # Stage 1: Detect plate region
        plate_result = self.plate_detector.detect_plate(vehicle_crop)
        
        if plate_result is None:
            return None
        
        plate_bbox, plate_conf = plate_result
        
        # Stage 2: Extract and preprocess plate
        plate_crop = self._extract_plate(vehicle_crop, plate_bbox)
        preprocessed = self._preprocess_plate(plate_crop)
        
        # Stage 3: Run OCR
        ocr_results = self.ocr.readtext(preprocessed)
        
        if not ocr_results:
            return None
        
        # Stage 4: Post-process text
        raw_text = ' '.join([r[1] for r in ocr_results])
        processed_text = self._postprocess_text(raw_text)
        
        # Calculate overall confidence
        avg_conf = sum(r[2] for r in ocr_results) / len(ocr_results)
        overall_conf = (plate_conf + avg_conf) / 2
        
        return PlateRecognition(
            text=processed_text,
            confidence=overall_conf,
            bbox=plate_bbox,
            vehicle_bbox=vehicle_bbox,
            raw_text=raw_text
        )
    
    def recognize_batch(
        self,
        detections_with_crops: List[Tuple[Detection, np.ndarray]]
    ) -> List[Optional[PlateRecognition]]:
        """
        Recognize plates from multiple vehicle crops.
        
        Args:
            detections_with_crops: List of (Detection, crop) tuples
            
        Returns:
            List of PlateRecognition objects (None for failed recognitions)
        """
        results = []
        
        for det, crop in detections_with_crops:
            recognition = self.recognize(crop, det.bbox)
            results.append(recognition)
        
        return results
    
    def _extract_plate(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Extract plate region from image"""
        x1, y1, x2, y2 = bbox
        
        # Ensure valid bounds
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        return image[y1:y2, x1:x2].copy()
    
    def _preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for better OCR.
        
        Applies:
        - Resize to standard size
        - Grayscale conversion
        - Contrast enhancement
        - Noise reduction
        - Binarization
        """
        if plate_img.size == 0:
            return plate_img
        
        # Resize to standard size
        plate_img = cv2.resize(plate_img, (300, 100))
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Adaptive thresholding for binarization
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _postprocess_text(self, text: str) -> str:
        """
        Post-process OCR text to match Indian plate format.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned and formatted plate text
        """
        # Remove unwanted characters
        text = text.upper()
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        if len(text) < 4:
            return text
        
        # Try to format as Indian plate: XX 00 XX 0000
        # State code (2 letters) + District (2 digits) + Series (1-3 letters) + Number (4 digits)
        
        # Extract components
        state_code = text[:2]
        
        # Validate state code
        if state_code not in self.INDIAN_STATE_CODES:
            # Try to correct common OCR errors in state code
            state_code = self._correct_state_code(state_code)
        
        # Format the rest
        remaining = text[2:]
        
        # Try to identify district number (should be 2 digits)
        district = ''
        series = ''
        number = ''
        
        i = 0
        # Extract district (1-2 digits)
        while i < len(remaining) and remaining[i].isdigit() and len(district) < 2:
            district += remaining[i]
            i += 1
        
        # Extract series (1-3 letters)
        while i < len(remaining) and remaining[i].isalpha() and len(series) < 3:
            series += remaining[i]
            i += 1
        
        # Extract number (up to 4 digits)
        while i < len(remaining) and remaining[i].isdigit() and len(number) < 4:
            number += remaining[i]
            i += 1
        
        # Format with spaces
        formatted = f"{state_code} {district} {series} {number}".strip()
        formatted = re.sub(r'\s+', ' ', formatted)
        
        return formatted
    
    def _correct_state_code(self, code: str) -> str:
        """Try to correct common OCR errors in state code"""
        corrections = {
            'M4': 'MH', 'M#': 'MH',
            'D1': 'DL', 'Dl': 'DL',
            'K4': 'KA', 'K#': 'KA',
            'U9': 'UP', 'U?': 'UP',
        }
        return corrections.get(code, code)
    
    def clear_cache(self):
        """Clear the plate recognition cache"""
        self.plate_cache.clear()


class PlateRecognitionSimple:
    """
    Simplified plate recognition using only OCR.
    Use this when plate detection model is not available.
    """
    
    def __init__(self, ocr_languages: List[str] = ['en']):
        self.ocr = easyocr.Reader(ocr_languages, gpu=False)
    
    def recognize_from_vehicle(
        self,
        vehicle_crop: np.ndarray
    ) -> Optional[str]:
        """
        Try to recognize plate directly from vehicle crop.
        Searches for text regions that match plate pattern.
        """
        # Look for text in lower portion of vehicle (where plates usually are)
        h, w = vehicle_crop.shape[:2]
        lower_region = vehicle_crop[int(h*0.5):, :]
        
        # Run OCR
        results = self.ocr.readtext(lower_region)
        
        # Filter for plate-like text
        plate_pattern = r'[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}'
        
        for bbox, text, conf in results:
            cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
            if re.search(plate_pattern, cleaned):
                return cleaned
        
        return None
