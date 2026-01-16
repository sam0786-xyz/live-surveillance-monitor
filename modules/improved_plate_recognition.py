"""
Improved License Plate Recognition
Uses multiple detection strategies for better accuracy
"""

import cv2
import numpy as np
import re
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import easyocr


@dataclass
class PlateRecognition:
    """Represents a recognized license plate"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    vehicle_bbox: Tuple[int, int, int, int]
    raw_text: str
    
    @property
    def is_valid(self) -> bool:
        """Check if plate text looks like a valid plate"""
        # More permissive check - just needs some letters and numbers
        cleaned = re.sub(r'[^A-Z0-9]', '', self.text.upper())
        has_letters = bool(re.search(r'[A-Z]', cleaned))
        has_numbers = bool(re.search(r'[0-9]', cleaned))
        return len(cleaned) >= 4 and has_letters and has_numbers


class ImprovedPlateRecognizer:
    """
    Multi-strategy license plate recognition.
    Uses multiple approaches to maximize plate detection.
    """
    
    # Indian state codes
    INDIAN_STATE_CODES = [
        'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'GA',
        'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH',
        'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK',
        'TN', 'TS', 'TR', 'UK', 'UP', 'WB'
    ]
    
    def __init__(self, ocr_languages: List[str] = ['en']):
        """Initialize with EasyOCR"""
        print("   └─ Initializing EasyOCR...")
        self.ocr = easyocr.Reader(ocr_languages, gpu=False)
        print("   └─ EasyOCR ready!")
    
    def recognize(
        self,
        vehicle_crop: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int]
    ) -> Optional[PlateRecognition]:
        """
        Try multiple strategies to recognize plate.
        """
        if vehicle_crop.size == 0:
            return None
        
        results = []
        
        # Strategy 1: Direct OCR on full vehicle crop
        result = self._ocr_full_crop(vehicle_crop)
        if result:
            results.append(result)
        
        # Strategy 2: Focus on lower portion (where plates usually are)
        result = self._ocr_lower_region(vehicle_crop)
        if result:
            results.append(result)
        
        # Strategy 3: Find plate-like regions and OCR them
        result = self._ocr_plate_regions(vehicle_crop)
        if result:
            results.append(result)
        
        # Strategy 4: Enhanced preprocessing + OCR
        result = self._ocr_enhanced(vehicle_crop)
        if result:
            results.append(result)
        
        if not results:
            return None
        
        # Return best result (highest confidence)
        results.sort(key=lambda x: x['confidence'], reverse=True)
        best = results[0]
        
        return PlateRecognition(
            text=best['text'],
            confidence=best['confidence'],
            bbox=(0, 0, vehicle_crop.shape[1], vehicle_crop.shape[0]),
            vehicle_bbox=vehicle_bbox,
            raw_text=best['raw']
        )
    
    def _ocr_full_crop(self, image: np.ndarray) -> Optional[Dict]:
        """Run OCR on the full vehicle crop"""
        try:
            results = self.ocr.readtext(image, detail=1, paragraph=False)
            return self._filter_plate_results(results)
        except Exception as e:
            print(f"   └─ OCR full crop error: {e}")
            return None
    
    def _ocr_lower_region(self, image: np.ndarray) -> Optional[Dict]:
        """Focus on lower 50% where plates usually are"""
        try:
            h = image.shape[0]
            lower = image[int(h * 0.4):, :]
            results = self.ocr.readtext(lower, detail=1, paragraph=False)
            return self._filter_plate_results(results)
        except Exception as e:
            return None
    
    def _ocr_plate_regions(self, image: np.ndarray) -> Optional[Dict]:
        """Find rectangular regions and OCR them"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter
            blur = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Edge detection
            edges = cv2.Canny(blur, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular contours (plates)
            h, w = image.shape[:2]
            candidates = []
            
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Look for 4-sided contours
                if len(approx) >= 4:
                    x, y, cw, ch = cv2.boundingRect(contour)
                    aspect = cw / ch if ch > 0 else 0
                    area_ratio = (cw * ch) / (w * h)
                    
                    # License plate aspect ratio is typically between 2:1 to 5:1
                    if 1.5 <= aspect <= 6.0 and 0.01 <= area_ratio <= 0.3:
                        candidates.append((x, y, cw, ch, area_ratio))
            
            # OCR each candidate
            for x, y, cw, ch, _ in candidates[:5]:
                # Add padding
                pad = 5
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + cw + pad)
                y2 = min(h, y + ch + pad)
                
                region = image[y1:y2, x1:x2]
                if region.size > 0:
                    results = self.ocr.readtext(region, detail=1, paragraph=False)
                    plate = self._filter_plate_results(results)
                    if plate:
                        return plate
            
            return None
        except Exception as e:
            return None
    
    def _ocr_enhanced(self, image: np.ndarray) -> Optional[Dict]:
        """Apply enhancement before OCR"""
        try:
            # Resize for better OCR
            scale = 2.0
            resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
            
            results = self.ocr.readtext(denoised, detail=1, paragraph=False)
            return self._filter_plate_results(results)
        except Exception as e:
            return None
    
    def _filter_plate_results(self, ocr_results: List) -> Optional[Dict]:
        """Filter OCR results to find plate-like text"""
        if not ocr_results:
            return None
        
        best_plate = None
        best_score = 0
        
        for bbox, text, conf in ocr_results:
            # Clean the text
            cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            if len(cleaned) < 4:
                continue
            
            # Score based on plate-like characteristics
            score = conf
            
            # Bonus for having state code
            if len(cleaned) >= 2:
                possible_state = cleaned[:2]
                if possible_state in self.INDIAN_STATE_CODES:
                    score += 0.3
            
            # Bonus for plate-like length (9-10 chars typical for Indian)
            if 8 <= len(cleaned) <= 11:
                score += 0.2
            
            # Bonus for having numbers
            if re.search(r'\d{4}', cleaned):
                score += 0.2
            
            # Check if better than current best
            if score > best_score and score > 0.3:
                best_score = score
                best_plate = {
                    'text': self._format_plate(cleaned),
                    'raw': text,
                    'confidence': min(score, 1.0)
                }
        
        return best_plate
    
    def _format_plate(self, text: str) -> str:
        """Format plate text with proper spacing"""
        text = text.upper()
        
        # Try to format as: XX 00 XX 0000
        if len(text) >= 9:
            # State code (2 letters)
            state = text[:2]
            remaining = text[2:]
            
            # District (1-2 digits)
            district = ''
            i = 0
            while i < len(remaining) and remaining[i].isdigit() and len(district) < 2:
                district += remaining[i]
                i += 1
            
            # Series (1-3 letters)
            series = ''
            while i < len(remaining) and remaining[i].isalpha() and len(series) < 3:
                series += remaining[i]
                i += 1
            
            # Number (up to 4 digits)
            number = ''
            while i < len(remaining) and remaining[i].isdigit() and len(number) < 4:
                number += remaining[i]
                i += 1
            
            return f"{state} {district} {series} {number}".strip()
        
        return text


# Simplified initialization for the server
def create_plate_recognizer():
    """Factory function to create plate recognizer"""
    return ImprovedPlateRecognizer(ocr_languages=['en'])
