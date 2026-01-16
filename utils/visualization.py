"""
Visualization Utilities
Drawing bounding boxes, labels, trajectories, and overlays
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from models.detector import Detection
from models.tracker import Track


@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    # Colors (BGR format)
    vehicle_color: Tuple[int, int, int] = (0, 255, 0)      # Green
    selected_color: Tuple[int, int, int] = (0, 165, 255)   # Orange
    plate_color: Tuple[int, int, int] = (255, 0, 0)        # Blue
    text_color: Tuple[int, int, int] = (255, 255, 255)     # White
    trajectory_color: Tuple[int, int, int] = (0, 255, 255) # Yellow
    
    # Appearance
    bbox_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2
    
    # Display options
    show_confidence: bool = True
    show_track_id: bool = True
    show_class: bool = True
    show_fps: bool = True
    show_trajectory: bool = True


class Visualizer:
    """
    Visualization utilities for surveillance dashboard.
    Draws detections, tracks, plates, and overlays on frames.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            show_confidence: Whether to show confidence scores
            
        Returns:
            Frame with detections drawn
        """
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.config.vehicle_color
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.config.bbox_thickness)
            
            # Build label
            label_parts = []
            if self.config.show_class:
                label_parts.append(det.class_name)
            if show_confidence and self.config.show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                self._draw_label(output, label, (x1, y1), color)
        
        return output
    
    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        trajectories: Optional[Dict[int, List[Tuple[int, int]]]] = None
    ) -> np.ndarray:
        """
        Draw tracked objects on frame.
        
        Args:
            frame: Input frame
            tracks: List of tracks
            trajectories: Optional trajectory data per track ID
            
        Returns:
            Frame with tracks drawn
        """
        output = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            
            # Use different color for selected track
            if track.is_selected:
                color = self.config.selected_color
                thickness = self.config.bbox_thickness + 2
            else:
                color = self.config.vehicle_color
                thickness = self.config.bbox_thickness
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Draw selection indicator
            if track.is_selected:
                # Draw corner accents
                corner_len = 15
                # Top-left
                cv2.line(output, (x1, y1), (x1 + corner_len, y1), color, 3)
                cv2.line(output, (x1, y1), (x1, y1 + corner_len), color, 3)
                # Top-right
                cv2.line(output, (x2, y1), (x2 - corner_len, y1), color, 3)
                cv2.line(output, (x2, y1), (x2, y1 + corner_len), color, 3)
                # Bottom-left
                cv2.line(output, (x1, y2), (x1 + corner_len, y2), color, 3)
                cv2.line(output, (x1, y2), (x1, y2 - corner_len), color, 3)
                # Bottom-right
                cv2.line(output, (x2, y2), (x2 - corner_len, y2), color, 3)
                cv2.line(output, (x2, y2), (x2, y2 - corner_len), color, 3)
                
                # Add "TRACKING" label
                self._draw_label(output, "TRACKING", (x1, y1 - 25), color, bg_alpha=0.8)
            
            # Build label
            label_parts = []
            if self.config.show_track_id:
                label_parts.append(f"ID:{track.track_id}")
            if self.config.show_class:
                label_parts.append(track.class_name)
            if self.config.show_confidence:
                label_parts.append(f"{track.confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                self._draw_label(output, label, (x1, y1), color)
            
            # Draw trajectory
            if self.config.show_trajectory and trajectories and track.track_id in trajectories:
                traj = trajectories[track.track_id]
                self._draw_trajectory(output, traj, color if track.is_selected else self.config.trajectory_color)
        
        return output
    
    def draw_plate_recognition(
        self,
        frame: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int],
        plate_text: str,
        confidence: float
    ) -> np.ndarray:
        """
        Draw license plate recognition result.
        
        Args:
            frame: Input frame
            vehicle_bbox: Vehicle bounding box
            plate_text: Recognized plate text
            confidence: Recognition confidence
            
        Returns:
            Frame with plate annotation
        """
        output = frame.copy()
        x1, y1, x2, y2 = vehicle_bbox
        
        # Draw plate text below vehicle
        label = f"🔖 {plate_text}"
        label_y = y2 + 20
        
        self._draw_label(
            output, label, (x1, label_y),
            self.config.plate_color,
            bg_alpha=0.8
        )
        
        return output
    
    def draw_info_overlay(
        self,
        frame: np.ndarray,
        info: Dict[str, any]
    ) -> np.ndarray:
        """
        Draw information overlay on frame.
        
        Args:
            frame: Input frame
            info: Dictionary of info to display
            
        Returns:
            Frame with overlay
        """
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Semi-transparent background for info bar
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
        
        # Build info string
        info_parts = []
        if 'fps' in info:
            info_parts.append(f"FPS: {info['fps']:.1f}")
        if 'vehicles' in info:
            info_parts.append(f"Vehicles: {info['vehicles']}")
        if 'tracking' in info:
            info_parts.append(f"Tracking: {info['tracking']}")
        if 'plates' in info:
            info_parts.append(f"Plates: {info['plates']}")
        
        info_text = " | ".join(info_parts)
        
        cv2.putText(
            output, info_text, (10, 28),
            self.font, 0.7, self.config.text_color, 2
        )
        
        return output
    
    def draw_selection_cursor(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        size: int = 20
    ) -> np.ndarray:
        """
        Draw selection cursor at position.
        
        Args:
            frame: Input frame
            x, y: Cursor position
            size: Cursor size
            
        Returns:
            Frame with cursor
        """
        output = frame.copy()
        color = self.config.selected_color
        
        # Draw crosshair
        cv2.line(output, (x - size, y), (x + size, y), color, 2)
        cv2.line(output, (x, y - size), (x, y + size), color, 2)
        cv2.circle(output, (x, y), size // 2, color, 2)
        
        return output
    
    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        bg_alpha: float = 0.7
    ):
        """Draw a label with background"""
        x, y = position
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, self.font, self.config.font_scale, self.config.font_thickness
        )
        
        # Draw background
        padding = 4
        y_label = y - text_h - padding * 2
        
        if y_label < 0:
            y_label = y + text_h + padding * 2
        
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y_label),
            (x + text_w + padding * 2, y_label + text_h + padding * 2),
            color, -1
        )
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
        
        # Draw text
        cv2.putText(
            frame, label,
            (x + padding, y_label + text_h + padding),
            self.font, self.config.font_scale,
            self.config.text_color, self.config.font_thickness
        )
    
    def _draw_trajectory(
        self,
        frame: np.ndarray,
        points: List[Tuple[int, int]],
        color: Tuple[int, int, int]
    ):
        """Draw trajectory line"""
        if len(points) < 2:
            return
        
        # Draw line with fading effect
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(alpha * 3))
            
            pt1 = points[i - 1]
            pt2 = points[i]
            
            cv2.line(frame, pt1, pt2, color, thickness)
        
        # Draw dot at current position
        if points:
            cv2.circle(frame, points[-1], 5, color, -1)
    
    def create_detection_thumbnail(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        size: Tuple[int, int] = (100, 60)
    ) -> np.ndarray:
        """
        Create a thumbnail of a detection.
        
        Args:
            frame: Source frame
            bbox: Bounding box of detection
            size: Output thumbnail size
            
        Returns:
            Thumbnail image
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure valid bounds
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        crop = frame[y1:y2, x1:x2].copy()
        
        if crop.size == 0:
            return np.zeros((*size[::-1], 3), dtype=np.uint8)
        
        thumbnail = cv2.resize(crop, size)
        return thumbnail


def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """
    Encode frame to JPEG bytes for streaming.
    
    Args:
        frame: Input frame
        quality: JPEG quality (0-100)
        
    Returns:
        JPEG encoded bytes
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_params)
    return buffer.tobytes()


def encode_frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    """
    Encode frame to base64 string for web display.
    
    Args:
        frame: Input frame
        quality: JPEG quality
        
    Returns:
        Base64 encoded string
    """
    import base64
    jpeg_bytes = encode_frame_to_jpeg(frame, quality)
    return base64.b64encode(jpeg_bytes).decode('utf-8')
