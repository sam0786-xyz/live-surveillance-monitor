"""
Object Tracking Module using DeepSORT
Supports selection-based tracking for surveillance applications
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from deep_sort_realtime.deepsort_tracker import DeepSort


@dataclass
class Track:
    """Represents a tracked object"""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_name: str
    confidence: float
    is_confirmed: bool = False
    is_selected: bool = False
    age: int = 0
    hits: int = 0
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of track"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside track bbox"""
        x1, y1, x2, y2 = self.bbox
        return x1 <= x <= x2 and y1 <= y <= y2


class ObjectTracker:
    """
    DeepSORT-based object tracker with selection support.
    Enables click-to-track functionality for surveillance dashboard.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.3,
        nn_budget: Optional[int] = 100
    ):
        """
        Initialize the tracker.
        
        Args:
            max_age: Maximum frames to keep track without detection
            n_init: Number of consecutive detections before track is confirmed
            max_iou_distance: Maximum IOU distance for matching
            max_cosine_distance: Maximum cosine distance for appearance matching
            nn_budget: Budget for appearance feature storage
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget
        )
        
        # Selected track for focused tracking
        self.selected_track_id: Optional[int] = None
        
        # Track history for visualization
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.history_length = 50
        
        # Track metadata
        self.track_metadata: Dict[int, dict] = {}
    
    def update(
        self,
        detections: List,
        frame: np.ndarray
    ) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of Detection objects from detector
            frame: Current frame for appearance features
            
        Returns:
            List of Track objects
        """
        if not detections:
            # Update tracker with empty detections
            self.tracker.update_tracks([], frame=frame)
            return []
        
        # Convert detections to format expected by DeepSort
        # Format: [[x1, y1, x2, y2, confidence, class_id], ...]
        det_list = []
        class_names = {}
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # DeepSort expects [left, top, width, height]
            w = x2 - x1
            h = y2 - y1
            det_list.append(([x1, y1, w, h], det.confidence, det.class_name))
            class_names[det.class_name] = det.class_name
        
        # Update tracks
        tracks = self.tracker.update_tracks(det_list, frame=frame)
        
        # Convert to Track objects
        result = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Get class name from detection
            det_class = track.det_class if hasattr(track, 'det_class') else 'vehicle'
            
            t = Track(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                class_name=det_class if det_class else 'vehicle',
                confidence=track.det_conf if hasattr(track, 'det_conf') else 0.0,
                is_confirmed=track.is_confirmed(),
                is_selected=(track_id == self.selected_track_id),
                age=track.time_since_update if hasattr(track, 'time_since_update') else 0,
                hits=track.hits if hasattr(track, 'hits') else 0
            )
            
            # Update track history
            self._update_history(track_id, t.center)
            
            result.append(t)
        
        # Clean up old track histories
        self._cleanup_history(set(t.track_id for t in result))
        
        return result
    
    def select_track_at_point(
        self,
        x: int,
        y: int,
        tracks: List[Track]
    ) -> Optional[int]:
        """
        Select a track at the given point (click position).
        
        Args:
            x: X coordinate of click
            y: Y coordinate of click
            tracks: Current list of tracks
            
        Returns:
            Selected track ID or None if no track at point
        """
        for track in tracks:
            if track.contains_point(x, y):
                self.selected_track_id = track.track_id
                return track.track_id
        
        return None
    
    def select_track_by_id(self, track_id: int) -> bool:
        """
        Select a track by its ID.
        
        Args:
            track_id: ID of track to select
            
        Returns:
            True if track exists
        """
        self.selected_track_id = track_id
        return True
    
    def clear_selection(self):
        """Clear the current track selection"""
        self.selected_track_id = None
    
    def get_selected_track(self, tracks: List[Track]) -> Optional[Track]:
        """
        Get the currently selected track.
        
        Args:
            tracks: Current list of tracks
            
        Returns:
            Selected Track or None
        """
        if self.selected_track_id is None:
            return None
        
        for track in tracks:
            if track.track_id == self.selected_track_id:
                return track
        
        # Selected track no longer exists
        return None
    
    def get_track_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """
        Get the trajectory history for a track.
        
        Args:
            track_id: ID of the track
            
        Returns:
            List of (x, y) center points
        """
        return self.track_history.get(track_id, [])
    
    def _update_history(self, track_id: int, center: Tuple[int, int]):
        """Update trajectory history for a track"""
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        
        self.track_history[track_id].append(center)
        
        # Limit history length
        if len(self.track_history[track_id]) > self.history_length:
            self.track_history[track_id] = \
                self.track_history[track_id][-self.history_length:]
    
    def _cleanup_history(self, active_ids: set):
        """Remove history for tracks that no longer exist"""
        stale_ids = set(self.track_history.keys()) - active_ids
        for track_id in stale_ids:
            # Keep history for a bit in case track reappears
            if track_id in self.track_metadata:
                age = self.track_metadata[track_id].get('stale_age', 0) + 1
                if age > 30:  # Remove after 30 frames
                    del self.track_history[track_id]
                    del self.track_metadata[track_id]
                else:
                    self.track_metadata[track_id]['stale_age'] = age
            else:
                self.track_metadata[track_id] = {'stale_age': 1}
    
    def reset(self):
        """Reset the tracker state"""
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7
        )
        self.selected_track_id = None
        self.track_history.clear()
        self.track_metadata.clear()


class SimpleTracker:
    """
    Simple IOU-based tracker as fallback.
    Lighter weight than DeepSORT for resource-constrained scenarios.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1
        self.selected_track_id: Optional[int] = None
    
    def update(self, detections: List, frame: np.ndarray = None) -> List[Track]:
        """Update tracker with new detections using IOU matching"""
        if not detections:
            self._age_tracks()
            return self._get_active_tracks()
        
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)
        
        # Update matched tracks
        for det_idx, track_id in matched:
            det = detections[det_idx]
            self.tracks[track_id].update({
                'bbox': det.bbox,
                'confidence': det.confidence,
                'class_name': det.class_name,
                'age': 0,
                'hits': self.tracks[track_id]['hits'] + 1
            })
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self.tracks[self.next_id] = {
                'bbox': det.bbox,
                'confidence': det.confidence,
                'class_name': det.class_name,
                'age': 0,
                'hits': 1
            }
            self.next_id += 1
        
        # Age unmatched tracks
        for track_id in unmatched_tracks:
            self.tracks[track_id]['age'] += 1
        
        # Remove old tracks
        self._remove_old_tracks()
        
        return self._get_active_tracks()
    
    def _match_detections(self, detections):
        """Match detections to tracks using IOU"""
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        for det_idx, det in enumerate(detections):
            best_iou = 0
            best_track = None
            
            for track_id in unmatched_tracks:
                iou = self._compute_iou(det.bbox, self.tracks[track_id]['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track = track_id
            
            if best_track is not None:
                matched.append((det_idx, best_track))
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(best_track)
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _compute_iou(self, box1, box2):
        """Compute IOU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _age_tracks(self):
        """Increment age of all tracks"""
        for track_id in self.tracks:
            self.tracks[track_id]['age'] += 1
    
    def _remove_old_tracks(self):
        """Remove tracks that haven't been updated recently"""
        to_remove = [
            tid for tid, t in self.tracks.items()
            if t['age'] > self.max_age
        ]
        for tid in to_remove:
            del self.tracks[tid]
    
    def _get_active_tracks(self) -> List[Track]:
        """Get list of active Track objects"""
        return [
            Track(
                track_id=tid,
                bbox=t['bbox'],
                class_name=t['class_name'],
                confidence=t['confidence'],
                is_confirmed=t['hits'] >= 3,
                is_selected=(tid == self.selected_track_id),
                age=t['age'],
                hits=t['hits']
            )
            for tid, t in self.tracks.items()
        ]
    
    def select_track_at_point(self, x: int, y: int, tracks: List[Track]) -> Optional[int]:
        """Select track at click position"""
        for track in tracks:
            if track.contains_point(x, y):
                self.selected_track_id = track.track_id
                return track.track_id
        return None
    
    def clear_selection(self):
        """Clear selection"""
        self.selected_track_id = None
