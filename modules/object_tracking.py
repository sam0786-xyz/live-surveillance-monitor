"""
Object Tracking Pipeline Module
Integrates detection with tracking and selection-based tracking
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass
from enum import Enum

from models.detector import Detection
from models.tracker import ObjectTracker, Track, SimpleTracker


class TrackingMode(Enum):
    """Tracking mode options"""
    ALL = "all"           # Track all detected objects
    SELECTED = "selected" # Only track selected object
    NONE = "none"         # Disable tracking


@dataclass
class TrackingResult:
    """Result of tracking update"""
    tracks: List[Track]
    selected_track: Optional[Track]
    frame_id: int
    processing_time_ms: float
    
    @property
    def count(self) -> int:
        return len(self.tracks)
    
    @property
    def has_selection(self) -> bool:
        return self.selected_track is not None


class TrackingPipeline:
    """
    Complete tracking pipeline with selection support.
    
    Features:
    - Click-to-track: Select any detected object to focus on it
    - Visual trajectory: Show movement path of tracked object
    - Track management: Handle track creation, update, deletion
    - Event callbacks: Notify on track events (new, lost, selected)
    """
    
    def __init__(
        self,
        algorithm: str = "deepsort",
        max_age: int = 30,
        n_init: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize tracking pipeline.
        
        Args:
            algorithm: Tracking algorithm ("deepsort" or "simple")
            max_age: Maximum frames to keep track without detection
            n_init: Minimum detections before track is confirmed
            iou_threshold: IOU threshold for track association
        """
        if algorithm == "deepsort":
            self.tracker = ObjectTracker(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=1 - iou_threshold
            )
        else:
            self.tracker = SimpleTracker(
                iou_threshold=iou_threshold,
                max_age=max_age
            )
        
        self.mode = TrackingMode.ALL
        self.frame_count = 0
        
        # Callbacks
        self._on_track_new: Optional[Callable] = None
        self._on_track_lost: Optional[Callable] = None
        self._on_track_selected: Optional[Callable] = None
        
        # Track state for event detection
        self._previous_track_ids: set = set()
    
    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray
    ) -> TrackingResult:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections from detector
            frame: Current frame
            
        Returns:
            TrackingResult with tracks and metadata
        """
        import time
        start_time = time.time()
        
        self.frame_count += 1
        
        # Skip tracking if disabled
        if self.mode == TrackingMode.NONE:
            return TrackingResult(
                tracks=[],
                selected_track=None,
                frame_id=self.frame_count,
                processing_time_ms=0
            )
        
        # Update tracker
        tracks = self.tracker.update(detections, frame)
        
        # Get current track IDs
        current_track_ids = {t.track_id for t in tracks}
        
        # Detect new tracks
        new_tracks = current_track_ids - self._previous_track_ids
        if new_tracks and self._on_track_new:
            for track_id in new_tracks:
                track = next((t for t in tracks if t.track_id == track_id), None)
                if track:
                    self._on_track_new(track)
        
        # Detect lost tracks
        lost_tracks = self._previous_track_ids - current_track_ids
        if lost_tracks and self._on_track_lost:
            for track_id in lost_tracks:
                self._on_track_lost(track_id)
        
        self._previous_track_ids = current_track_ids
        
        # Get selected track
        selected_track = self.tracker.get_selected_track(tracks)
        
        # In SELECTED mode, filter to only selected track
        if self.mode == TrackingMode.SELECTED and selected_track:
            tracks = [selected_track]
        
        processing_time = (time.time() - start_time) * 1000
        
        return TrackingResult(
            tracks=tracks,
            selected_track=selected_track,
            frame_id=self.frame_count,
            processing_time_ms=processing_time
        )
    
    def select_at_point(
        self,
        x: int,
        y: int,
        tracks: List[Track]
    ) -> Optional[Track]:
        """
        Select a track at the given click position.
        
        Args:
            x: X coordinate of click
            y: Y coordinate of click
            tracks: Current list of tracks
            
        Returns:
            Selected track or None
        """
        track_id = self.tracker.select_track_at_point(x, y, tracks)
        
        if track_id is not None:
            selected = next((t for t in tracks if t.track_id == track_id), None)
            
            if selected and self._on_track_selected:
                self._on_track_selected(selected)
            
            return selected
        
        return None
    
    def select_by_id(self, track_id: int) -> bool:
        """Select a track by its ID"""
        return self.tracker.select_track_by_id(track_id)
    
    def clear_selection(self):
        """Clear current track selection"""
        self.tracker.clear_selection()
    
    def get_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """
        Get the trajectory (path) of a track.
        
        Args:
            track_id: Track ID
            
        Returns:
            List of (x, y) center points
        """
        return self.tracker.get_track_trajectory(track_id)
    
    def set_mode(self, mode: TrackingMode):
        """Set tracking mode"""
        self.mode = mode
    
    def reset(self):
        """Reset tracker state"""
        self.tracker.reset()
        self.frame_count = 0
        self._previous_track_ids.clear()
    
    # Event callbacks
    def on_track_new(self, callback: Callable[[Track], None]):
        """Register callback for new track events"""
        self._on_track_new = callback
    
    def on_track_lost(self, callback: Callable[[int], None]):
        """Register callback for lost track events"""
        self._on_track_lost = callback
    
    def on_track_selected(self, callback: Callable[[Track], None]):
        """Register callback for track selection events"""
        self._on_track_selected = callback
    
    def get_statistics(self, result: TrackingResult) -> dict:
        """Get tracking statistics"""
        return {
            'total_tracks': result.count,
            'confirmed_tracks': sum(1 for t in result.tracks if t.is_confirmed),
            'selected_track_id': result.selected_track.track_id if result.selected_track else None,
            'tracking_fps': round(1000 / result.processing_time_ms, 1) if result.processing_time_ms > 0 else 0,
            'mode': self.mode.value
        }


class MultiObjectTracker:
    """
    Advanced multi-object tracker with zone-based tracking.
    Supports tracking in specific regions of interest.
    """
    
    def __init__(self):
        self.tracker = ObjectTracker()
        self.zones: Dict[str, Tuple[int, int, int, int]] = {}
        self.zone_tracks: Dict[str, List[int]] = {}
    
    def add_zone(self, name: str, region: Tuple[int, int, int, int]):
        """Add a tracking zone"""
        self.zones[name] = region
        self.zone_tracks[name] = []
    
    def remove_zone(self, name: str):
        """Remove a tracking zone"""
        if name in self.zones:
            del self.zones[name]
            del self.zone_tracks[name]
    
    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray
    ) -> Dict[str, List[Track]]:
        """
        Update tracker and return tracks per zone.
        
        Returns:
            Dictionary mapping zone names to tracks in that zone
        """
        tracks = self.tracker.update(detections, frame)
        
        result = {'all': tracks}
        
        for zone_name, region in self.zones.items():
            zone_tracks = self._filter_tracks_in_region(tracks, region)
            result[zone_name] = zone_tracks
            self.zone_tracks[zone_name] = [t.track_id for t in zone_tracks]
        
        return result
    
    def _filter_tracks_in_region(
        self,
        tracks: List[Track],
        region: Tuple[int, int, int, int]
    ) -> List[Track]:
        """Filter tracks to those within a region"""
        rx1, ry1, rx2, ry2 = region
        
        return [
            t for t in tracks
            if rx1 <= t.center[0] <= rx2 and ry1 <= t.center[1] <= ry2
        ]
    
    def get_zone_entry_count(self, zone_name: str) -> int:
        """Get count of tracks that entered a zone"""
        return len(self.zone_tracks.get(zone_name, []))
