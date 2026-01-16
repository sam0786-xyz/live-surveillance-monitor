"""
Video Input Handler
Supports webcam, video files, and RTSP streams
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time


class VideoSource(Enum):
    """Video source types"""
    WEBCAM = "webcam"
    FILE = "file"
    RTSP = "rtsp"
    HTTP = "http"


@dataclass
class FrameInfo:
    """Information about a video frame"""
    frame: np.ndarray
    frame_number: int
    timestamp: float
    width: int
    height: int
    fps: float


class VideoHandler:
    """
    Video input handler with support for multiple sources.
    Provides buffered reading for smooth playback.
    """
    
    def __init__(
        self,
        source: str = "0",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        buffer_size: int = 5
    ):
        """
        Initialize video handler.
        
        Args:
            source: Video source (0 for webcam, path for file, URL for stream)
            width: Desired frame width
            height: Desired frame height
            fps: Desired FPS
            buffer_size: Number of frames to buffer
        """
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        self.buffer_size = buffer_size
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.source_type = self._detect_source_type()
        
        self.frame_count = 0
        self.actual_fps = 0
        self.last_frame_time = 0
        
        # Threading for buffered reading
        self._buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def _detect_source_type(self) -> VideoSource:
        """Detect the type of video source"""
        if self.source.isdigit() or self.source == "0":
            return VideoSource.WEBCAM
        elif self.source.startswith("rtsp://"):
            return VideoSource.RTSP
        elif self.source.startswith("http"):
            return VideoSource.HTTP
        else:
            return VideoSource.FILE
    
    def open(self) -> bool:
        """
        Open the video source.
        
        Returns:
            True if successful
        """
        try:
            # Convert source to int for webcam
            source = int(self.source) if self.source.isdigit() else self.source
            
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                return False
            
            # Set properties for webcam
            if self.source_type == VideoSource.WEBCAM:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Get actual properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS) or self.target_fps
            
            return True
            
        except Exception as e:
            print(f"Error opening video source: {e}")
            return False
    
    def close(self):
        """Close the video source"""
        self.stop_buffered_reading()
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def read(self) -> Optional[FrameInfo]:
        """
        Read a single frame.
        
        Returns:
            FrameInfo or None if read failed
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate actual FPS
        if self.last_frame_time > 0:
            time_diff = current_time - self.last_frame_time
            if time_diff > 0:
                self.actual_fps = 0.9 * self.actual_fps + 0.1 * (1.0 / time_diff)
        
        self.last_frame_time = current_time
        
        return FrameInfo(
            frame=frame,
            frame_number=self.frame_count,
            timestamp=current_time,
            width=frame.shape[1],
            height=frame.shape[0],
            fps=self.actual_fps
        )
    
    def read_frames(self) -> Generator[FrameInfo, None, None]:
        """
        Generator that yields frames.
        
        Yields:
            FrameInfo for each frame
        """
        while True:
            frame_info = self.read()
            if frame_info is None:
                break
            yield frame_info
    
    def start_buffered_reading(self):
        """Start background thread for buffered frame reading"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._buffer_frames, daemon=True)
        self._thread.start()
    
    def stop_buffered_reading(self):
        """Stop buffered reading thread"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        
        # Clear buffer
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                break
    
    def _buffer_frames(self):
        """Background thread for buffering frames"""
        while self._running:
            frame_info = self.read()
            
            if frame_info is None:
                if self.source_type == VideoSource.FILE:
                    # Loop video file
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_count = 0
                    continue
                else:
                    break
            
            try:
                self._buffer.put(frame_info, timeout=0.1)
            except queue.Full:
                # Drop oldest frame if buffer is full
                try:
                    self._buffer.get_nowait()
                    self._buffer.put_nowait(frame_info)
                except (queue.Empty, queue.Full):
                    pass
    
    def read_buffered(self, timeout: float = 0.1) -> Optional[FrameInfo]:
        """
        Read a frame from the buffer.
        
        Args:
            timeout: Maximum time to wait for frame
            
        Returns:
            FrameInfo or None
        """
        try:
            return self._buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_total_frames(self) -> int:
        """Get total frame count (for video files)"""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def seek(self, frame_number: int) -> bool:
        """
        Seek to a specific frame (video files only).
        
        Args:
            frame_number: Frame number to seek to
            
        Returns:
            True if successful
        """
        if self.cap is None or self.source_type != VideoSource.FILE:
            return False
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.frame_count = frame_number
        return True
    
    def get_properties(self) -> dict:
        """Get video properties"""
        return {
            'source': self.source,
            'source_type': self.source_type.value,
            'width': self.width,
            'height': self.height,
            'fps': self.actual_fps,
            'frame_count': self.frame_count,
            'total_frames': self.get_total_frames()
        }


class TestVideoGenerator:
    """
    Generate test video frames for development/testing.
    Creates synthetic traffic scenes.
    """
    
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        num_cars: int = 5
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.num_cars = num_cars
        
        self.frame_count = 0
        self.cars = self._initialize_cars()
    
    def _initialize_cars(self) -> list:
        """Initialize random cars"""
        cars = []
        for i in range(self.num_cars):
            car = {
                'x': np.random.randint(0, self.width - 100),
                'y': np.random.randint(0, self.height - 60),
                'w': np.random.randint(80, 150),
                'h': np.random.randint(40, 80),
                'vx': np.random.randint(-5, 5),
                'vy': np.random.randint(-3, 3),
                'color': tuple(map(int, np.random.randint(100, 255, 3)))
            }
            cars.append(car)
        return cars
    
    def _update_cars(self):
        """Update car positions"""
        for car in self.cars:
            car['x'] += car['vx']
            car['y'] += car['vy']
            
            # Bounce off edges
            if car['x'] < 0 or car['x'] + car['w'] > self.width:
                car['vx'] *= -1
            if car['y'] < 0 or car['y'] + car['h'] > self.height:
                car['vy'] *= -1
            
            # Clamp positions
            car['x'] = max(0, min(car['x'], self.width - car['w']))
            car['y'] = max(0, min(car['y'], self.height - car['h']))
    
    def generate_frame(self) -> FrameInfo:
        """Generate a test frame with moving cars"""
        # Create background (road-like)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Gray background
        
        # Draw road lines
        for y in range(0, self.height, 100):
            cv2.line(frame, (0, y), (self.width, y), (80, 80, 80), 2)
        
        # Draw cars
        for i, car in enumerate(self.cars):
            x, y, w, h = car['x'], car['y'], car['w'], car['h']
            color = car['color']
            
            # Draw car body
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
            
            # Draw windows
            win_x, win_y = x + w // 4, y + h // 6
            win_w, win_h = w // 2, h // 3
            cv2.rectangle(frame, (win_x, win_y), (win_x + win_w, win_y + win_h), (100, 100, 100), -1)
            
            # Draw "license plate"
            plate_x, plate_y = x + w // 4, y + h - 15
            plate_w = w // 2
            cv2.rectangle(frame, (plate_x, plate_y), (plate_x + plate_w, plate_y + 10), (255, 255, 255), -1)
            cv2.putText(frame, f"MH{i:02d}A{i:04d}", (plate_x + 2, plate_y + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
        
        # Update positions for next frame
        self._update_cars()
        self.frame_count += 1
        
        return FrameInfo(
            frame=frame,
            frame_number=self.frame_count,
            timestamp=time.time(),
            width=self.width,
            height=self.height,
            fps=self.fps
        )
    
    def generate_frames(self, count: int = -1) -> Generator[FrameInfo, None, None]:
        """Generate multiple test frames"""
        generated = 0
        while count < 0 or generated < count:
            yield self.generate_frame()
            generated += 1
            time.sleep(1.0 / self.fps)
