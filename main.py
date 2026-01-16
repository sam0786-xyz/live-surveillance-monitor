"""
Drone Surveillance System - Main Entry Point
Integrates all components: detection, tracking, plate recognition, and dashboard

Usage:
    # Test mode with webcam
    python main.py --mode test
    
    # With video file
    python main.py --source path/to/video.mp4
    
    # With drone stream
    python main.py --source rtsp://drone-ip:port/stream
    
    # Dashboard only (for development)
    python main.py --dashboard-only
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional
import threading
import queue

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config, AppConfig
from utils.video_handler import VideoHandler, TestVideoGenerator, FrameInfo
from utils.visualization import Visualizer, encode_frame_to_base64
from models.detector import VehicleDetector, Detection
from models.tracker import ObjectTracker, Track
from modules.car_detection import CarDetectionPipeline
from modules.plate_recognition import PlateRecognitionPipeline
from modules.object_tracking import TrackingPipeline, TrackingMode


class SurveillancePipeline:
    """
    Main surveillance pipeline integrating all components.
    """
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.is_running = False
        
        # Initialize components
        print("🔧 Initializing components...")
        
        # Car detection
        print("  └─ Loading car detection model...")
        self.car_detector = CarDetectionPipeline(
            model_path=config.detection.model,
            confidence_threshold=config.detection.confidence_threshold,
            device=config.detection.device,
            enable_preprocessing=True
        )
        
        # Plate recognition (optional)
        self.plate_recognizer = None
        if config.plate_recognition.enabled:
            print("  └─ Loading plate recognition model...")
            try:
                self.plate_recognizer = PlateRecognitionPipeline(
                    confidence_threshold=config.plate_recognition.confidence_threshold,
                    device=config.detection.device
                )
            except Exception as e:
                print(f"  ⚠️  Plate recognition disabled: {e}")
        
        # Object tracking
        print("  └─ Initializing tracker...")
        self.tracker = TrackingPipeline(
            algorithm=config.tracking.algorithm,
            max_age=config.tracking.max_age,
            n_init=config.tracking.min_hits,
            iou_threshold=config.tracking.iou_threshold
        )
        
        # Visualization
        self.visualizer = Visualizer()
        
        # Video handler
        self.video_handler = None
        
        # Stats
        self.stats = {
            'fps': 0,
            'vehicles': 0,
            'plates': 0,
            'tracking': None
        }
        
        # Recognized plates cache
        self.recognized_plates = []
        
        # Frame queue for async processing
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        print("✅ Pipeline initialized!")
    
    def set_video_source(self, source: str):
        """Set the video source"""
        self.video_handler = VideoHandler(
            source=source,
            width=self.config.video.width,
            height=self.config.video.height,
            fps=self.config.video.fps
        )
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame through the pipeline.
        
        Returns:
            dict with processed frame and metadata
        """
        start_time = time.time()
        
        # Step 1: Car Detection
        detection_result = self.car_detector.detect(frame, preprocess=False)
        detections = detection_result.detections
        
        # Step 2: Object Tracking
        tracking_result = self.tracker.update(detections, frame)
        tracks = tracking_result.tracks
        selected_track = tracking_result.selected_track
        
        # Step 3: Plate Recognition (for new vehicles)
        plates_found = []
        if self.plate_recognizer and detections:
            for det, crop in self.car_detector.detect_with_crops(frame):
                if crop.size > 0:
                    try:
                        plate = self.plate_recognizer.recognize(crop, det.bbox)
                        if plate and plate.is_valid:
                            plates_found.append({
                                'text': plate.text,
                                'confidence': plate.confidence,
                                'vehicle_bbox': list(det.bbox)
                            })
                    except Exception:
                        pass
        
        # Update recognized plates
        for plate in plates_found:
            if plate['text'] not in [p['text'] for p in self.recognized_plates[-20:]]:
                self.recognized_plates.append(plate)
        
        # Step 4: Visualization
        # Get trajectories for visualization
        trajectories = {}
        for track in tracks:
            traj = self.tracker.get_trajectory(track.track_id)
            if traj:
                trajectories[track.track_id] = traj
        
        # Draw tracks on frame
        output_frame = self.visualizer.draw_tracks(frame, tracks, trajectories)
        
        # Draw plate recognition results
        for plate in plates_found:
            output_frame = self.visualizer.draw_plate_recognition(
                output_frame,
                tuple(plate['vehicle_bbox']),
                plate['text'],
                plate['confidence']
            )
        
        # Draw info overlay
        processing_time = (time.time() - start_time) * 1000
        fps = 1000 / processing_time if processing_time > 0 else 0
        
        self.stats = {
            'fps': fps,
            'vehicles': len(tracks),
            'plates': len(self.recognized_plates),
            'tracking': selected_track.track_id if selected_track else None
        }
        
        output_frame = self.visualizer.draw_info_overlay(output_frame, self.stats)
        
        return {
            'frame': output_frame,
            'detections': [{'bbox': list(d.bbox), 'confidence': d.confidence, 'class': d.class_name} for d in detections],
            'tracks': [{'track_id': t.track_id, 'bbox': list(t.bbox), 'class': t.class_name, 'selected': t.is_selected} for t in tracks],
            'plates': plates_found,
            'stats': self.stats
        }
    
    def handle_click(self, x: int, y: int, tracks: list) -> Optional[int]:
        """Handle click for track selection"""
        from models.tracker import Track
        
        track_objects = [
            Track(
                track_id=t['track_id'],
                bbox=tuple(t['bbox']),
                class_name=t.get('class', 'vehicle'),
                confidence=0.0,
                is_confirmed=True
            )
            for t in tracks
        ]
        
        selected = self.tracker.select_at_point(x, y, track_objects)
        return selected.track_id if selected else None
    
    def clear_selection(self):
        """Clear track selection"""
        self.tracker.clear_selection()
    
    def run_local(self, source: str = "0"):
        """
        Run the pipeline locally with OpenCV window.
        For testing without dashboard.
        """
        self.set_video_source(source)
        
        if not self.video_handler.open():
            print("❌ Failed to open video source")
            return
        
        print(f"📹 Video source opened: {self.video_handler.get_properties()}")
        print("Press 'q' to quit, click to select object for tracking")
        
        self.is_running = True
        
        # Mouse callback for selection
        selected_point = [None]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_point[0] = (x, y)
        
        cv2.namedWindow("Drone Surveillance")
        cv2.setMouseCallback("Drone Surveillance", mouse_callback)
        
        last_tracks = []
        
        try:
            for frame_info in self.video_handler.read_frames():
                if not self.is_running:
                    break
                
                # Handle click selection
                if selected_point[0]:
                    x, y = selected_point[0]
                    self.handle_click(x, y, last_tracks)
                    selected_point[0] = None
                
                # Process frame
                result = self.process_frame(frame_info.frame)
                last_tracks = result['tracks']
                
                # Display
                cv2.imshow("Drone Surveillance", result['frame'])
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.clear_selection()
                elif key == ord('s'):
                    cv2.imwrite(f"screenshot_{int(time.time())}.jpg", result['frame'])
                    print("📸 Screenshot saved!")
        
        finally:
            self.is_running = False
            self.video_handler.close()
            cv2.destroyAllWindows()
    
    def run_with_test_video(self):
        """Run with synthetic test video"""
        print("🎬 Running with test video generator...")
        
        generator = TestVideoGenerator(
            width=1280,
            height=720,
            fps=30,
            num_cars=8
        )
        
        self.is_running = True
        
        cv2.namedWindow("Drone Surveillance - Test Mode")
        
        selected_point = [None]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_point[0] = (x, y)
        
        cv2.setMouseCallback("Drone Surveillance - Test Mode", mouse_callback)
        
        last_tracks = []
        
        try:
            for frame_info in generator.generate_frames():
                if not self.is_running:
                    break
                
                # Handle click selection
                if selected_point[0]:
                    x, y = selected_point[0]
                    self.handle_click(x, y, last_tracks)
                    selected_point[0] = None
                
                # Process frame
                result = self.process_frame(frame_info.frame)
                last_tracks = result['tracks']
                
                # Display
                cv2.imshow("Drone Surveillance - Test Mode", result['frame'])
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.clear_selection()
        
        finally:
            self.is_running = False
            cv2.destroyAllWindows()


def run_dashboard_server(pipeline: Optional[SurveillancePipeline] = None):
    """Run the FastAPI dashboard server"""
    import uvicorn
    from dashboard.server import app, state
    
    if pipeline:
        state.pipeline = pipeline
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="Drone Surveillance System")
    parser.add_argument('--source', type=str, default='0',
                       help='Video source: 0 for webcam, path for file, URL for stream')
    parser.add_argument('--mode', type=str, choices=['local', 'test', 'dashboard'],
                       default='local', help='Running mode')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--dashboard-only', action='store_true',
                       help='Run dashboard server only (no ML processing)')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          🚁 StelX Drone Surveillance System 🚁            ║
    ║                                                          ║
    ║  Car Detection | License Plate Recognition | Tracking    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    if args.dashboard_only:
        print("🌐 Starting dashboard server only...")
        run_dashboard_server()
        return
    
    # Load configuration
    print(f"📄 Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Create pipeline
    pipeline = SurveillancePipeline(config)
    
    if args.mode == 'test':
        # Run with test video generator
        pipeline.run_with_test_video()
    
    elif args.mode == 'dashboard':
        # Run with dashboard
        print("🌐 Starting with web dashboard...")
        print("📺 Open http://localhost:8000 in your browser")
        
        # Start dashboard in separate thread
        dashboard_thread = threading.Thread(
            target=run_dashboard_server,
            args=(pipeline,),
            daemon=True
        )
        dashboard_thread.start()
        
        # Run pipeline with video source
        pipeline.run_local(args.source)
    
    else:
        # Local mode with OpenCV window
        pipeline.run_local(args.source)


if __name__ == "__main__":
    main()
