"""
FastAPI Dashboard Server - Real-Time Optimized Edition
High-speed video streaming with detection, tracking, and plate recognition
"""

import asyncio
import json
import time
import base64
import re
from typing import Optional, Dict, List
from pathlib import Path
import io

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.visualization import encode_frame_to_base64, Visualizer
from models.detector import VehicleDetector, Detection
from models.tracker import ObjectTracker, Track


# Initialize FastAPI app
app = FastAPI(
    title="StelX Surveillance Dashboard",
    description="Real-time vehicle detection, tracking, and license plate recognition",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# Global state and models
class AppState:
    """Application state container with performance optimization"""
    def __init__(self):
        self.is_running = False
        self.connected_clients: List[WebSocket] = []
        self.current_stats: Dict = {}
        self.recognized_plates: List[Dict] = []
        self.detections: List[Dict] = []
        self.selected_track_id: Optional[int] = None
        
        # Crowd monitoring
        self.crowd_stats: Dict = {
            'count': 0,
            'density': 'low',
            'density_thresholds': {'low': 5, 'medium': 15}
        }
        self.person_detections: List[Dict] = []
        
        # Models (lazy loaded)
        self._detector = None
        self._tracker = None
        self._plate_recognizer = None
        self._visualizer = None
        
        # Performance tracking
        self.frame_count = 0
        self.ocr_every_n_frames = 10  # Only run OCR every N frames for real-time
        self.last_plates = []  # Cache plates between OCR runs
        
    @property
    def detector(self):
        if self._detector is None:
            print("🔧 Loading vehicle/person detector...")
            self._detector = VehicleDetector(
                model_path="yolov8n.pt",
                confidence_threshold=0.20,  # Low threshold for dark cars
                device="mps",
                classes=[0, 2, 5, 7]  # 0=person, 2=car, 5=bus, 7=truck
            )
        return self._detector
    
    @property
    def tracker(self):
        if self._tracker is None:
            print("🔧 Loading tracker...")
            self._tracker = ObjectTracker()
        return self._tracker
    
    @property
    def plate_recognizer(self):
        if self._plate_recognizer is None:
            print("🔧 Loading plate recognizer...")
            try:
                import easyocr
                self._plate_recognizer = easyocr.Reader(['en'], gpu=False)
                print("✅ Plate recognizer ready!")
            except Exception as e:
                print(f"⚠️ Plate recognition not available: {e}")
                self._plate_recognizer = False
        return self._plate_recognizer if self._plate_recognizer else None
    
    @property
    def visualizer(self):
        if self._visualizer is None:
            self._visualizer = Visualizer()
        return self._visualizer


state = AppState()


def process_frame_fast(image: np.ndarray, run_ocr: bool = False) -> Dict:
    """
    Fast frame processing optimized for real-time.
    OCR only runs when run_ocr=True to maintain frame rate.
    Includes crowd monitoring for person detection.
    """
    start_time = time.time()
    
    # Run detection (fast - ~30-50ms on M4)
    all_detections = state.detector.detect(image)
    
    # Separate person detections from vehicle detections
    person_detections = [d for d in all_detections if d.class_name == 'person']
    vehicle_detections = [d for d in all_detections if d.class_name != 'person']
    
    # Update crowd stats
    person_count = len(person_detections)
    thresholds = state.crowd_stats['density_thresholds']
    if person_count < thresholds['low']:
        density_level = 'low'
    elif person_count < thresholds['medium']:
        density_level = 'medium'
    else:
        density_level = 'high'
    
    state.crowd_stats['count'] = person_count
    state.crowd_stats['density'] = density_level
    
    # Run tracking on vehicles only (fast - ~5-10ms)
    tracks = state.tracker.update(vehicle_detections, image)
    
    # Plate recognition - only run periodically (on vehicles only)
    plates_found = state.last_plates.copy() if not run_ocr else []
    
    if run_ocr and state.plate_recognizer and vehicle_detections:
        print(f"   🔍 Running OCR on {len(vehicle_detections)} vehicles...")
        for det in vehicle_detections[:3]:  # Limit to 3 vehicles for speed
            x1, y1, x2, y2 = det.bbox
            try:
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = image[y1:y2, x1:x2]
                
                if crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
                    # Try OCR on FULL crop first (catches plates in any position)
                    results = state.plate_recognizer.readtext(crop)
                    for bbox, text, conf in results:
                        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                        has_letters = bool(re.search(r'[A-Z]', cleaned))
                        has_numbers = bool(re.search(r'[0-9]', cleaned))
                        # Accept 4+ character plates with letters AND numbers
                        if len(cleaned) >= 4 and has_letters and has_numbers and conf > 0.3:
                            plate_info = {
                                'text': cleaned,
                                'confidence': float(conf),
                                'raw': text
                            }
                            # Avoid duplicates
                            if not any(p['text'] == cleaned for p in plates_found):
                                plates_found.append(plate_info)
                                print(f"   └─ Plate: {cleaned} ({conf:.0%})")
            except Exception as e:
                print(f"   └─ OCR error: {e}")
        
        state.last_plates = plates_found
    
    # Get trajectories for selected track only (optimization)
    trajectories = {}
    for track in tracks:
        if track.is_selected or track.track_id == state.selected_track_id:
            try:
                traj = state.tracker.get_track_trajectory(track.track_id)
                if traj:
                    trajectories[track.track_id] = traj
            except:
                pass
    
    # Draw visualizations
    output_frame = image.copy()
    
    try:
        # Draw person detections (crowd monitoring) - ORANGE color
        for i, det in enumerate(person_detections):
            x1, y1, x2, y2 = det.bbox
            h, w = output_frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Color based on density
            if density_level == 'high':
                color = (0, 0, 255)  # Red for high density
            elif density_level == 'medium':
                color = (0, 165, 255)  # Orange for medium
            else:
                color = (0, 255, 255)  # Yellow for low
            
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            label = f"Person"
            label_y = max(20, y1 - 10)
            cv2.putText(output_frame, label, (x1, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw vehicle detections with track IDs
        for i, det in enumerate(vehicle_detections):
            x1, y1, x2, y2 = det.bbox
            h, w = output_frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Check if this detection is being tracked
            is_selected = False
            track_id = i + 1
            for track in tracks:
                tx1, ty1, tx2, ty2 = track.bbox
                if abs(tx1 - x1) < 50 and abs(ty1 - y1) < 50:
                    track_id = track.track_id
                    is_selected = track.track_id == state.selected_track_id
                    break
            
            # Color: cyan for selected, green for normal
            color = (255, 200, 0) if is_selected else (0, 255, 0)
            thickness = 3 if is_selected else 2
            
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            label = f"ID:{track_id} {det.class_name}"
            label_y = max(20, y1 - 10)
            cv2.putText(output_frame, label, (x1, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw trajectory for selected
            if is_selected and track_id in trajectories:
                points = trajectories[track_id]
                for j in range(1, len(points)):
                    cv2.line(output_frame, points[j-1], points[j], (0, 255, 255), 2)
        
        # Draw plates on frame
        for i, plate in enumerate(plates_found[:3]):
            if i < len(vehicle_detections):
                x1, y1, x2, y2 = vehicle_detections[i].bbox
                plate_y = min(output_frame.shape[0] - 10, y2 + 25)
                text = f"🔖 {plate['text']}"
                cv2.putText(output_frame, text, (x1, plate_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw crowd stats overlay in top-right corner
        if person_count > 0:
            overlay_text = f"Crowd: {person_count} ({density_level.upper()})"
            text_size = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            overlay_x = output_frame.shape[1] - text_size[0] - 20
            overlay_y = 40
            # Background rectangle
            cv2.rectangle(output_frame, (overlay_x - 10, overlay_y - 30),
                         (overlay_x + text_size[0] + 10, overlay_y + 10), (0, 0, 0), -1)
            # Density color for text
            if density_level == 'high':
                text_color = (0, 0, 255)
            elif density_level == 'medium':
                text_color = (0, 165, 255)
            else:
                text_color = (0, 255, 0)
            cv2.putText(output_frame, overlay_text, (overlay_x, overlay_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    except Exception as e:
        print(f"   └─ Draw error: {e}")
    
    # Calculate stats
    processing_time = (time.time() - start_time) * 1000
    fps = 1000 / processing_time if processing_time > 0 else 0
    
    # Build detection data (vehicles only for detection grid)
    detection_data = []
    for i, det in enumerate(vehicle_detections[:10]):
        x1, y1, x2, y2 = det.bbox
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]
        
        thumb_b64 = None
        if crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 5:
            try:
                thumb = cv2.resize(crop, (80, 50))
                _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
                thumb_b64 = base64.b64encode(buffer).decode('utf-8')
            except:
                pass
        
        # Find track ID
        track_id = i + 1
        for track in tracks:
            tx1, ty1, _, _ = track.bbox
            if abs(tx1 - x1) < 50 and abs(ty1 - y1) < 50:
                track_id = track.track_id
                break
        
        detection_data.append({
            'track_id': track_id,
            'class': det.class_name,
            'confidence': float(det.confidence),
            'bbox': [x1, y1, x2, y2],
            'thumbnail': thumb_b64
        })
    
    # Encode output frame
    frame_b64 = encode_frame_to_base64(output_frame, quality=75)  # Lower quality for speed
    
    return {
        'frame_base64': frame_b64,
        'detections': detection_data,
        'plates': plates_found,
        'crowd': {
            'count': person_count,
            'density': density_level
        },
        'stats': {
            'vehicles': len(vehicle_detections),
            'plates': len(plates_found),
            'fps': round(fps, 1),
            'tracking': state.selected_track_id,
            'crowd_count': person_count,
            'crowd_density': density_level
        }
    }


# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard HTML"""
    html_path = static_path / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Dashboard files not found</h1>")


@app.get("/api/status")
async def get_status():
    """Get current system status"""
    return {
        "is_running": state.is_running,
        "connected_clients": len(state.connected_clients),
        "stats": state.current_stats,
        "model_loaded": state._detector is not None
    }


@app.post("/api/process_image")
async def process_image(file: UploadFile = File(...)):
    """Process an uploaded image - always run OCR"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        print(f"📷 Processing image: {image.shape}")
        result = process_frame_fast(image, run_ocr=True)  # Always OCR for images
        
        # Store plates
        for plate in result['plates']:
            if plate['text'] not in [p['text'] for p in state.recognized_plates[-50:]]:
                state.recognized_plates.append(plate)
        
        print(f"   └─ Detections: {result['stats']['vehicles']}, Plates: {result['stats']['plates']}")
        return JSONResponse(content=result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process_frame")
async def process_frame(frame: UploadFile = File(...)):
    """Process a video frame - OCR only periodically"""
    try:
        contents = await frame.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid frame")
        
        # Increment frame counter and run OCR every N frames
        state.frame_count += 1
        run_ocr = (state.frame_count % state.ocr_every_n_frames == 0)
        
        result = process_frame_fast(image, run_ocr=run_ocr)
        
        # Store new plates
        for plate in result['plates']:
            if plate['text'] not in [p['text'] for p in state.recognized_plates[-50:]]:
                state.recognized_plates.append(plate)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(content={
            'frame_base64': '',
            'detections': [],
            'plates': state.last_plates,
            'stats': {'vehicles': 0, 'plates': len(state.last_plates), 'fps': 0}
        })


@app.post("/api/select/{track_id}")
async def select_track(track_id: int):
    """Select a track for focused tracking"""
    state.selected_track_id = track_id
    try:
        state.tracker.select_track_by_id(track_id)
    except:
        pass
    return {"selected": track_id}


@app.post("/api/clear_selection")
async def clear_selection():
    """Clear track selection"""
    state.selected_track_id = None
    try:
        state.tracker.clear_selection()
    except:
        pass
    return {"status": "selection_cleared"}


@app.get("/api/plates")
async def get_plates():
    """Get list of recognized plates"""
    return {"plates": state.recognized_plates[-50:]}


@app.delete("/api/plates")
async def clear_plates():
    """Clear plate history"""
    state.recognized_plates.clear()
    state.last_plates.clear()
    return {"status": "plates_cleared"}


@app.get("/api/detections")
async def get_detections():
    """Get recent detections"""
    return {"detections": state.detections[-20:]}


@app.get("/api/crowd_stats")
async def get_crowd_stats():
    """Get current crowd monitoring statistics"""
    return {
        "crowd_count": state.crowd_stats['count'],
        "density": state.crowd_stats['density'],
        "thresholds": state.crowd_stats['density_thresholds']
    }


# WebSocket endpoints
@app.websocket("/ws/control")
async def control_handler(websocket: WebSocket):
    """WebSocket endpoint for control messages"""
    await websocket.accept()
    state.connected_clients.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "click":
                x = data.get("x", 0)
                y = data.get("y", 0)
                # Find detection at click position
                for det in state.detections[-20:]:
                    bbox = det.get('bbox', [0, 0, 0, 0])
                    if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                        state.selected_track_id = det['track_id']
                        await websocket.send_json({
                            "type": "track_selected",
                            "track_id": det['track_id']
                        })
                        break
                        
            elif action == "select":
                track_id = data.get("track_id")
                state.selected_track_id = track_id
                await websocket.send_json({
                    "type": "track_selected",
                    "track_id": track_id
                })
                
            elif action == "clear_selection":
                state.selected_track_id = None
                try:
                    state.tracker.clear_selection()
                except:
                    pass
                await websocket.send_json({
                    "type": "selection_cleared"
                })
                
            elif action == "set_confidence":
                value = data.get("value", 0.5)
                if state._detector:
                    state._detector.confidence_threshold = value
                await websocket.send_json({
                    "type": "confidence_updated",
                    "value": value
                })
                
    except WebSocketDisconnect:
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)
    except Exception as e:
        print(f"Control WebSocket error: {e}")
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)


@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    """WebSocket endpoint for video streaming"""
    await websocket.accept()
    state.connected_clients.append(websocket)
    
    try:
        while True:
            await asyncio.sleep(0.1)
            await websocket.send_json({
                "type": "stats",
                "data": state.current_stats
            })
    except WebSocketDisconnect:
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)


# Application lifecycle
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          🚁 StelX Surveillance Dashboard 🚁              ║
    ║             Real-Time Optimized Edition                   ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    print("📡 Server running at http://localhost:8000")
    print("⚡ Optimized for real-time: OCR runs every 10 frames")
    print("📤 Upload images or videos to detect vehicles and plates")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    state.is_running = False
    for client in state.connected_clients:
        try:
            await client.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
