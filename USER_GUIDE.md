# 📖 User Guide - StelX Drone Surveillance System

## Getting Started

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10 or higher
- ~2GB disk space

### Installation

```bash
cd drone_surveillance
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Dashboard

```bash
python dashboard/server.py
```

Open **http://localhost:8000** in your browser.

---

## Using the Dashboard

### 1. Upload Media

**For Images:**
- Drag and drop an image onto the video area
- Or click "Upload Media" in the sidebar

**For Videos:**
- Drag and drop a video file (MP4, AVI, MOV)
- Press **Play** to start processing

### 2. View Detections

- **Green boxes** = Detected vehicles
- **Vehicle thumbnails** appear in the right panel
- **Stats** update in real-time (vehicle count, FPS)

### 3. License Plate Recognition

- Plates are automatically detected from vehicles
- Recognized plates appear in the **License Plates** panel
- Click **Export** to download plates as CSV

### 4. Click-to-Track

1. Click on any detected vehicle in the video
2. The vehicle will be highlighted with a **cyan border**
3. The tracking ID appears in the Active Tracking card
4. Click elsewhere or press "Clear" to stop tracking

---

## Running Modes

### Dashboard Mode (Recommended)
```bash
python dashboard/server.py
```

### Test Mode (Synthetic Video)
```bash
python main.py --mode test
```

### Camera Mode
```bash
python main.py --source 0              # Webcam
python main.py --source video.mp4      # Video file
python main.py --source rtsp://...     # RTSP stream
```

---

## Keyboard Controls (CLI Mode)

| Key | Action |
|-----|--------|
| **Click** | Select vehicle for tracking |
| **C** | Clear tracking selection |
| **S** | Save screenshot |
| **Q** | Quit |

---

## Configuration

Edit `config.yaml` to customize:

```yaml
detection:
  confidence_threshold: 0.20    # Lower = more detections
  
plate_recognition:
  enabled: true                 # Set false to disable

tracking:
  max_age: 30                   # How long to keep lost tracks
```

---

## Troubleshooting

### No vehicles detected
- Lower `confidence_threshold` in config.yaml
- Ensure vehicles are visible and not too small

### Plates not recognized
- Plate must be clearly visible (not blurry)
- Works best with front/rear view of vehicles
- Optimized for Indian license plate format

### Slow performance
- Use `yolov8n.pt` model (nano version)
- Disable plate recognition if not needed
- Process lower resolution video

---

## API Usage

### Process an Image
```bash
curl -X POST "http://localhost:8000/api/process_image" \
  -F "file=@car_image.jpg"
```

### Get Detected Plates
```bash
curl "http://localhost:8000/api/plates"
```

---

*For technical details, see README.md*
