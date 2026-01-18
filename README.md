# 🚁 StelX Drone Surveillance System

**Real-time vehicle detection, crowd monitoring, license plate recognition, and object tracking for drone surveillance.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-green)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-red)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M4%20Optimized-lightgrey)
![High Altitude](https://img.shields.io/badge/High%20Altitude-200--300m-orange)

---

## 🎯 Features

- **🚗 Vehicle Detection** - Detects cars, trucks, buses in real-time using YOLOv11
- **👥 Crowd Monitoring** - Counts people and tracks crowd density (LOW/MEDIUM/HIGH)
- **🔖 License Plate Recognition** - Reads license plates (optimized for Indian format)
- **🎯 Click-to-Track** - Select and follow any vehicle by clicking on it
- **📊 Professional Dashboard** - Modern web UI with real-time stats
- **📤 File Upload** - Process images or videos via drag-and-drop
- **⚡ Real-Time Processing** - Optimized for 15-25 FPS on Apple Silicon
- **🚨 Crowd Alerts** - Automatic notifications when crowd density is high

---

## 🛠 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Detection** | YOLOv11l (Ultralytics) | High-accuracy detection for small objects |
| **Preprocessing** | OpenCV CLAHE | Contrast enhancement for low-quality images |
| **OCR** | EasyOCR | License plate text extraction |
| **Tracking** | DeepSORT | Multi-object tracking with re-ID |
| **Backend** | FastAPI + WebSocket | Real-time API and streaming |
| **Frontend** | Vanilla JS + CSS | Professional dashboard UI |

---

## � Installation

```bash
# Clone the repository
git clone <repository-url>
cd drone_surveillance

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Run the Dashboard
```bash
python dashboard/server.py
```
Then open **http://localhost:8000** in your browser.

### Upload & Detect
1. Drag an image or video onto the upload area
2. View detected vehicles with bounding boxes
3. See recognized license plates in the right panel
4. Click any vehicle to track it

---

## 📁 Project Structure

```
drone_surveillance/
├── main.py                 # Entry point for CLI mode
├── config.yaml             # Configuration settings
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── USER_GUIDE.md           # Detailed usage guide
│
├── dashboard/
│   ├── server.py           # FastAPI backend (real-time optimized)
│   └── static/             # Frontend (HTML/CSS/JS)
│
├── models/
│   ├── detector.py         # YOLOv11 vehicle detection with SAHI
│   └── tracker.py          # DeepSORT tracking
│
├── modules/
│   ├── car_detection.py    # Detection pipeline
│   ├── plate_recognition.py # LPR with OCR
│   ├── improved_plate_recognition.py # Enhanced LPR
│   └── object_tracking.py  # Tracking pipeline
│
├── utils/
│   ├── video_handler.py    # Video/camera input
│   ├── visualization.py    # Drawing utilities
│   └── config_loader.py    # Config management
│
└── weights/                # Model weights (auto-downloaded)
```

---

## ⚙️ Configuration

Edit `config.yaml`:

```yaml
detection:
  model: "yolo11l.pt"           # yolo11l for high-altitude small objects
  confidence_threshold: 0.15    # Lower for small object detection
  device: "mps"                 # mps (Apple GPU) / cpu
  classes: [0, 2, 5, 7]         # 0=person, 2=car, 5=bus, 7=truck

preprocessing:                  # Image enhancement for drone footage
  enabled: true
  clahe:                        # Contrast enhancement
    enabled: true
    clip_limit: 2.0
  sharpening:                   # Edge enhancement
    enabled: true
  upscaling:                    # Upscale low-res inputs
    enabled: true
    min_dimension: 640
    target_dimension: 1280

sahi:                           # Sliced inference for small objects
  enabled: true
  slice_height: 512
  slice_width: 512
  overlap_ratio: 0.2

crowd_monitoring:
  enabled: true
  density_thresholds:
    low: 5                      # Below this = LOW density
    medium: 15                  # Below this = MEDIUM, above = HIGH

plate_recognition:
  enabled: true
  ocr_languages: ["en"]

tracking:
  max_age: 30                   # Frames to keep track after occlusion
```

---

## 🖥 Dashboard Features

| Feature | Description |
|---------|-------------|
| **Video Feed** | Real-time processed video with annotations |
| **Vehicle Grid** | Thumbnails of detected vehicles |
| **Plate List** | Recognized plates with confidence scores |
| **Crowd Monitor** | People count and density level (LOW/MEDIUM/HIGH) |
| **Stats Cards** | Vehicle count, plate count, crowd count, FPS |
| **Click-to-Track** | Click on any vehicle to highlight and follow |
| **Crowd Alerts** | Visual alerts when crowd density is HIGH |
| **Export** | Download plates as CSV |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard UI |
| `POST` | `/api/process_image` | Process an image |
| `POST` | `/api/process_frame` | Process a video frame |
| `GET` | `/api/plates` | Get detected plates |
| `GET` | `/api/crowd_stats` | Get crowd count and density |
| `POST` | `/api/select/{id}` | Select a track |
| `WS` | `/ws/control` | WebSocket for controls |

---

## 📈 Performance

Tested on **M4 MacBook Air**:

| Mode | FPS | Notes |
|------|-----|-------|
| YOLOv11l Detection | ~16 | With CLAHE/sharpening preprocessing |
| Detection + SAHI | ~10-12 | Sliced inference for small objects |
| Full Pipeline (with OCR) | ~8-15 | OCR every 10 frames |

---

## 🔮 Future Integration

Ready for drone camera integration:
```bash
# RTSP stream from drone
python main.py --source rtsp://drone-ip:port/stream
```

---

## 📄 License

Proprietary - StelX Dynamics Pvt. Ltd.

---

*Last Updated: January 2026*
