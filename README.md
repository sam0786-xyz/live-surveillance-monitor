# 🚁 StelX Drone Surveillance System

**Real-time vehicle detection, crowd monitoring, license plate recognition, and live object tracking for drone surveillance.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-green)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-red)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M4%20Optimized-lightgrey)
![High Altitude](https://img.shields.io/badge/High%20Altitude-200--300m-orange)

---

## 🎯 Features

### Core Detection
- **🚗 Vehicle Detection** - Detects cars, trucks, buses using custom-trained YOLOv11 models
- **👥 Person Detection** - High-altitude person detection with 97% mAP accuracy
- **🔖 License Plate Recognition** - Reads license plates with EasyOCR (Indian format optimized)
- **📊 Crowd Monitoring** - Real-time people count with density levels (LOW/MEDIUM/HIGH)

### Live Tracking
- **🎬 Live Video Tracking** - Bounding boxes move with vehicles during video playback
- **🎯 Click-to-Track** - Select any vehicle by clicking to highlight and follow
- **� Real-Time Updates** - Detection boxes update every 3 frames for smooth tracking
- **📈 DeepSORT Tracking** - Multi-object tracking with re-identification

### Dashboard Features
- **📤 Drag & Drop Upload** - Process images or videos instantly
- **📹 Live Camera** - Real-time webcam detection
- **⚡ Real-Time Processing** - Optimized for 15-25 FPS on Apple Silicon
- **🚨 Crowd Alerts** - Automatic visual alerts when crowd density is high
- **📥 Export CSV** - Download detected plates for reporting

---

## 🛠 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Detection** | YOLOv11 (Ultralytics) + SAHI | High-accuracy detection for small objects at altitude |
| **Person Model** | Custom YOLOv11s (97% mAP) | Trained for aerial person detection |
| **Vehicle Model** | Custom YOLOv11s (69% mAP) | Trained for cars, trucks, buses |
| **Plate Model** | Custom YOLOv11n (62% mAP) | Fast license plate region detection |
| **Preprocessing** | OpenCV CLAHE + Sharpening | Contrast enhancement for low-quality drone images |
| **OCR** | EasyOCR | License plate text extraction |
| **Tracking** | DeepSORT | Multi-object tracking with re-identification |
| **Backend** | FastAPI + WebSocket | Real-time API and streaming |
| **Frontend** | Vanilla JS + CSS | Professional dashboard with live tracking overlay |

---

## � Live Tracking System

The dashboard features **real-time bounding box tracking** during video playback:

1. **Video plays at full framerate** - Smooth playback without stuttering
2. **Every 3rd frame** is sent to the server for YOLO detection
3. **Detection results** are received with bounding box coordinates
4. **Boxes are drawn** on the live video canvas in real-time
5. **Boxes update** as new detections come in, following moving objects

### Box Colors:
- 🟢 **Green** - Vehicles (cars, trucks, buses)
- 🟠 **Orange** - People
- 🔵 **Cyan** - Selected/tracked object

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/sam0786-xyz/live-surveillance-monitor.git
cd drone_surveillance

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Models (First Run)
Models are downloaded automatically on first run, or you can train custom models:

```bash
# Train custom models (optional - uses COCO128 dataset)
python training/train_fast.py --all --epochs 50
```

---

## 🚀 Quick Start

### Run the Dashboard
```bash
source venv/bin/activate
python dashboard/server.py
```

Then open **http://localhost:8000** in your browser.

### How to Use
1. **Upload a video** - Drag & drop onto the upload area
2. **Click Play** - Watch live tracking boxes on vehicles and people
3. **Click any vehicle** - Highlight it for focused tracking
4. **View plates** - Recognized plates appear in the right panel
5. **Monitor crowd** - See people count and density level

---

## 📁 Project Structure

```
drone_surveillance/
├── main.py                 # Entry point for CLI mode
├── config.yaml             # Configuration settings
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── dashboard/
│   ├── server.py           # FastAPI backend (real-time optimized)
│   └── static/
│       ├── index.html      # Dashboard UI
│       ├── dashboard.js    # Live tracking logic
│       └── dashboard.css   # Professional styling
│
├── models/
│   ├── detector.py         # YOLOv11 detection with SAHI
│   └── tracker.py          # DeepSORT tracking
│
├── modules/
│   ├── car_detection.py    # Detection pipeline
│   ├── plate_recognition.py# LPR with OCR
│   └── object_tracking.py  # Tracking pipeline
│
├── training/
│   ├── train_fast.py       # Fast training script (COCO128)
│   └── train_all_models.py # Full training script
│
├── utils/
│   ├── video_handler.py    # Video/camera input
│   ├── visualization.py    # Drawing utilities
│   └── config_loader.py    # Config management
│
└── weights/                # Model weights (auto-downloaded)
    ├── person_detector.pt  # Custom person model
    ├── vehicle_detector.pt # Custom vehicle model
    └── plate_detector.pt   # Custom plate model
```

---

## ⚙️ Configuration

Edit `config.yaml`:

```yaml
detection:
  model: "yolo11s.pt"                # Base model
  person_model: "weights/person_detector.pt"   # Custom person detector
  vehicle_model: "weights/vehicle_detector.pt" # Custom vehicle detector
  plate_model: "weights/plate_detector.pt"     # Custom plate detector
  confidence_threshold: 0.25         # Detection confidence
  device: "mps"                      # mps (Apple GPU) / cpu / cuda

preprocessing:
  enabled: true
  clahe:
    enabled: true
    clip_limit: 2.0
  sharpening:
    enabled: true

sahi:
  enabled: true
  slice_height: 512
  slice_width: 512

crowd_monitoring:
  enabled: true
  density_thresholds:
    low: 5
    medium: 15

plate_recognition:
  enabled: true
  ocr_languages: ["en"]
```

---

## 🖥 Dashboard Features

| Feature | Description |
|---------|-------------|
| **Live Video Tracking** | Real-time bounding boxes that move with objects |
| **Video Feed** | Processed video with annotations |
| **Vehicle Grid** | Thumbnails of detected vehicles |
| **Plate List** | Recognized plates with confidence scores |
| **Crowd Monitor** | People count and density level |
| **Stats Cards** | Vehicle count, plate count, crowd count, FPS |
| **Click-to-Track** | Click on any vehicle to highlight and follow |
| **Crowd Alerts** | Visual alerts when crowd density is HIGH |
| **Export CSV** | Download plates as CSV |
| **Live Camera** | Real-time webcam detection |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard UI |
| `POST` | `/api/process_image` | Process an image |
| `POST` | `/api/process_frame` | Process a video frame |
| `GET` | `/api/plates` | Get detected plates |
| `GET` | `/api/crowd_stats` | Get crowd count and density |
| `POST` | `/api/select/{id}` | Select a track for following |
| `WS` | `/ws/control` | WebSocket for real-time controls |

---

## 📈 Performance

Tested on **M4 MacBook Air**:

| Mode | FPS | Notes |
|------|-----|-------|
| YOLOv11 Detection | ~16-20 | With preprocessing |
| Detection + SAHI | ~10-12 | Sliced inference for small objects |
| Full Pipeline (with OCR) | ~8-15 | OCR every 10 frames |
| Video Playback | 30+ | Smooth with async processing |

---

## 🏋️ Training Custom Models

Train your own models with the provided scripts:

```bash
# Fast training with COCO128 (~30 minutes)
python training/train_fast.py --all --epochs 50

# Train individual models
python training/train_fast.py --person --epochs 50
python training/train_fast.py --vehicle --epochs 50
python training/train_fast.py --plate --epochs 50
```

Models are saved to `weights/` directory.

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
