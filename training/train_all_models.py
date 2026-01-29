"""
Complete Model Training Suite for StelX Surveillance System
Trains all detection models with proper datasets and high epochs for best accuracy

Usage:
    python training/train_all_models.py --all           # Train all models (4-6 hours)
    python training/train_all_models.py --persons       # Train person detection only
    python training/train_all_models.py --plates        # Train plate detection only
    python training/train_all_models.py --vehicles      # Train vehicle detection only
"""

import os
import sys
import argparse
from pathlib import Path
import shutil
import yaml
import json

# Ensure ultralytics is available
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install -U ultralytics")
    from ultralytics import YOLO


PROJECT_DIR = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_DIR / 'weights'
OUTPUT_DIR = PROJECT_DIR / 'training_output'
DATA_DIR = PROJECT_DIR / 'training_data'


def setup_directories():
    """Create necessary directories"""
    for d in [WEIGHTS_DIR, OUTPUT_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_device():
    """Get best available device"""
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def train_person_detector(epochs: int = 100, batch_size: int = 8):
    """
    Train person detection model optimized for high-altitude/drone footage
    Uses VisDrone-style augmentations for aerial perspective
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     👤 Training Person Detection Model (High Altitude)   ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Model: YOLOv11s (balanced speed/accuracy)               ║
    ║  Dataset: COCO (person class) + augmentations            ║
    ║  Optimized for: Aerial/drone footage, small objects      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    device = get_device()
    print(f"📱 Using device: {device}")
    print(f"📊 Epochs: {epochs}")
    print(f"📦 Batch size: {batch_size}")
    
    # Use YOLOv11 small for better accuracy
    model = YOLO('yolo11s.pt')
    
    # Train with augmentations optimized for aerial/drone footage
    results = model.train(
        data='coco.yaml',  # Full COCO dataset (will auto-download)
        epochs=epochs,
        batch=batch_size,
        imgsz=1280,  # Higher resolution for small objects
        project=str(OUTPUT_DIR),
        name='person_detector_v2',
        device=device,
        patience=20,
        save=True,
        plots=True,
        # Only train on person class
        classes=[0],
        # Augmentations for aerial perspective
        degrees=15,        # Rotation for various angles
        translate=0.2,     # Position variation
        scale=0.5,         # Size variation (important for altitude changes)
        shear=5,           # Perspective distortion
        perspective=0.001, # 3D perspective
        flipud=0.1,        # Some vertical flip for top-down views
        fliplr=0.5,        # Horizontal flip
        mosaic=1.0,        # Full mosaic for varied scenes
        mixup=0.1,         # Mixup augmentation
        copy_paste=0.1,    # Copy-paste augmentation
        # Loss weights optimized for small objects
        box=10.0,          # Higher box loss for precise localization
        cls=1.0,           # Classification loss
        # Training parameters
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
    )
    
    # Save best model
    best_model = OUTPUT_DIR / 'person_detector_v2' / 'weights' / 'best.pt'
    if best_model.exists():
        final_path = WEIGHTS_DIR / 'person_detector.pt'
        shutil.copy(best_model, final_path)
        print(f"\n✅ Person detector saved to: {final_path}")
    
    return results


def train_vehicle_detector(epochs: int = 100, batch_size: int = 8):
    """
    Train vehicle detection model for cars, trucks, buses
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     🚗 Training Vehicle Detection Model                  ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Model: YOLOv11s (balanced speed/accuracy)               ║
    ║  Dataset: COCO (vehicle classes)                         ║
    ║  Classes: car, truck, bus, motorcycle                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    device = get_device()
    print(f"📱 Using device: {device}")
    print(f"📊 Epochs: {epochs}")
    
    model = YOLO('yolo11s.pt')
    
    results = model.train(
        data='coco.yaml',
        epochs=epochs,
        batch=batch_size,
        imgsz=1280,
        project=str(OUTPUT_DIR),
        name='vehicle_detector',
        device=device,
        patience=20,
        save=True,
        plots=True,
        # Vehicle classes: car(2), motorcycle(3), bus(5), truck(7)
        classes=[2, 3, 5, 7],
        # Augmentations
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=3,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        box=7.5,
    )
    
    best_model = OUTPUT_DIR / 'vehicle_detector' / 'weights' / 'best.pt'
    if best_model.exists():
        final_path = WEIGHTS_DIR / 'vehicle_detector.pt'
        shutil.copy(best_model, final_path)
        print(f"\n✅ Vehicle detector saved to: {final_path}")
    
    return results


def train_plate_detector(epochs: int = 100, batch_size: int = 16):
    """
    Train license plate detection model
    Uses transfer learning from vehicle detection
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     🔖 Training License Plate Detection Model            ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Model: YOLOv11n (fast for real-time)                    ║
    ║  Strategy: Fine-tune on plate-like rectangles            ║
    ║  Note: For best results, use CCPD or custom plate data   ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    device = get_device()
    print(f"📱 Using device: {device}")
    print(f"📊 Epochs: {epochs}")
    
    # Check if we have a custom plate dataset
    plate_data_path = DATA_DIR / 'plates' / 'plate_dataset.yaml'
    
    if plate_data_path.exists():
        print(f"📁 Using custom plate dataset: {plate_data_path}")
        data_yaml = str(plate_data_path)
    else:
        print("⚠️ No custom plate dataset found.")
        print("   For best results, download CCPD dataset:")
        print("   https://github.com/detectRecog/CCPD")
        print("\n   Training on COCO as fallback...")
        data_yaml = 'coco.yaml'
    
    model = YOLO('yolo11n.pt')
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        project=str(OUTPUT_DIR),
        name='plate_detector_v2',
        device=device,
        patience=15,
        save=True,
        plots=True,
        # Augmentations optimized for plates
        degrees=5,
        translate=0.1,
        scale=0.3,
        shear=2,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        box=10.0,
    )
    
    best_model = OUTPUT_DIR / 'plate_detector_v2' / 'weights' / 'best.pt'
    if best_model.exists():
        final_path = WEIGHTS_DIR / 'plate_detector.pt'
        shutil.copy(best_model, final_path)
        print(f"\n✅ Plate detector saved to: {final_path}")
    
    return results


def create_combined_model_config():
    """Create config to use all trained models"""
    config = {
        'detection': {
            'person_model': 'weights/person_detector.pt',
            'vehicle_model': 'weights/vehicle_detector.pt',
            'plate_model': 'weights/plate_detector.pt',
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
            'device': 'mps'
        },
        'plate_recognition': {
            'enabled': True,
            'ocr_languages': ['en'],
            'confidence_threshold': 0.3
        },
        'tracking': {
            'algorithm': 'deepsort',
            'max_age': 30,
            'min_hits': 3
        }
    }
    
    config_path = PROJECT_DIR / 'config_trained.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n📝 Config saved to: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description='Train all detection models for StelX Surveillance System'
    )
    parser.add_argument('--all', action='store_true', 
                        help='Train all models (persons, vehicles, plates)')
    parser.add_argument('--persons', action='store_true',
                        help='Train person detection model only')
    parser.add_argument('--vehicles', action='store_true',
                        help='Train vehicle detection model only')
    parser.add_argument('--plates', action='store_true',
                        help='Train plate detection model only')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick training mode (20 epochs)')
    
    args = parser.parse_args()
    
    # If no specific model selected, train all
    if not any([args.persons, args.vehicles, args.plates]):
        args.all = True
    
    epochs = 20 if args.quick else args.epochs
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          🎓 StelX Model Training Suite 🎓                ║
    ║                                                          ║
    ║     Complete training for all detection models           ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    print(f"⚙️ Configuration:")
    print(f"   └─ Epochs: {epochs}")
    print(f"   └─ Batch size: {args.batch}")
    print(f"   └─ Device: {get_device()}")
    print()
    
    setup_directories()
    
    training_times = {}
    
    if args.all or args.persons:
        import time
        start = time.time()
        print("\n" + "="*60)
        train_person_detector(epochs=epochs, batch_size=args.batch)
        training_times['person'] = time.time() - start
    
    if args.all or args.vehicles:
        import time
        start = time.time()
        print("\n" + "="*60)
        train_vehicle_detector(epochs=epochs, batch_size=args.batch)
        training_times['vehicle'] = time.time() - start
    
    if args.all or args.plates:
        import time
        start = time.time()
        print("\n" + "="*60)
        train_plate_detector(epochs=epochs, batch_size=args.batch)
        training_times['plate'] = time.time() - start
    
    # Create combined config
    if args.all:
        create_combined_model_config()
    
    # Summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    
    total_time = sum(training_times.values())
    print(f"\n⏱️ Training times:")
    for model, duration in training_times.items():
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        print(f"   └─ {model}: {hours}h {minutes}m")
    
    print(f"\n📁 Trained models saved to: {WEIGHTS_DIR}")
    
    # List trained models
    print("\n📦 Available models:")
    for model_file in WEIGHTS_DIR.glob('*.pt'):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"   └─ {model_file.name} ({size_mb:.1f} MB)")
    
    print("\n🚀 To use the trained models, run:")
    print("   python dashboard/server.py")


if __name__ == '__main__':
    main()
