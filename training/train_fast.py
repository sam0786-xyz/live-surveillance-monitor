"""
Fast Model Training for StelX Surveillance System
Uses COCO128 (small dataset) for quick training (~30 minutes total)
Optimized for M4 Mac performance

Usage:
    python training/train_fast.py --all     # Train all models (~30 min)
    python training/train_fast.py --persons # Train person detection (~10 min)
"""

import os
import sys
import argparse
from pathlib import Path
import shutil
import time

try:
    from ultralytics import YOLO
except ImportError:
    os.system("pip install -U ultralytics")
    from ultralytics import YOLO


PROJECT_DIR = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_DIR / 'weights'
OUTPUT_DIR = PROJECT_DIR / 'training_output'


def get_device():
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def train_person_detector(epochs: int = 25):
    """Train person detector optimized for surveillance/drone footage"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   👤 Training Person Detector (Fast Mode)                ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    device = get_device()
    print(f"📱 Device: {device} | Epochs: {epochs}")
    
    model = YOLO('yolo11s.pt')
    
    results = model.train(
        data='coco128.yaml',  # Small dataset for fast training
        epochs=epochs,
        batch=8,
        imgsz=640,  # Standard size for speed
        project=str(OUTPUT_DIR),
        name='person_fast',
        device=device,
        patience=10,
        save=True,
        classes=[0],  # Person only
        # Augmentations for aerial/surveillance
        degrees=15,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        box=10.0,  # Higher weight for precise boxes
    )
    
    # Save model
    best = OUTPUT_DIR / 'person_fast' / 'weights' / 'best.pt'
    if best.exists():
        WEIGHTS_DIR.mkdir(exist_ok=True)
        shutil.copy(best, WEIGHTS_DIR / 'person_detector.pt')
        print(f"✅ Saved: {WEIGHTS_DIR}/person_detector.pt")
    
    return results


def train_vehicle_detector(epochs: int = 25):
    """Train vehicle detector for cars, trucks, buses"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   🚗 Training Vehicle Detector (Fast Mode)               ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    device = get_device()
    print(f"📱 Device: {device} | Epochs: {epochs}")
    
    model = YOLO('yolo11s.pt')
    
    results = model.train(
        data='coco128.yaml',
        epochs=epochs,
        batch=8,
        imgsz=640,
        project=str(OUTPUT_DIR),
        name='vehicle_fast',
        device=device,
        patience=10,
        save=True,
        classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
        degrees=10,
        scale=0.4,
        mosaic=1.0,
    )
    
    best = OUTPUT_DIR / 'vehicle_fast' / 'weights' / 'best.pt'
    if best.exists():
        WEIGHTS_DIR.mkdir(exist_ok=True)
        shutil.copy(best, WEIGHTS_DIR / 'vehicle_detector.pt')
        print(f"✅ Saved: {WEIGHTS_DIR}/vehicle_detector.pt")
    
    return results


def train_plate_detector(epochs: int = 25):
    """Train plate detector (uses general detection fine-tuning)"""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   🔖 Training Plate Detector (Fast Mode)                 ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    device = get_device()
    print(f"📱 Device: {device} | Epochs: {epochs}")
    
    model = YOLO('yolo11n.pt')  # Nano for speed
    
    results = model.train(
        data='coco128.yaml',
        epochs=epochs,
        batch=16,
        imgsz=640,
        project=str(OUTPUT_DIR),
        name='plate_fast',
        device=device,
        patience=10,
        save=True,
        # Optimize for small rectangular objects
        degrees=5,
        scale=0.3,
        shear=2,
        mosaic=0.5,
        box=10.0,
    )
    
    best = OUTPUT_DIR / 'plate_fast' / 'weights' / 'best.pt'
    if best.exists():
        WEIGHTS_DIR.mkdir(exist_ok=True)
        shutil.copy(best, WEIGHTS_DIR / 'plate_detector.pt')
        print(f"✅ Saved: {WEIGHTS_DIR}/plate_detector.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Fast model training')
    parser.add_argument('--all', action='store_true', help='Train all models')
    parser.add_argument('--persons', action='store_true', help='Train person detector')
    parser.add_argument('--vehicles', action='store_true', help='Train vehicle detector')
    parser.add_argument('--plates', action='store_true', help='Train plate detector')
    parser.add_argument('--epochs', type=int, default=25, help='Epochs per model')
    
    args = parser.parse_args()
    
    if not any([args.all, args.persons, args.vehicles, args.plates]):
        args.all = True
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║       🚀 StelX Fast Model Training 🚀                    ║
    ║          Optimized for ~30 minute completion             ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    WEIGHTS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    total_start = time.time()
    
    if args.all or args.persons:
        start = time.time()
        train_person_detector(args.epochs)
        print(f"⏱️ Person detector: {(time.time()-start)/60:.1f} min")
    
    if args.all or args.vehicles:
        start = time.time()
        train_vehicle_detector(args.epochs)
        print(f"⏱️ Vehicle detector: {(time.time()-start)/60:.1f} min")
    
    if args.all or args.plates:
        start = time.time()
        train_plate_detector(args.epochs)
        print(f"⏱️ Plate detector: {(time.time()-start)/60:.1f} min")
    
    total_time = (time.time() - total_start) / 60
    
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║   ✅ TRAINING COMPLETE!                                  ║
    ╠══════════════════════════════════════════════════════════╣
    ║   Total time: {total_time:.1f} minutes                              
    ║   Models saved to: weights/                              
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # List models
    print("\n📦 Trained models:")
    for f in WEIGHTS_DIR.glob('*.pt'):
        size = f.stat().st_size / (1024*1024)
        print(f"   └─ {f.name} ({size:.1f} MB)")
    
    print("\n🚀 Run dashboard: python dashboard/server.py")


if __name__ == '__main__':
    main()
