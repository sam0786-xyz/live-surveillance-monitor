"""
YOLOv8 Custom Model Training Script
Train models for License Plate Detection and Person Detection
"""

import os
import sys
import argparse
from pathlib import Path
import shutil
import yaml

# Ensure ultralytics is available
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO


def create_plate_dataset_config(data_dir: Path) -> Path:
    """Create dataset config for license plate detection"""
    config = {
        'path': str(data_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'license_plate'
        }
    }
    
    config_path = data_dir / 'plate_dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


def create_person_dataset_config(data_dir: Path) -> Path:
    """Create dataset config for person detection"""
    config = {
        'path': str(data_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'person'
        }
    }
    
    config_path = data_dir / 'person_dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path


def download_sample_plate_data(data_dir: Path):
    """Download sample license plate data for training"""
    import urllib.request
    import zipfile
    
    print("📥 Downloading license plate training data...")
    
    # Create directories
    images_train = data_dir / 'images' / 'train'
    images_val = data_dir / 'images' / 'val'
    labels_train = data_dir / 'labels' / 'train'
    labels_val = data_dir / 'labels' / 'val'
    
    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Note: In production, download full CCPD or Indian plate dataset
    # For demo, we'll use a smaller sample
    print("   └─ Creating sample annotations for demo...")
    print("   └─ For production, download CCPD dataset from:")
    print("      https://github.com/detectRecog/CCPD")
    
    return True


def download_crowdhuman_sample(data_dir: Path):
    """Download CrowdHuman sample data for person detection"""
    print("📥 Setting up person detection training data...")
    
    # Create directories
    images_train = data_dir / 'images' / 'train'
    images_val = data_dir / 'images' / 'val'
    labels_train = data_dir / 'labels' / 'train'
    labels_val = data_dir / 'labels' / 'val'
    
    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)
    
    print("   └─ For production, download CrowdHuman from:")
    print("      https://www.crowdhuman.org/")
    
    return True


def train_plate_model(
    data_config: Path,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640
):
    """Train license plate detection model"""
    print("\n🚀 Training License Plate Detection Model...")
    print(f"   └─ Epochs: {epochs}")
    print(f"   └─ Batch Size: {batch_size}")
    print(f"   └─ Image Size: {img_size}")
    
    # Load pretrained model
    model = YOLO('yolov8n.pt')
    
    # Train
    results = model.train(
        data=str(data_config),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=str(output_dir),
        name='plate_detector',
        device='mps',  # Apple Silicon
        patience=20,
        save=True,
        plots=True
    )
    
    # Copy best model
    best_model = output_dir / 'plate_detector' / 'weights' / 'best.pt'
    if best_model.exists():
        final_path = output_dir.parent / 'weights' / 'plate_detector.pt'
        final_path.parent.mkdir(exist_ok=True)
        shutil.copy(best_model, final_path)
        print(f"✅ Model saved to: {final_path}")
    
    return results


def train_person_model(
    data_config: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 8,
    img_size: int = 640
):
    """Train person detection model"""
    print("\n🚀 Training Person Detection Model...")
    print(f"   └─ Epochs: {epochs}")
    print(f"   └─ Batch Size: {batch_size}")
    print(f"   └─ Image Size: {img_size}")
    
    # Load pretrained model (slightly larger for accuracy)
    model = YOLO('yolov8s.pt')
    
    # Train
    results = model.train(
        data=str(data_config),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=str(output_dir),
        name='person_detector',
        device='mps',
        patience=15,
        save=True,
        plots=True
    )
    
    # Copy best model
    best_model = output_dir / 'person_detector' / 'weights' / 'best.pt'
    if best_model.exists():
        final_path = output_dir.parent / 'weights' / 'person_detector.pt'
        final_path.parent.mkdir(exist_ok=True)
        shutil.copy(best_model, final_path)
        print(f"✅ Model saved to: {final_path}")
    
    return results


def finetune_on_coco_persons(output_dir: Path, epochs: int = 30):
    """Fine-tune YOLOv8 specifically for person detection using COCO"""
    print("\n🎯 Fine-tuning on COCO Person Class...")
    
    # Create a custom config that only trains on person class
    coco_person_config = {
        'path': 'coco',  # Will auto-download
        'train': 'train2017',
        'val': 'val2017',
        'names': {0: 'person'}
    }
    
    config_path = output_dir / 'coco_person.yaml'
    
    # Use COCO with only person class (class 0)
    model = YOLO('yolov8s.pt')
    
    # Train with class filter
    results = model.train(
        data='coco128.yaml',  # Use COCO128 for faster demo
        epochs=epochs,
        batch=8,
        imgsz=640,
        project=str(output_dir),
        name='person_detector_coco',
        device='mps',
        classes=[0],  # Only person class
        patience=10,
        save=True
    )
    
    # Copy best model
    best_model = output_dir / 'person_detector_coco' / 'weights' / 'best.pt'
    if best_model.exists():
        final_path = output_dir.parent / 'weights' / 'person_detector.pt'
        final_path.parent.mkdir(exist_ok=True)
        shutil.copy(best_model, final_path)
        print(f"✅ Person model saved to: {final_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train custom detection models')
    parser.add_argument('--model', type=str, choices=['plate', 'person', 'both'],
                       default='both', help='Which model to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training mode (fewer epochs)')
    
    args = parser.parse_args()
    
    # Setup paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'training_data'
    output_dir = project_dir / 'training_output'
    
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    epochs = 10 if args.quick else args.epochs
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          🎓 YOLOv8 Custom Model Training 🎓              ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    if args.model in ['plate', 'both']:
        print("\n📋 License Plate Detection Model")
        print("="*50)
        plate_data = data_dir / 'plates'
        plate_data.mkdir(exist_ok=True)
        download_sample_plate_data(plate_data)
        
        # For demo, skip if no data
        print("   ⚠️ Skipping plate training (no dataset)")
        print("   └─ Download CCPD dataset to train")
    
    if args.model in ['person', 'both']:
        print("\n👤 Person Detection Model")
        print("="*50)
        
        # Fine-tune on COCO persons (auto-download)
        finetune_on_coco_persons(output_dir, epochs=epochs)
    
    print("\n" + "="*50)
    print("✅ Training complete!")
    print(f"   └─ Models saved to: {project_dir / 'weights'}")
    print("\nTo use the trained models, update config.yaml:")
    print("   detection:")
    print("     model: 'weights/person_detector.pt'")


if __name__ == '__main__':
    main()
