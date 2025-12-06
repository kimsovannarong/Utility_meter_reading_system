#!/usr/bin/env python3
"""
Project setup script
"""

import os
import subprocess
import sys
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure"""
    directories = [
        # Models
        "models/detectors/yolov8/weights",
        "models/detectors/yolov10/weights",
        "models/detectors/yolov11/weights",
        "models/detectors/detr/weights",
        "models/ocr_models/paddleocr",
        "models/ocr_models/easyocr",
        "models/ocr_models/tesseract",
        "models/ocr_models/doctr",
        "models/ocr_models/custom_crnn",
        "models/comparisons/detection_results",
        "models/comparisons/ocr_results",
        "models/comparisons/pipeline_results",
        
        # Datasets
        "datasets/detection/train/images",
        "datasets/detection/train/labels",
        "datasets/detection/val/images",
        "datasets/detection/val/labels",
        "datasets/detection/test/images",
        "datasets/detection/test/labels",
        "datasets/evaluation/raw",
        "datasets/evaluation/cropped",
        
        # Results
        "results/detection_plots",
        "results/ocr_testing",
        "results/pipeline_outputs",
        "results/comparisons",
        
        # Configs
        "configs",
        
        # Source
        "src"
    ]
    
    print("Creating project structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}")
    
    print("\nProject structure created successfully!")

def install_dependencies():
    """Install required packages"""
    print("\nInstalling dependencies...")
    
    # Check if requirements.txt exists
    if Path("requirements.txt").exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    else:
        print("requirements.txt not found. Creating default...")
        create_requirements_file()
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install additional packages based on OS
    import platform
    system = platform.system()
    
    if system == "Linux":
        # Install Tesseract on Linux
        try:
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "tesseract-ocr"], check=True)
            print("Installed Tesseract OCR")
        except:
            print("Note: Tesseract installation failed. You may need to install it manually.")
    
    print("\nDependencies installed successfully!")

def create_requirements_file():
    """Create requirements.txt if it doesn't exist"""
    requirements = """# Core & Deep Learning
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
transformers>=4.30.0
paddlepaddle>=2.5.0

# OCR Libraries
paddleocr>=2.7.0
easyocr>=1.7.0
pytesseract>=0.3.10
python-doctr>=0.5.0

# Computer Vision
opencv-python>=4.7.0
pillow>=10.0.0
albumentations>=1.3.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("Created requirements.txt")

def create_config_files():
    """Create configuration files"""
    print("\nCreating configuration files...")
    
    # Dataset configuration
    dataset_config = """# Dataset configuration for meter reading
path: ./datasets/detection  # dataset root dir
train: train/images
val: val/images
test: test/images

# Classes (3 classes: water meter, electric meter, digit region)
names:
  0: water_meter
  1: electric_meter
  2: digit_region

# Number of classes
nc: 3
"""
    
    with open("configs/dataset.yaml", "w") as f:
        f.write(dataset_config)
    
    # YOLOv8 training config
    yolov8_config = """# YOLOv8 Training Configuration
model: yolov8n.pt  # or yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
data: configs/dataset.yaml
epochs: 100
patience: 50
batch: 16
imgsz: 640
device: 0  # GPU device, use 'cpu' or '0,1,2,3' for multiple GPUs
workers: 8
project: models/detectors
name: yolov8
exist_ok: true  # overwrite existing project/name

# Optimization
lr0: 0.01  # initial learning rate
lrf: 0.01  # final learning rate factor
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# Augmentation (adjust based on Roboflow augmentations)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0

# Save settings
save_period: -1
save_dir: models/detectors/yolov8
"""
    
    with open("configs/yolov8.yaml", "w") as f:
        f.write(yolov8_config)
    
    print("Created config files in configs/ directory")

if __name__ == "__main__":
    print("=" * 60)
    print("Meter Reading Project Setup")
    print("=" * 60)
    
    create_project_structure()
    create_config_files()
    install_dependencies()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Place your Roboflow dataset in datasets/detection/")
    print("2. Update configs/dataset.yaml if needed")
    print("3. Run: python train_all.py")
    print("\nHappy training! 🚀")