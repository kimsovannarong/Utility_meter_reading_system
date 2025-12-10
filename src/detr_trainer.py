import os
import yaml
from pathlib import Path

def check_dataset_format(dataset_path='datasets/detection'):
    """Check if your dataset is in YOLO format and prepare for conversion."""
    
    print("Checking your dataset format...")
    
    # Check for YOLO structure
    train_img_dir = Path(dataset_path) / 'train' / 'images'
    train_lbl_dir = Path(dataset_path) / 'train' / 'labels'
    
    if train_img_dir.exists() and train_lbl_dir.exists():
        print(f"✓ Found YOLO format structure in: {dataset_path}")
        
        # Count files
        images = list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png"))
        labels = list(train_lbl_dir.glob("*.txt"))
        
        print(f"  Training images: {len(images)}")
        print(f"  Training labels: {len(labels)}")
        
        # Check data.yaml
        data_yaml = Path(dataset_path) / 'data.yaml'
        if data_yaml.exists():
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
            print(f"✓ Found data.yaml with {data.get('nc', 0)} classes")
            print(f"  Classes: {data.get('names', [])}")
            return 'yolo', data.get('nc', 0), data.get('names', [])
        else:
            print("✗ data.yaml not found")
            return 'yolo', None, None
    else:
        print("Dataset structure not recognized.")
        return 'unknown', None, None

# Run the check
format_type, num_classes, class_names = check_dataset_format()

if format_type == 'yolo':
    print("\n" + "="*60)
    print("YOUR DATASET IS IN YOLO FORMAT")
    print("="*60)
    print("\nNext steps to train DETR:")
    print("1. For RT-DETR: Re-export your dataset from Roboflow in COCO format[citation:2]")
    print("2. For standard DETR: Write a script to convert YOLO → COCO format")
    print(f"\nYour task has {num_classes} classes: {class_names}")
    
    # Create a simple conversion starter template
    print("\n" + "="*60)
    print("COCO CONVERSION TEMPLATE")
    print("="*60)
    
    coco_template = '''# You need to create a script to convert YOLO to COCO format
# COCO format requires a JSON file with this structure:
'''
    print(coco_template)