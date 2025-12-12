#!/usr/bin/env python3
"""
Dataset utilities for DETR training
"""

import os
import json
import yaml
import cv2
from pathlib import Path
from datetime import datetime
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class YoloToCocoConverter:
    """Convert YOLO format to COCO format for DETR"""
    
    def __init__(self, yolo_dir="datasets/detection"):
        self.yolo_dir = Path(yolo_dir)
        self.coco_dir = Path("datasets/coco_format")
        self.coco_dir.mkdir(parents=True, exist_ok=True)
        
        # Load YOLO config
        with open(self.yolo_dir / "data.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.categories = self.config['names']
        self.num_classes = self.config['nc']
        
    def convert_split(self, split='train'):
        """Convert a split (train/val/test) to COCO format"""
        
        print(f"Converting {split} split...")
        
        # COCO data structure
        coco_data = {
            "info": {
                "year": 2024,
                "version": "1.0",
                "description": "Meter Reading Dataset",
                "contributor": "",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [{"id": 1, "name": "Academic", "url": ""}],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        for i, cat_name in enumerate(self.categories):
            coco_data["categories"].append({
                "id": i,
                "name": cat_name,
                "supercategory": "meter"
            })
        
        # Paths
        img_dir = self.yolo_dir / split / "images"
        label_dir = self.yolo_dir / split / "labels"
        
        if not img_dir.exists():
            print(f"Warning: {img_dir} not found, skipping...")
            return None
        
        # Get image files
        image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        
        image_id = 0
        annotation_id = 0
        
        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            # Add image to COCO
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
                "license": 1,
                "date_captured": datetime.now().isoformat()
            })
            
            # Load corresponding label file
            label_path = label_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        box_w = float(parts[3])
                        box_h = float(parts[4])
                        
                        # Convert YOLO to COCO format
                        x_min = (x_center - box_w/2) * w
                        y_min = (y_center - box_h/2) * h
                        width = box_w * w
                        height = box_h * h
                        
                        # Ensure bbox is within image bounds
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        width = min(width, w - x_min)
                        height = min(height, h - y_min)
                        
                        # Add annotation
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [float(x_min), float(y_min), float(width), float(height)],
                            "area": float(width * height),
                            "iscrowd": 0
                        })
                        
                        annotation_id += 1
            
            image_id += 1
        
        # Save COCO JSON
        output_file = self.coco_dir / f"{split}.json"
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"✓ {split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
        
        return output_file
    
    def convert_all(self):
        """Convert all splits"""
        
        print("="*60)
        print("CONVERTING YOLO TO COCO FORMAT")
        print("="*60)
        
        print(f"Dataset: {self.yolo_dir}")
        print(f"Classes: {self.categories}")
        
        splits = ['train', 'val', 'test']
        results = {}
        
        for split in splits:
            result = self.convert_split(split)
            if result:
                results[split] = str(result)
        
        # Create dataset info file
        info = {
            "dataset": "meter_reading",
            "num_classes": self.num_classes,
            "classes": self.categories,
            "coco_files": results,
            "converted_date": datetime.now().isoformat()
        }
        
        info_file = self.coco_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✓ Conversion complete!")
        print(f"✓ Dataset info: {info_file}")
        
        return True

class MeterCocoDataset(Dataset):
    """Dataset for COCO format meter images for DETR"""
    
    def __init__(self, coco_json_path, image_dir, processor):
        self.coco_json_path = Path(coco_json_path)
        self.image_dir = Path(image_dir)
        self.processor = processor
        
        # Load COCO data
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.image_id_to_info = {}
        for img in self.coco_data['images']:
            self.image_id_to_info[img['id']] = img
        
        # Group annotations by image_id
        self.annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        self.image_ids = list(self.image_id_to_info.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_id_to_info[image_id]
        
        # Load image
        img_path = self.image_dir / image_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations for this image
        annotations = self.annotations_by_image.get(image_id, [])
        
        # Prepare annotations
        boxes = []
        labels = []
        
        for ann in annotations:
            # COCO bbox: [x_min, y_min, width, height]
            x_min, y_min, width, height = ann['bbox']
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            labels.append(ann['category_id'])
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4))
            labels = torch.zeros(0, dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        
        # Process with DETR processor
        encoding = self.processor(
            images=image,
            annotations=[{
                'boxes': boxes,
                'class_labels': labels
            }],
            return_tensors="pt"
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        return encoding