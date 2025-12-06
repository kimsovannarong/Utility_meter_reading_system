#!/usr/bin/env python3
"""
DETR Training Implementation for Meter Detection
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transformers import DetrForObjectDetection, DetrImageProcessor
from transformers import DetrConfig

class MeterDataset(Dataset):
    """Custom dataset for meter detection in DETR format"""
    
    def __init__(self, root_dir, split='train', transform=None, processor=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.processor = processor
        
        # Load images and annotations
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.labels_dir = os.path.join(root_dir, split, 'labels')
        
        self.image_files = [f for f in os.listdir(self.images_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Load class names
        with open(os.path.join(root_dir, 'data.yaml'), 'r') as f:
            import yaml
            data_config = yaml.safe_load(f)
            self.class_names = data_config.get('names', [])
            self.num_classes = len(self.class_names)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations (YOLO format)
        label_path = os.path.join(self.labels_dir, 
                                 os.path.splitext(img_name)[0] + '.txt')
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert YOLO to COCO format
                        x_min = x_center - width/2
                        y_min = y_center - height/2
                        x_max = x_center + width/2
                        y_max = y_center + height/2
                        
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id)
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4))
            labels = torch.zeros(0, dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        
        # Prepare for DETR processor
        if self.processor:
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
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels
        }

class DETRTrainer:
    """DETR Trainer for Meter Detection"""
    
    def __init__(self, num_classes=3, device='cpu'):
        self.device = torch.device(device)
        self.num_classes = num_classes
        
        # Initialize processor and model
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        
        # Load model with correct number of classes
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
    def train(self, train_loader, val_loader, epochs=25):
        """Training loop"""
        
        self.model.train()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                pixel_values = batch["pixel_values"].to(self.device)
                pixel_mask = batch["pixel_mask"].to(self.device)
                labels = [{k: v.to(self.device) for k, v in t.items()} 
                         for t in batch["labels"]]
                
                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            
            # Validation
            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"Validation Loss: {val_loss:.4f}")
        
        return self.model
    
    def validate(self, val_loader):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                pixel_mask = batch["pixel_mask"].to(self.device)
                labels = [{k: v.to(self.device) for k, v in t.items()} 
                         for t in batch["labels"]]
                
                outputs = self.model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        return total_loss / len(val_loader)
    
    def save_model(self, path):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_classes': self.num_classes
        }, path)
        print(f"Model saved to {path}")

def main():
    """Example usage of DETR trainer"""
    
    # Initialize trainer
    trainer = DETRTrainer(num_classes=3, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = MeterDataset(
        root_dir='datasets/detection',
        split='train',
        processor=trainer.processor
    )
    
    val_dataset = MeterDataset(
        root_dir='datasets/detection',
        split='val',
        processor=trainer.processor
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: x)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Train model
    trained_model = trainer.train(train_loader, val_loader, epochs=25)
    
    # Save model
    trainer.save_model('models/detectors/detr/meter_detr_model.pth')

if __name__ == "__main__":
    main()