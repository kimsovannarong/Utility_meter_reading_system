#!/usr/bin/env python3
"""
DETR Training Script
"""

import os
import sys
import json
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from detectors.base_trainer import BaseDetectorTrainer
from detectors.dataset_utils import convert_yolo_to_coco, MeterCocoDataset

try:
    from transformers import DetrForObjectDetection, DetrImageProcessor
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False

import argparse
from torch.utils.data import DataLoader
import torch.nn as nn

class DETRTrainer(BaseDetectorTrainer):
    """DETR specific trainer"""
    
    def __init__(self, config_path=None):
        super().__init__('detr', config_path)
        
        if not DETR_AVAILABLE:
            print("DETR dependencies not installed. Install with:")
            print("pip install transformers torchvision")
            self.model = None
            return
        
        # Load dataset info to get number of classes
        self.num_classes = self._get_num_classes()
        
        # Initialize DETR components
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
    
    def _get_num_classes(self):
        """Get number of classes from dataset"""
        data_yaml = self.config.get('data', 'datasets/detection/data.yaml')
        with open(data_yaml, 'r') as f:
            import yaml
            data_config = yaml.safe_load(f)
        return data_config.get('nc', 3)
    
    def _prepare_datasets(self):
        """Prepare COCO format datasets for DETR"""
        
        # Convert dataset if needed
        coco_dir = Path("datasets/coco_format")
        if not coco_dir.exists() or not list(coco_dir.glob("*.json")):
            print("Converting dataset to COCO format...")
            converter = convert_yolo_to_coco.YoloToCocoConverter()
            converter.convert_all()
        
        # Load datasets
        train_dataset = MeterCocoDataset(
            coco_json_path=coco_dir / "train.json",
            image_dir=Path("datasets/detection/train/images"),
            processor=self.processor
        )
        
        val_dataset = MeterCocoDataset(
            coco_json_path=coco_dir / "val.json",
            image_dir=Path("datasets/detection/val/images"),
            processor=self.processor
        )
        
        return train_dataset, val_dataset
    
    def train(self, epochs=None, batch_size=None):
        """Train DETR model"""
        print(f"\n{'='*60}")
        print(f"DETR TRAINING")
        print(f"{'='*60}")
        
        if not DETR_AVAILABLE or self.model is None:
            return False
        
        # Use provided values or config defaults
        epochs = epochs or self.config.get('epochs', 25)
        batch_size = batch_size or self.config.get('batch_size', 2)
        
        # Check dataset
        if not self.check_dataset():
            return False
        
        # Prepare datasets
        train_dataset, val_dataset = self._prepare_datasets()
        
        # Custom collate function for DETR
        def collate_fn(batch):
            pixel_values = [item["pixel_values"] for item in batch]
            pixel_mask = [item["pixel_mask"] for item in batch]
            labels = [item["labels"] for item in batch]
            
            pixel_values = torch.stack(pixel_values)
            pixel_mask = torch.stack(pixel_mask)
            
            return {
                "pixel_values": pixel_values,
                "pixel_mask": pixel_mask,
                "labels": labels
            }
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        print(f"Training Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Num classes: {self.num_classes}")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # Training loop
        print(f"\nStarting DETR training for {epochs} epochs...")
        start_time = time.time()
        
        self.model.train()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                pixel_values = batch["pixel_values"].to(self.device)
                pixel_mask = batch["pixel_mask"].to(self.device)
                labels = batch["labels"]
                
                # Prepare labels for DETR
                formatted_labels = []
                for label_set in labels:
                    formatted_labels.append({
                        'class_labels': label_set['class_labels'].to(self.device),
                        'boxes': label_set['boxes'].to(self.device)
                    })
                
                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=formatted_labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"models/detectors/detr/checkpoint_epoch{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'num_classes': self.num_classes
                }, checkpoint_path)
                print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        training_time = time.time() - start_time
        
        # Save final model
        final_path = f"models/detectors/detr/meter_detr_final_e{epochs}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'epochs_trained': epochs,
            'training_time': training_time
        }, final_path)
        
        # Save experiment log
        metrics = {
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'epochs': epochs,
            'batch_size': batch_size,
            'final_loss': avg_loss,
            'model_path': final_path
        }
        
        self.save_experiment_log(metrics)
        
        print(f"\n✓ DETR training completed in {training_time:.1f} seconds")
        print(f"✓ Model saved to: {final_path}")
        
        return True
    
    def evaluate(self, split='val'):
        """Evaluate DETR model"""
        print(f"\nEvaluating DETR on {split} set...")
        # Implementation for DETR evaluation
        pass

def main():
    parser = argparse.ArgumentParser(description='Train DETR for meter detection')
    parser.add_argument('--config', type=str, default='configs/detr.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size')
    
    args = parser.parse_args()
    
    trainer = DETRTrainer(args.config)
    trainer.train(epochs=args.epochs, batch_size=args.batch)
    trainer.cleanup()

if __name__ == "__main__":
    import time
    main()