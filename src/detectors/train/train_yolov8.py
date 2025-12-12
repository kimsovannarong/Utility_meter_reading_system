#!/usr/bin/env python3
"""
YOLOv8 Training Script
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from train.base_trainer import BaseDetectorTrainer
from ultralytics import YOLO
import argparse

class YOLOv8Trainer(BaseDetectorTrainer):
    """YOLOv8 specific trainer"""
    
    def __init__(self, config_path=None):
        super().__init__('yolov8', config_path)
        
        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')
    
    def train(self, epochs=None, batch_size=None):
        """Train YOLOv8 model"""
        print(f"\n{'='*60}")
        print(f"YOLOv8 TRAINING")
        print(f"{'='*60}")
        
        # Use provided values or config defaults
        epochs = epochs or self.config.get('epochs', 50)
        batch_size = batch_size or self.config.get('batch_size', 4)
        
        # Check dataset first
        if not self.check_dataset():
            return False
        
        print(f"Training Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {self.config.get('imgsz', 640)}")
        print(f"  Device: {'GPU' if self.device.type == 'cuda' else 'CPU'}")
        
        # Create experiment name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"yolov8_e{epochs}_b{batch_size}_{timestamp}"
        
        try:
            print(f"\nStarting YOLOv8 training...")
            start_time = time.time()
            
            # Train the model
            results = self.model.train(
                data=self.config['data'],
                epochs=epochs,
                batch=batch_size,
                imgsz=self.config.get('imgsz', 640),
                device='0' if self.device.type == 'cuda' else 'cpu',
                workers=2 if self.device.type == 'cpu' else 4,
                project=self.config.get('project', 'models/detectors/yolov8'),
                name=exp_name,
                exist_ok=True,
                verbose=True,
                # Optimization
                lr0=self.config.get('lr0', 0.01),
                lrf=self.config.get('lrf', 0.01),
                momentum=self.config.get('momentum', 0.937),
                weight_decay=self.config.get('weight_decay', 0.0005),
                # Augmentations
                hsv_h=self.config.get('hsv_h', 0.015),
                hsv_s=self.config.get('hsv_s', 0.7),
                hsv_v=self.config.get('hsv_v', 0.4),
                degrees=self.config.get('degrees', 0.0),
                translate=self.config.get('translate', 0.1),
                scale=self.config.get('scale', 0.5),
                fliplr=self.config.get('fliplr', 0.5),
                mosaic=self.config.get('mosaic', 1.0),
            )
            
            training_time = time.time() - start_time
            
            # Save results
            metrics = {
                'training_time_seconds': training_time,
                'training_time_minutes': training_time / 60,
                'epochs': epochs,
                'batch_size': batch_size,
                'experiment_name': exp_name
            }
            
            if hasattr(results, 'results_dict'):
                metrics.update(results.results_dict)
            
            self.save_experiment_log(metrics)
            
            print(f"\n✓ YOLOv8 training completed in {training_time:.1f} seconds")
            print(f"✓ Model saved to: {self.config.get('project')}/{exp_name}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ YOLOv8 training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate(self, split='val'):
        """Evaluate YOLOv8 model"""
        print(f"\nEvaluating YOLOv8 on {split} set...")
        
        # Find the latest model
        model_dir = Path(self.config.get('project', 'models/detectors/yolov8'))
        model_files = list(model_dir.glob('**/weights/best.pt'))
        
        if not model_files:
            print("No trained model found. Train first.")
            return None
        
        latest_model = max(model_files, key=os.path.getctime)
        model = YOLO(str(latest_model))
        
        # Run evaluation
        metrics = model.val(data=self.config['data'], split=split)
        
        print(f"\nEvaluation Results:")
        if hasattr(metrics, 'box'):
            print(f"  mAP@50: {metrics.box.map50:.3f}")
            print(f"  mAP@50-95: {metrics.box.map:.3f}")
            # Get mean precision and recall
            precision_mean = metrics.box.p.mean() if hasattr(metrics.box.p, 'mean') else metrics.box.p
            recall_mean = metrics.box.r.mean() if hasattr(metrics.box.r, 'mean') else metrics.box.r
        
            print(f"  Precision : {precision_mean:.3f}")
            print(f"  Recall : {recall_mean:.3f}")
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for meter detection')
    parser.add_argument('--config', type=str, default='configs/yolov8.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate instead of train')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Split to evaluate on')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = YOLOv8Trainer(args.config)
    
    if args.evaluate:
        trainer.evaluate(args.split)
    else:
        trainer.train(epochs=args.epochs, batch_size=args.batch)
    
    trainer.cleanup()

if __name__ == "__main__":
    import time
    from datetime import datetime
    main()