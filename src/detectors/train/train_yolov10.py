#!/usr/bin/env python3
"""
YOLOv10 Training Script
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from train.base_trainer import BaseDetectorTrainer
from ultralytics import YOLO
import argparse

class YOLOv10Trainer(BaseDetectorTrainer):
    """YOLOv10 specific trainer"""
    
    def __init__(self, config_path=None):
        super().__init__('yolov10', config_path)
        
        # Load YOLOv10 model (will download if not exists)
        try:
            self.model = YOLO('yolov10n.pt')
        except:
            print("YOLOv10 model not found. Trying to download...")
            self.model = YOLO('yolov10n.pt')  # This will download
    
    def train(self, epochs=None, batch_size=None):
        """Train YOLOv10 model"""
        print(f"\n{'='*60}")
        print(f"YOLOv10 TRAINING")
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
        
        # Create experiment name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"yolov10_e{epochs}_b{batch_size}_{timestamp}"
        
        try:
            print(f"\nStarting YOLOv10 training...")
            start_time = time.time()
            
            # Train the model
            results = self.model.train(
                data=self.config['data'],
                epochs=epochs,
                batch=batch_size,
                imgsz=self.config.get('imgsz', 640),
                device='0' if self.device.type == 'cuda' else 'cpu',
                workers=2 if self.device.type == 'cpu' else 4,
                project=self.config.get('project', 'models/detectors/yolov10'),
                name=exp_name,
                exist_ok=True,
                verbose=True,
                # YOLOv10 specific
                nms=False,  # YOLOv10 is NMS-free
                # Optimization
                lr0=self.config.get('lr0', 0.01),
                lrf=self.config.get('lrf', 0.01),
                momentum=self.config.get('momentum', 0.937),
                weight_decay=self.config.get('weight_decay', 0.0005),
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
            
            print(f"\n✓ YOLOv10 training completed in {training_time:.1f} seconds")
            print(f"✓ Model saved to: {self.config.get('project')}/{exp_name}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ YOLOv10 training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate(self, split='val'):
        """Evaluate YOLOv10 model"""
        print(f"\nEvaluating YOLOv10 on {split} set...")
        
        # Find the latest model
        model_dir = Path(self.config.get('project', 'models/detectors/yolov10'))
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
            print(f"  Precision: {metrics.box.p:.3f}")
            print(f"  Recall: {metrics.box.r:.3f}")
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv10 for meter detection')
    parser.add_argument('--config', type=str, default='configs/yolov10.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate instead of train')
    
    args = parser.parse_args()
    
    trainer = YOLOv10Trainer(args.config)
    
    if args.evaluate:
        trainer.evaluate()
    else:
        trainer.train(epochs=args.epochs, batch_size=args.batch)
    
    trainer.cleanup()

if __name__ == "__main__":
    import time
    from datetime import datetime
    main()