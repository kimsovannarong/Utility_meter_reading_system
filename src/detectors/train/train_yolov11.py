#!/usr/bin/env python3
"""
Complete YOLOv11 Training Script with MPS support
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from train.base_trainer import BaseDetectorTrainer
from ultralytics import YOLO
import argparse
import traceback
import time
from datetime import datetime
import torch

class YOLOv11Trainer(BaseDetectorTrainer):
    """YOLOv11 specific trainer with MPS support"""
    
    def __init__(self, config_path=None):
        super().__init__('yolov11', config_path)  # Changed to 'yolov11'
        
        # Initialize model with robust loading
        self.model = self._load_model()
    
    def _load_model(self):
        """Robust model loading with multiple fallback options"""
        print("Loading YOLOv11 model...")
        
        # List of possible model names to try
        model_names = [
            'yolov11n.pt',      # Official naming
            'yolo11n.pt',       # Alternative naming
            'yolo11n.yaml',     # Architecture from scratch
        ]
        
        for model_name in model_names:
            try:
                print(f"  Trying: {model_name}")
                
                # Try with download option for .pt files
                if model_name.endswith('.pt'):
                    model = YOLO(model_name, download='ultralytics/assets')
                else:
                    model = YOLO(model_name)
                
                print(f"  ✓ Successfully loaded: {model_name}")
                
                # Get model info
                if hasattr(model, 'names'):
                    print(f"  Model has {len(model.names)} classes")
                if hasattr(model, 'model'):
                    params = sum(p.numel() for p in model.model.parameters())
                    print(f"  Parameters: {params:,}")
                
                return model
                
            except Exception as e:
                print(f"  ✗ Failed: {str(e)[:100]}...")
                continue
        
        # If all attempts fail, provide clear instructions
        print("\n" + "="*60)
        print("COULD NOT LOAD YOLOv11 MODEL")
        print("="*60)
        print("Possible solutions:")
        print("1. Update ultralytics: pip install ultralytics --upgrade")
        print("2. Manually download the model:")
        print("   wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov11n.pt")
        print("3. Use a different model size:")
        print("   yolov11s.pt, yolov11m.pt, yolov11l.pt, yolov11x.pt")
        print("="*60)
        raise FileNotFoundError("Could not load any YOLOv11 model")
    
    def _get_device_string(self):
        """Get formatted device string for display"""
        if self.device.type == 'cuda':
            return 'GPU (CUDA)'
        elif self.device.type == 'mps':
            return 'Apple Silicon (MPS)'
        else:
            return 'CPU'
    
    def _get_training_device(self):
        """Determine the device argument for training"""
        if self.device.type == 'mps':
            return 'mps'
        elif self.device.type == 'cuda':
            return '0'  # Use first CUDA device
        else:
            return 'cpu'
    
    def _get_workers(self):
        """Determine appropriate number of workers"""
        if self.device.type in ['cpu', 'mps']:
            return 2  # Fewer workers for CPU/MPS stability
        else:
            return 4  # More workers for GPU
    
    def train(self, epochs=None, batch_size=None):
        """Train YOLOv11 model"""
        print(f"\n{'='*60}")
        print(f"YOLOv11 TRAINING")
        print(f"{'='*60}")
        
        # Use provided values or config defaults
        epochs = epochs or self.config.get('epochs', 50)
        batch_size = batch_size or self.config.get('batch_size', 4)
        
        # Check dataset first
        if not self.check_dataset():
            return False
        
        # Get device information
        device_str = self._get_device_string()
        training_device = self._get_training_device()
        workers = self._get_workers()
        
        print(f"Training Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {self.config.get('imgsz', 640)}")
        print(f"  Device: {device_str}")
        print(f"  Training device: {training_device}")
        print(f"  Workers: {workers}")
        
        # Create experiment name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"yolov11_e{epochs}_b{batch_size}_{timestamp}"
        
        try:
            print(f"\nStarting YOLOv11 training...")
            start_time = time.time()
            
            # Train the model
            results = self.model.train(
                data=self.config['data'],
                epochs=epochs,
                batch=batch_size,
                imgsz=self.config.get('imgsz', 640),
                device=training_device,  # Dynamic device selection
                workers=workers,         # Dynamic worker selection
                project=self.config.get('project', 'models/detectors/yolov11'),
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
                # MPS-specific optimizations
                patience=self.config.get('patience', 100),  # Early stopping patience
                save_period=self.config.get('save_period', -1),  # Save every epoch
                single_cls=self.config.get('single_cls', False),  # Single class mode
            )
            
            training_time = time.time() - start_time
            
            # Save results
            metrics = {
                'training_time_seconds': training_time,
                'training_time_minutes': training_time / 60,
                'epochs': epochs,
                'batch_size': batch_size,
                'device': training_device,
                'experiment_name': exp_name,
                'model_name': 'yolov11n',
                'dataset': self.config.get('data', 'unknown'),
            }
            
            if hasattr(results, 'results_dict'):
                metrics.update(results.results_dict)
            
            self.save_experiment_log(metrics)
            
            print(f"\n{'='*60}")
            print(f"✓ YOLOv11 TRAINING COMPLETED")
            print(f"{'='*60}")
            print(f"  Time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
            print(f"  Model saved to: {self.config.get('project', 'models/detectors/yolov11')}/{exp_name}")
            print(f"  Best weights: {exp_name}/weights/best.pt")
            print(f"  Last weights: {exp_name}/weights/last.pt")
            print(f"{'='*60}")
            
            return True
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"✗ YOLOv11 TRAINING FAILED")
            print(f"{'='*60}")
            print(f"Error: {e}")
            print(f"\nFull traceback:")
            traceback.print_exc()
            print(f"{'='*60}")
            return False
    
    def evaluate(self, split='val', model_path=None):
        """Evaluate YOLOv11 model"""
        print(f"\n{'='*60}")
        print(f"EVALUATING YOLOv11 ON {split.upper()} SET")
        print(f"{'='*60}")
        
        # Find the model to evaluate
        if model_path:
            eval_model_path = Path(model_path)
            if not eval_model_path.exists():
                print(f"✗ Specified model not found: {model_path}")
                return None
        else:
            # Find the latest model
            model_dir = Path(self.config.get('project', 'models/detectors/yolov11'))
            model_files = list(model_dir.glob('**/weights/best.pt'))
            
            if not model_files:
                print("✗ No trained model found. Train first or specify --model-path")
                return None
            
            eval_model_path = max(model_files, key=os.path.getctime)
        
        print(f"Evaluating model: {eval_model_path}")
        
        # Load the specific model
        model = YOLO(str(eval_model_path))
        
        # Run evaluation
        try:
            metrics = model.val(
                data=self.config['data'], 
                split=split,
                verbose=True
            )
            
            print(f"\n{'='*60}")
            print(f"EVALUATION RESULTS")
            print(f"{'='*60}")
            
            if hasattr(metrics, 'box'):
                print(f"\nOverall Metrics:")
                print(f"{'-'*40}")
                print(f"  mAP@50:     {metrics.box.map50:.4f}")
                print(f"  mAP@50-95:  {metrics.box.map:.4f}")
                
                # Calculate mean precision and recall
                if hasattr(metrics.box.p, 'mean'):
                    precision_mean = metrics.box.p.mean()
                    recall_mean = metrics.box.r.mean()
                else:
                    precision_mean = metrics.box.p
                    recall_mean = metrics.box.r
                
                print(f"  Precision:  {precision_mean:.4f}")
                print(f"  Recall:     {recall_mean:.4f}")
                
                # Save evaluation results
                eval_metrics = {
                    'split': split,
                    'model_path': str(eval_model_path),
                    'mAP50': float(metrics.box.map50),
                    'mAP50_95': float(metrics.box.map),
                    'precision': float(precision_mean),
                    'recall': float(recall_mean),
                    'evaluation_timestamp': datetime.now().isoformat(),
                }
                
                self.save_experiment_log(eval_metrics, suffix=f"_eval_{split}")
            
            return metrics
            
        except Exception as e:
            print(f"\n✗ Evaluation failed: {e}")
            return None
    
    def predict(self, image_path, model_path=None, confidence=0.5):
        """Run inference on an image"""
        print(f"\nRunning prediction on: {image_path}")
        
        # Load model for prediction
        if model_path:
            predict_model = YOLO(model_path)
        else:
            # Try to find the latest model
            model_dir = Path(self.config.get('project', 'models/detectors/yolov11'))
            model_files = list(model_dir.glob('**/weights/best.pt'))
            
            if not model_files:
                print("No trained model found for prediction")
                return None
            
            latest_model = max(model_files, key=os.path.getctime)
            predict_model = YOLO(str(latest_model))
        
        # Run prediction
        results = predict_model.predict(
            source=image_path,
            conf=confidence,
            save=True,
            save_txt=True,
            project='predictions',
            name=f"yolov11_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Train, evaluate, or predict with YOLOv11')
    parser.add_argument('--config', type=str, default='configs/yolov11.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate instead of train')
    parser.add_argument('--predict', type=str, default=None,
                       help='Path to image for prediction')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Split to evaluate on')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to specific model for evaluation/prediction')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for prediction')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = YOLOv11Trainer(args.config)
    
    # Choose action
    if args.predict:
        trainer.predict(args.predict, args.model_path, args.confidence)
    elif args.evaluate:
        trainer.evaluate(args.split, args.model_path)
    else:
        trainer.train(epochs=args.epochs, batch_size=args.batch)
    
    trainer.cleanup()

if __name__ == "__main__":
    main()