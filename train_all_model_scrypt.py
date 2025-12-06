#!/usr/bin/env python3
"""
Complete Meter Reading Training System
Trains: YOLOv8, YOLOv10, DETR with epochs 25, 50, 75, 100
"""

import os
import sys
import yaml
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ultralytics import YOLO

# For DETR
try:
    from transformers import DetrForObjectDetection, DetrImageProcessor
    from transformers import DetrConfig
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False
    print("Note: DETR dependencies not installed. Install with: pip install transformers")

class MeterDetectionTrainer:
    def __init__(self):
        self.device = self.setup_device()
        self.setup_directories()
        
    def setup_device(self):
        """Setup training device (CPU/GPU)"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("✓ Using CPU for training")
        return device
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'models/detectors/yolov8',
            'models/detectors/yolov10',
            'models/detectors/detr',
            'experiments/yolov8',
            'experiments/yolov10',
            'experiments/detr',
            'results/training_logs',
            'datasets/detection_cache'  # For DETR cached features
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print("✓ Directory structure created")
    
    def check_dataset(self):
        """Verify dataset exists and is properly formatted"""
        print("\n" + "="*60)
        print("DATASET VERIFICATION")
        print("="*60)
        
        data_yaml = 'datasets/detection/data.yaml'
        if not os.path.exists(data_yaml):
            print(f"✗ ERROR: Dataset config not found: {data_yaml}")
            return False
        
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        print(f"✓ Dataset config loaded")
        print(f"  Classes: {data_config.get('names', [])}")
        print(f"  Number of classes: {data_config.get('nc', 0)}")
        
        # Check train/val directories
        required_paths = [
            'datasets/detection/train/images',
            'datasets/detection/train/labels',
            'datasets/detection/valid/images',
            'datasets/detection/valid/labels'
        ]
        
        for path in required_paths:
            if os.path.exists(path):
                files = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg', '.txt'))])
                print(f"✓ {path}: {files} files")
            else:
                print(f"✗ {path}: NOT FOUND")
                return False
        
        return True
    
    def train_yolov8(self, epochs, batch_size, experiment_id):
        """Train YOLOv8 model"""
        print(f"\n{'='*60}")
        print(f"YOLOv8 TRAINING")
        print(f"Epochs: {epochs} | Batch: {batch_size}")
        print(f"{'='*60}")
        
        # Configuration
        config = {
            'model': 'yolov8n.pt',
            'data': 'datasets/detection/data.yaml',
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': 640,
            'device': 'cpu' if self.device.type == 'cpu' else '0',
            'workers': 2 if self.device.type == 'cpu' else 4,
            'project': 'models/detectors/yolov8',
            'name': f'yolov8_e{epochs}_b{batch_size}',
            'exist_ok': True,
            'verbose': True,
            # Optimization
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            # Augmentations
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'fliplr': 0.5,
            'mosaic': 1.0,
        }
        
        try:
            # Load and train model
            model = YOLO(config['model'])
            
            print(f"Starting YOLOv8 training for {epochs} epochs...")
            start_time = time.time()
            
            results = model.train(
                data=config['data'],
                epochs=config['epochs'],
                batch=config['batch'],
                imgsz=config['imgsz'],
                device=config['device'],
                workers=config['workers'],
                project=config['project'],
                name=config['name'],
                exist_ok=config['exist_ok'],
                verbose=config['verbose'],
                lr0=config['lr0'],
                lrf=config['lrf'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay'],
                warmup_epochs=config['warmup_epochs'],
                hsv_h=config['hsv_h'],
                hsv_s=config['hsv_s'],
                hsv_v=config['hsv_v'],
                degrees=config['degrees'],
                translate=config['translate'],
                scale=config['scale'],
                fliplr=config['fliplr'],
                mosaic=config['mosaic'],
            )
            
            training_time = time.time() - start_time
            
            # Save results
            self.save_experiment_results(
                model_name='yolov8',
                epochs=epochs,
                batch_size=batch_size,
                experiment_id=experiment_id,
                config=config,
                training_time=training_time,
                results=results
            )
            
            print(f"\n✓ YOLOv8 training completed in {training_time:.1f} seconds")
            print(f"  Model saved to: {config['project']}/{config['name']}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ YOLOv8 training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_yolov10(self, epochs, batch_size, experiment_id):
        """Train YOLOv10 model"""
        print(f"\n{'='*60}")
        print(f"YOLOv10 TRAINING")
        print(f"Epochs: {epochs} | Batch: {batch_size}")
        print(f"{'='*60}")
        
        # Configuration (similar to YOLOv8 with adjustments)
        config = {
            'model': 'yolov10n.pt',
            'data': 'datasets/detection/data.yaml',
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': 640,
            'device': 'cpu' if self.device.type == 'cpu' else '0',
            'workers': 2 if self.device.type == 'cpu' else 4,
            'project': 'models/detectors/yolov10',
            'name': f'yolov10_e{epochs}_b{batch_size}',
            'exist_ok': True,
            'verbose': True,
            # YOLOv10 specific
            'nms': False,  # YOLOv10 is NMS-free
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
        }
        
        try:
            # Load YOLOv10 model
            model = YOLO(config['model'])
            
            print(f"Starting YOLOv10 training for {epochs} epochs...")
            start_time = time.time()
            
            results = model.train(
                data=config['data'],
                epochs=config['epochs'],
                batch=config['batch'],
                imgsz=config['imgsz'],
                device=config['device'],
                workers=config['workers'],
                project=config['project'],
                name=config['name'],
                exist_ok=config['exist_ok'],
                verbose=config['verbose'],
                lr0=config['lr0'],
                lrf=config['lrf'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay'],
            )
            
            training_time = time.time() - start_time
            
            # Save results
            self.save_experiment_results(
                model_name='yolov10',
                epochs=epochs,
                batch_size=batch_size,
                experiment_id=experiment_id,
                config=config,
                training_time=training_time,
                results=results
            )
            
            print(f"\n✓ YOLOv10 training completed in {training_time:.1f} seconds")
            print(f"  Model saved to: {config['project']}/{config['name']}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ YOLOv10 training failed: {e}")
            # If YOLOv10 fails, it might not be available. Try to download it.
            print("Attempting to download YOLOv10...")
            try:
                # Try to download YOLOv10
                model = YOLO('yolov10n.pt')
                print("YOLOv10 downloaded successfully. Please run training again.")
            except:
                print("Could not download YOLOv10. You may need to install it manually.")
            return False
    
    def train_detr(self, epochs, batch_size, experiment_id):
        """Train DETR model"""
        print(f"\n{'='*60}")
        print(f"DETR TRAINING")
        print(f"Epochs: {epochs} | Batch: {batch_size}")
        print(f"{'='*60}")
        
        if not DETR_AVAILABLE:
            print("DETR dependencies not installed.")
            print("Install with: pip install transformers torchvision")
            return False
        
        try:
            # Load dataset configuration
            with open('datasets/detection/data.yaml', 'r') as f:
                data_config = yaml.safe_load(f)
            
            num_classes = data_config.get('nc', 3)  # Default to 3 classes
            
            print(f"Setting up DETR for {num_classes} classes...")
            
            # Load DETR model
            model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            
            # Move model to device
            model = model.to(self.device)
            
            # Setup image processor
            image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            
            # Note: DETR requires custom dataset preparation
            # This is a simplified version - you'll need to implement:
            # 1. Convert YOLO format to COCO format
            # 2. Create custom dataset class
            # 3. Implement training loop
            
            print("\nDETR training setup requires additional steps:")
            print("1. Convert YOLO annotations to COCO format")
            print("2. Create custom DETR dataset class")
            print("3. Implement training loop with transformers")
            
            print("\nFor now, saving DETR configuration...")
            
            # Save DETR configuration
            config = {
                'model': 'facebook/detr-resnet-50',
                'epochs': epochs,
                'batch_size': batch_size,
                'num_classes': num_classes,
                'device': str(self.device),
                'experiment_id': experiment_id,
                'timestamp': datetime.now().isoformat()
            }
            
            config_file = f"experiments/detr/detr_config_e{epochs}_b{batch_size}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            
            print(f"✓ DETR configuration saved to: {config_file}")
            print("\nTo implement DETR training, you need to:")
            print("1. Convert your dataset to COCO format")
            print("2. Use the transformers library training example")
            print("3. Adjust for your specific meter detection task")
            
            return False  # Return False since full training not implemented
            
        except Exception as e:
            print(f"\n✗ DETR setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_experiment_results(self, model_name, epochs, batch_size, experiment_id, config, training_time, results=None):
        """Save experiment results and metrics"""
        
        result_data = {
            'model': model_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'experiment_id': experiment_id,
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'device': str(self.device),
        }
        
        # Add metrics if available
        if results and hasattr(results, 'results_dict'):
            result_data['metrics'] = results.results_dict
        
        # Save to JSON
        result_file = f"experiments/{model_name}/exp_{experiment_id}_e{epochs}_b{batch_size}.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        print(f"✓ Experiment results saved to: {result_file}")
    
    def run_epoch_experiments(self, model_type, epochs_list, batch_size):
        """Run experiments for different epochs"""
        
        print(f"\n{'#'*60}")
        print(f"RUNNING {model_type.upper()} EPOCH EXPERIMENTS")
        print(f"Epochs to test: {epochs_list}")
        print(f"Batch size: {batch_size}")
        print(f"{'#'*60}")
        
        results = {}
        
        for i, epochs in enumerate(epochs_list, 1):
            print(f"\n▶ Experiment {i}/{len(epochs_list)}: {epochs} epochs")
            
            if model_type == 'yolov8':
                success = self.train_yolov8(epochs, batch_size, experiment_id=i)
            elif model_type == 'yolov10':
                success = self.train_yolov10(epochs, batch_size, experiment_id=i)
            elif model_type == 'detr':
                success = self.train_detr(epochs, batch_size, experiment_id=i)
            else:
                print(f"Unknown model type: {model_type}")
                continue
            
            results[epochs] = success
            
            # Brief pause between experiments
            if i < len(epochs_list):
                print("\n" + "-"*40)
                print(f"Waiting before next experiment...")
                print("-"*40)
                time.sleep(2)  # 2 second pause
        
        return results
    
    def compare_results(self):
        """Compare results from all experiments"""
        print(f"\n{'='*60}")
        print("EXPERIMENT COMPARISON")
        print(f"{'='*60}")
        
        # This would load all experiment results and create comparison
        print("Comparison report generation not implemented in this version.")
        print("Check individual experiment files in experiments/ directory.")

def main():
    parser = argparse.ArgumentParser(description='Train meter detection models with epoch experiments')
    parser.add_argument('--model', type=str, default='yolov8',
                       choices=['yolov8', 'yolov10', 'detr', 'all'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, nargs='+', default=[25, 50, 75, 100],
                       help='List of epochs to test')
    parser.add_argument('--batch', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with minimal epochs (10, 25)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("METER DETECTION MODEL TRAINING")
    print("="*60)
    
    # Quick test mode
    if args.quick:
        args.epochs = [10, 25]
        print("Quick test mode enabled")
    
    # Initialize trainer
    trainer = MeterDetectionTrainer()
    
    # Check dataset
    if not trainer.check_dataset():
        print("\n✗ Dataset check failed. Please fix issues before training.")
        return
    
    print(f"\nTraining Configuration:")
    print(f"  Model(s): {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    
    # Determine which models to train
    if args.model == 'all':
        models_to_train = ['yolov8', 'yolov10', 'detr']
    else:
        models_to_train = [args.model]
    
    all_results = {}
    
    # Train each model
    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"STARTING {model_type.upper()} TRAINING")
        print(f"{'='*60}")
        
        results = trainer.run_epoch_experiments(
            model_type=model_type,
            epochs_list=args.epochs,
            batch_size=args.batch
        )
        
        all_results[model_type] = results
        
        print(f"\n{model_type.upper()} experiments completed!")
    
    # Generate comparison
    trainer.compare_results()
    
    print("\n" + "="*60)
    print("TRAINING PROCESS COMPLETE")
    print("="*60)
    print("\n📁 Output Structure:")
    print("  models/detectors/yolov8/ - YOLOv8 trained models")
    print("  models/detectors/yolov10/ - YOLOv10 trained models")
    print("  models/detectors/detr/ - DETR configurations")
    print("  experiments/ - Experiment logs and results")
    print("\n🔍 Next Steps:")
    print("  1. Check model performance in experiments/ folder")
    print("  2. Select best model based on mAP@50")
    print("  3. Test selected model on validation set")
    print("  4. Implement OCR pipeline for digit reading")

if __name__ == "__main__":
    main()