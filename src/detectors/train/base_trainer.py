#!/usr/bin/env python3
"""
Base trainer for all detection models
"""

import os
import json
import yaml
import time
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
import torch

class BaseDetectorTrainer(ABC):
    """Abstract base class for all detector trainers"""
    
    def __init__(self, model_name, config_path=None):
        self.model_name = model_name
        self.device = self.setup_device()
        self.config = self._load_config(config_path)
        
        # Create output directories
        self._setup_directories()
    
    def setup_device(self):
        """Setup training device (CPU/GPU)"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("✓ Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device('cpu')
            print("✓ Using CPU for training")
        return device

    
    def _load_config(self, config_path=None):
        """Load configuration from YAML file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Return default configuration
        return {
            'data': 'datasets/detection/data.yaml',
            'epochs': 50,
            'batch_size': 4,
            'imgsz': 640,
            'project': f'models/detectors/{self.model_name}',
            'save_period': 10,
            'patience': 30,
            'optimizer': 'SGD',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005
        }
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            f'models/detectors/{self.model_name}',
            f'experiments/{self.model_name}',
            f'results/{self.model_name}/training',
            f'results/{self.model_name}/validation'
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def check_dataset(self):
        """Verify dataset exists and is properly formatted"""
        print(f"\n{'='*60}")
        print(f"DATASET VERIFICATION for {self.model_name.upper()}")
        print(f"{'='*60}")
        
        data_yaml = self.config.get('data', 'datasets/detection/data.yaml')
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
            'datasets/detection/valid/labels',
            # 'datasets/detection/test/images',
            # 'datasets/detection/test/labels',
        ]
        
        for path in required_paths:
            if os.path.exists(path):
                files = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg', '.txt'))])
                print(f"✓ {path}: {files} files")
            else:
                print(f"✗ {path}: NOT FOUND")
                return False
        
        return True
    
    @abstractmethod
    def train(self, epochs=None, batch_size=None):
        """Train the model - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def evaluate(self, split='val'):
        """Evaluate the model - to be implemented by subclasses"""
        pass
    
    def save_experiment_log(self, metrics):
        """Save experiment results and metrics"""
        result_data = {
            'model': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'device': str(self.device),
            'metrics': metrics
        }
        
        # Save to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = f"experiments/{self.model_name}/exp_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        print(f"✓ Experiment log saved to: {result_file}")
        return result_file
    
    def cleanup(self):
        """Cleanup resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()