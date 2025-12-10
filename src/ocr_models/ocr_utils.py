#!/usr/bin/env python3
"""
Shared utilities for OCR models
"""

import os
import cv2
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
import json
from datetime import datetime

class BaseOCRTrainer(ABC):
    """Abstract base class for all OCR trainers"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            f'models/ocr/{self.model_name}',
            f'results/ocr/{self.model_name}',
            'datasets/digits/train',
            'datasets/digits/test'
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def preprocess_digit_image(self, image):
        """Preprocess digit image for OCR"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        # Resize to consistent size
        gray = cv2.resize(gray, (32, 32))
        
        # Normalize
        gray = gray.astype(np.float32) / 255.0
        
        return gray
    
    def extract_digits_from_meter(self, image, detector_results):
        """Extract digit regions from meter detection results"""
        # Find digit region boxes (class 2 in your dataset)
        digit_boxes = []
        for box in detector_results:
            if box['class'] == 2:  # digit_region class
                digit_boxes.append(box)
        
        if not digit_boxes:
            return []
        
        # Sort boxes from left to right
        digit_boxes.sort(key=lambda x: x['bbox'][0])
        
        # Extract and preprocess each digit
        digits = []
        for box in digit_boxes:
            x1, y1, x2, y2 = map(int, box['bbox'])
            digit_img = image[y1:y2, x1:x2]
            
            if digit_img.size > 0:
                preprocessed = self.preprocess_digit_image(digit_img)
                digits.append({
                    'image': preprocessed,
                    'bbox': box['bbox'],
                    'confidence': box['confidence']
                })
        
        return digits
    
    @abstractmethod
    def train(self, train_data_path):
        """Train the OCR model"""
        pass
    
    @abstractmethod
    def recognize(self, image):
        """Recognize digits in image"""
        pass
    
    def save_results(self, accuracy, test_samples):
        """Save OCR evaluation results"""
        result_data = {
            'model': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'test_samples': test_samples,
            'parameters': self.get_model_info()
        }
        
        result_file = f"results/ocr/{self.model_name}/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        print(f"✓ Results saved to: {result_file}")
    
    @abstractmethod
    def get_model_info(self):
        """Get model information"""
        pass