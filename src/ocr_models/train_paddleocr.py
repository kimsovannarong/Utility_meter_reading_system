#!/usr/bin/env python3
"""
PaddleOCR Training Script
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ocr_models.ocr_utils import BaseOCRTrainer
import argparse

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

class PaddleOCRTrainer(BaseOCRTrainer):
    """PaddleOCR specific trainer"""
    
    def __init__(self):
        super().__init__('paddleocr')
        
        if not PADDLEOCR_AVAILABLE:
            print("PaddleOCR not installed. Install with:")
            print("pip install paddleocr paddlepaddle")
            self.model = None
            return
        
        # Initialize PaddleOCR
        self.model = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            det=False,  # We already have detection from YOLO/DETR
            rec=True,   # Only recognition needed
            show_log=False
        )
    
    def train(self, train_data_path=None):
        """Train PaddleOCR model (fine-tuning)"""
        print(f"\n{'='*60}")
        print(f"PADDLEOCR SETUP")
        print(f"{'='*60}")
        
        if not PADDLEOCR_AVAILABLE:
            return False
        
        print("PaddleOCR uses pre-trained models.")
        print("For custom training, you need to:")
        print("1. Prepare digit images with labels")
        print("2. Use PaddleOCR's training tools")
        print("3. Fine-tune on your meter digits")
        
        # For now, just test the model
        return self.test_sample()
    
    def recognize(self, image):
        """Recognize digits using PaddleOCR"""
        if not PADDLEOCR_AVAILABLE or self.model is None:
            return []
        
        # Run OCR
        result = self.model.ocr(image, cls=True)
        
        if result is None or len(result) == 0:
            return []
        
        digits = []
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            
            # Extract only digits
            digit_text = ''.join(filter(str.isdigit, text))
            
            if digit_text:
                digits.append({
                    'text': digit_text,
                    'confidence': confidence,
                    'bbox': line[0]
                })
        
        return digits
    
    def test_sample(self):
        """Test PaddleOCR on sample image"""
        print("\nTesting PaddleOCR on sample digit...")
        
        # Create a sample digit image for testing
        import numpy as np
        sample_img = np.ones((50, 200, 3), dtype=np.uint8) * 255
        cv2.putText(sample_img, "12345", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        result = self.recognize(sample_img)
        
        if result:
            print(f"✓ PaddleOCR detected: {result[0]['text']}")
            return True
        else:
            print("✗ PaddleOCR test failed")
            return False
    
    def get_model_info(self):
        """Get PaddleOCR model information"""
        return {
            'type': 'PaddleOCR',
            'language': 'en',
            'detection': False,
            'recognition': True,
            'angle_classification': True
        }

def main():
    parser = argparse.ArgumentParser(description='Setup PaddleOCR for digit recognition')
    parser.add_argument('--test', action='store_true',
                       help='Test PaddleOCR on sample image')
    
    args = parser.parse_args()
    
    trainer = PaddleOCRTrainer()
    
    if args.test:
        trainer.test_sample()
    else:
        trainer.train()

if __name__ == "__main__":
    import cv2
    import numpy as np
    main()