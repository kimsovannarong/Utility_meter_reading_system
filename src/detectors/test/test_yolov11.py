#!/usr/bin/env python3
"""
Test YOLO model on test dataset
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test YOLOv11 model on test dataset')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model weights (e.g., models/detectors/yolov11/train/weights/best.pt)')
    parser.add_argument('--data', type=str, default='datasets/detection/data.yaml',
                       help='Path to data.yaml')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed validation output')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {args.model}")
        print(f"   Please provide a valid path to best.pt")
        exit(1)
    
    print(f"Testing model: {model_path}")
    
    # Load and evaluate
    model = YOLO(str(model_path))
    results = model.val(
        data=args.data, 
        split="test", 
        verbose=False
    )
    
    # Print results
    print("\n" + "="*60)
    print(f"TEST RESULTS - {model_path.parent.parent.name}")
    print("="*60)
    print(f"Model path: {model_path}")
    print(f"Data config: {args.data}")
    print("-" * 40)
    
    if hasattr(results, 'box'):
        print(f"mAP@50:    {results.box.map50:.4f}")
        print(f"mAP@50-95: {results.box.map:.4f}")
        
        # Handle precision and recall (they are arrays)
        if hasattr(results.box.p, 'mean'):
            precision = results.box.p.mean()
            recall = results.box.r.mean()
        else:
            precision = results.box.p
            recall = results.box.r
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        
        # Show per-class results if available
        if hasattr(results, 'names'):
            print("\nPer-class results:")
            for i, class_name in results.names.items():
                if i < len(results.box.ap50):
                    print(f"  {class_name}: AP@50={results.box.ap50[i]:.4f}")
    
    print("="*60)

if __name__ == "__main__":
    main()