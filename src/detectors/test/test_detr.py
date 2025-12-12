#!/usr/bin/env python3
"""
Test DETR model on test dataset
Compatible with HuggingFace Transformers DETR implementation
"""

import argparse
import torch
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
import json
import torchvision.transforms as T
from transformers import DetrForObjectDetection, DetrImageProcessor

def load_detr_model(model_path):
    """Load a fine-tuned DETR model"""
    print(f"Loading DETR model from: {model_path}")
    
    # Check if it's a local directory or HuggingFace model ID
    if Path(model_path).exists():
        # Load local model
        model = DetrForObjectDetection.from_pretrained(model_path)
        processor = DetrImageProcessor.from_pretrained(model_path)
    else:
        # Try loading from HuggingFace hub
        try:
            model = DetrForObjectDetection.from_pretrained(model_path)
            processor = DetrImageProcessor.from_pretrained(model_path)
        except:
            print(f"❌ Could not load model: {model_path}")
            print("   Provide either a local path or HuggingFace model ID")
            exit(1)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on device: {device}")
    return model, processor, device

def load_data_config(data_yaml_path):
    """Load dataset configuration"""
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get test images path
    base_path = Path(data_yaml_path).parent
    test_path = base_path / config.get('test', 'test/images')
    
    if not test_path.exists():
        print(f"❌ Test path not found: {test_path}")
        exit(1)
    
    return config, test_path

def load_class_names(data_config):
    """Extract class names from data config"""
    names = data_config.get('names', [])
    
    # Handle different formats
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    else:
        # Default if not specified
        return {0: 'object'}

def calculate_metrics(predictions, targets, iou_threshold=0.5):
    """Calculate mAP metrics for DETR predictions"""
    # Note: This is a simplified mAP calculation
    # For production, use torchmetrics or pycocotools
    
    if len(predictions) == 0:
        return {'mAP50': 0.0, 'mAP50_95': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    # Simplified implementation - for accurate mAP, use dedicated libraries
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred in predictions:
        if pred['score'] > 0.5:  # Confidence threshold
            # Check if matches any ground truth
            matched = False
            for target in targets:
                iou = calculate_iou(pred['bbox'], target['bbox'])
                if iou > iou_threshold and pred['label'] == target['label']:
                    matched = True
                    break
            
            if matched:
                true_positives += 1
            else:
                false_positives += 1
    
    false_negatives = len([t for t in targets if t['label'] not in 
                          [p['label'] for p in predictions if p['score'] > 0.5]])
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Simplified mAP - in reality should calculate at multiple IoU thresholds
    mAP50 = precision * recall if precision > 0 and recall > 0 else 0
    
    return {
        'mAP50': float(mAP50),
        'mAP50_95': float(mAP50 * 0.8),  # Approximation
        'precision': float(precision),
        'recall': float(recall),
    }

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def evaluate_detr_on_test_set(model, processor, device, test_path, class_names, num_samples=None):
    """Evaluate DETR model on test set"""
    print(f"Evaluating on test set: {test_path}")
    
    # Find all test images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(test_path.glob(ext)))
    
    if num_samples:
        test_images = test_images[:num_samples]
    
    print(f"Found {len(test_images)} test images")
    
    all_metrics = []
    
    for i, img_path in enumerate(test_images):
        if i % 10 == 0:
            print(f"  Processing image {i+1}/{len(test_images)}...")
        
        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process predictions
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.5
        )[0]
        
        # Convert to standard format
        predictions = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            predictions.append({
                'score': score.item(),
                'label': label.item(),
                'bbox': box.tolist(),
                'class_name': class_names.get(label.item(), f'class_{label.item()}')
            })
        
        # TODO: Load ground truth from labels file
        # For now, using empty targets (you need to implement this)
        targets = []  # Should load from labels/test/*.txt
        
        # Calculate metrics for this image
        metrics = calculate_metrics(predictions, targets)
        all_metrics.append(metrics)
    
    # Aggregate metrics
    if all_metrics:
        avg_metrics = {
            'mAP50': np.mean([m['mAP50'] for m in all_metrics]),
            'mAP50_95': np.mean([m['mAP50_95'] for m in all_metrics]),
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
        }
    else:
        avg_metrics = {'mAP50': 0.0, 'mAP50_95': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    return avg_metrics

def main():
    parser = argparse.ArgumentParser(description='Test DETR model on test dataset')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to DETR model or HuggingFace model ID (e.g., facebook/detr-resnet-50)')
    parser.add_argument('--data', type=str, default='datasets/detection/data.yaml',
                       help='Path to data.yaml')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of test samples to evaluate (default: all)')
    
    args = parser.parse_args()
    
    # Load model
    model, processor, device = load_detr_model(args.model)
    
    # Load dataset config
    data_config, test_path = load_data_config(args.data)
    class_names = load_class_names(data_config)
    
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {list(class_names.values())}")
    
    # Evaluate
    print(f"\n{'─' * 40}")
    print("Starting evaluation...")
    
    metrics = evaluate_detr_on_test_set(
        model, processor, device, test_path, class_names, args.samples
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"TEST RESULTS - DETR")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Device: {device}")
    print(f"{'─' * 40}")
    
    print(f"mAP@50:    {metrics['mAP50']:.4f}")
    print(f"mAP@50-95: {metrics['mAP50_95']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    
    # Calculate F1 score
    if metrics['precision'] > 0 and metrics['recall'] > 0:
        f1 = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        print(f"F1 Score:  {f1:.4f}")
    
    print(f"\n{'='*60}")
    print(f"⚠️  Note: This is a simplified mAP calculation")
    print(f"For accurate COCO-style mAP, use pycocotools with proper")
    print(f"ground truth loading and evaluation protocol.")
    print(f"{'='*60}")
    
    # Save results
    results = {
        'model': args.model,
        'data': args.data,
        'metrics': metrics,
        'class_names': class_names,
        'device': str(device)
    }
    
    results_file = Path(f"detr_results_{Path(args.model).name}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    # Install required packages:
    # pip install transformers torch torchvision pillow pyyaml numpy
    
    main()