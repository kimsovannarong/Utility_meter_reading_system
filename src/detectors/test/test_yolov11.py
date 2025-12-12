#!/usr/bin/env python3
"""
Test YOLOv11 model on test dataset - Minimal version
"""

import os
from pathlib import Path
from ultralytics import YOLO

# Find latest model
models = list(Path("models/detectors/yolov11").glob("**/weights/best.pt"))
if not models:
    print("No models found")
    exit(1)

model_path = max(models, key=os.path.getctime)

# Load and evaluate
model = YOLO(str(model_path))
results = model.val(data="datasets/detection/data.yaml", split="test", verbose=False)

# Print results
print("\n" + "="*50)
print(f"TEST RESULTS - {model_path.parent.parent.name}")
print("="*50)
print(f"mAP@50:    {results.box.map50:.4f}")
print(f"mAP@50-95: {results.box.map:.4f}")
print(f"Precision: {results.box.p.mean():.4f}")
print(f"Recall:    {results.box.r.mean():.4f}")
print("="*50)