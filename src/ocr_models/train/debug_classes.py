# debug_classes.py
from ultralytics import YOLO
from pathlib import Path
import cv2

model = YOLO('models/detectors/YOLOv8/weights/best.pt')

# Test on first 3 validation images
val_dir = Path('datasets/detection/valid/images')
images = list(val_dir.glob('*.jpg'))[:3]

print("=" * 60)
print("MODEL CLASS MAPPING:")
for idx, name in model.names.items():
    print(f"  Class {idx}: '{name}'")
print("=" * 60)

for img_path in images:
    print(f"\n🔍 Testing: {img_path.name}")
    results = model(str(img_path), conf=0.2)
    
    if results[0].boxes is None:
        print("  No detections")
        continue
    
    print(f"  Found {len(results[0].boxes)} detections:")
    
    for i, box in enumerate(results[0].boxes):
        class_id = int(box.cls)
        conf = float(box.conf[0])
        class_name = model.names.get(class_id, f"unknown_{class_id}")
        
        print(f"    [{i}] Class {class_id} ('{class_name}'): conf={conf:.3f}")
        
        # Save to visually inspect
        img = cv2.imread(str(img_path))
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(f"inspect_class_{class_id}.jpg", crop)
        print(f"      Saved as inspect_class_{class_id}.jpg")
    
    print("-" * 40)

print("\n" + "=" * 60)
print("INSTRUCTIONS:")
print("1. Look at the 'inspect_class_X.jpg' files")
print("2. Which class contains DIGITS?")
print("3. That's the class ID you need in your script!")
print("=" * 60)