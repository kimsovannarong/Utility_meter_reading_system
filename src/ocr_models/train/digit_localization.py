""" 1
Generates cropped digit images using your trained YOLO detector.
"""
import cv2
import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import shutil

class CropGenerator:
    def __init__(self, config):
        self.config = config
        # inject the best detector model path here 
        self.model = YOLO(config['yolo_model_path']) 
        
        # Setup paths
        # cropped images will be saved here in output_dir
        self.base_dir = Path(config['output_dir'])
        self.images_dir = self.base_dir / 'images'
        self.train_dir = self.images_dir / 'train'
        self.val_dir = self.images_dir / 'valid'
        
        # Create directories
        for dir_path in [self.train_dir, self.val_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {'train': 0, 'val': 0}
        
    def process_dataset(self, split='train'):
        """Process all images in a split."""
        print(f"\n📂 Processing {split} split...")
        
        # Paths for this split
        img_source_dir = Path(self.config['source_images_dir']) / split/'images'
        label_source_dir = Path(self.config['source_labels_dir']) / split/'labels'
        output_dir = self.train_dir if split == 'train' else self.val_dir
        
        # Get all images
        image_files = list(img_source_dir.glob('*.jpg')) + list(img_source_dir.glob('*.png'))
        print(f"  Found {len(image_files)} images in {img_source_dir}")
        crop_count = 0
        
        for img_path in image_files:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Run YOLO inference
            results = self.model(str(img_path), conf=0.25, verbose=False)
            
            # Process detections
            if results[0].boxes is not None:
                print(f"    DEBUG: Found {len(results[0].boxes)} total detections")
                for i, box in enumerate(results[0].boxes):
                    # CHECK YOUR CLASS ID HERE!
                    # Usually: 0=water, 1=digits, 2=electricity
                    if int(box.cls) == 0:  # DIGIT REGION CLASS
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Crop and save
                        crop = img[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        
                        # Create filename
                        crop_name = f"{img_path.stem}_crop{crop_count:04d}.jpg"
                        crop_path = output_dir / crop_name
                        cv2.imwrite(str(crop_path), crop)
                        
                        crop_count += 1
            
            # Progress update
            if len(image_files) > 0 and (image_files.index(img_path) + 1) % 50 == 0:
                print(f"  Processed {image_files.index(img_path) + 1}/{len(image_files)} images...")
        
        self.stats[split] = crop_count
        print(f"  Generated {crop_count} digit crops for {split}")
        return crop_count
    
    def run(self):
        """Main execution function."""
        print("🚀 Starting crop generation...")
        print(f"Model: {self.config['yolo_model_path']}")
        
        # Process both splits
        train_crops = self.process_dataset('trains')
        val_crops = self.process_dataset('valid')
        
        # Save dataset info
        info_file = self.base_dir / 'dataset_info.txt'
        with open(info_file, 'w') as f:
            f.write(f"Train crops: {train_crops}\n")
            f.write(f"Val crops: {val_crops}\n")
            f.write(f"Total crops: {train_crops + val_crops}\n")
        
        print(f"\n✅ Generation complete!")
        print(f"   Train crops: {train_crops}")
        print(f"   Val crops: {val_crops}")
        print(f"   Total: {train_crops + val_crops}")
        print(f"\n📁 Output saved to: {self.base_dir}")

# CONFIGURATION - UPDATE THESE PATHS!
config = {
    'yolo_model_path': 'models/detectors/YOLOv8/weights/best.pt',  # Your trained YOLO model
    'source_images_dir': 'datasets/detection',  # Base detection dataset directory
    'source_labels_dir': 'datasets/detection',  # Base detection dataset directory
    'output_dir': 'datasets/ocr'        # Output folder for cropped digits
}

if __name__ == "__main__":
    # Verify paths exist
    for key in ['yolo_model_path', 'source_images_dir', 'source_labels_dir']:
        if not Path(config[key]).exists():
            print(f"❌ Error: Path doesn't exist - {config[key]}")
            exit(1)
    
    # Run generator
    generator = CropGenerator(config)
    generator.run()