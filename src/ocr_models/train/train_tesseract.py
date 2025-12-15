#!/usr/bin/env python3
"""
TRAIN TesseractOCR on your digit data
Usage: python train_tesseract.py
Note: Requires Tesseract 5+ and tesstrain installed
"""
import os
import sys
import subprocess
from pathlib import Path
import shutil

def train_tesseract():
    print("=" * 60)
    print("🔤 TRAINING TESSERACT OCR MODEL")
    print("=" * 60)
    
    # Configuration
    data_dir = Path("datasets/ocr/tesseract_data")
    output_dir = Path("trained_models/tesseract_digits")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if training data exists
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if not train_dir.exists() or len(list(train_dir.glob("*.png"))) == 0:
        print("❌ Tesseract training data not found!")
        print("   Run Step 3 (03_prepare_ocr_data.py) first!")
        return False
    
    print(f"📊 Training data: {len(list(train_dir.glob('*.png')))} images")
    print(f"📊 Validation data: {len(list(val_dir.glob('*.png')))} images")
    
    # Method 1: Using tesstrain (recommended for Linux/macOS)
    print("\n🏋️  Method 1: Training with tesstrain...")
    
    # Create training file list
    with open("tesseract_train_files.txt", "w") as f:
        for img_file in train_dir.glob("*.png"):
            f.write(f"{img_file}\n")
    
    # Tesseract training command structure
    # Note: Actual command depends on your system installation
    print("\n⚠️  Tesseract training is system-dependent.")
    print("   For Linux/macOS with tesstrain installed, run:")
    print("\n   make training \\")
    print("     MODEL_NAME=digits \\")
    print("     START_MODEL=eng \\")
    print("     TESSDATA=/usr/share/tesseract-ocr/5/tessdata \\")
    print("     DATA_DIR=datasets/ocr/tesseract_data/train \\")
    print("     MAX_ITERATIONS=1000")
    
    # Method 2: Fine-tuning with existing model (simpler)
    print("\n" + "-" * 40)
    print("🏋️  Method 2: Fine-tuning (Simpler Approach)")
    print("-" * 40)
    
    # Create .box files for training (required by Tesseract)
    print("\nCreating .box files for training...")
    
    for img_file in train_dir.glob("*.png"):
        gt_file = img_file.with_suffix(".gt.txt")
        box_file = img_file.with_suffix(".box")
        
        if gt_file.exists():
            with open(gt_file, "r") as f:
                text = f.read().strip()
            
            # Create .box file format
            img = Image.open(img_file)
            width, height = img.size
            
            with open(box_file, "w") as f:
                for i, char in enumerate(text):
                    f.write(f"{char} {i*20} {0} {(i+1)*20} {height} 0\n")
    
    print(f"✅ Created {len(list(train_dir.glob('*.box')))} box files")
    
    # Tesseract training steps
    print("\n📚 Tesseract Training Steps:")
    print("1. Combine .tif files: tiffcp *.tif digits.tif")
    print("2. Create unicharset: unicharset_extractor *.box")
    print("3. Set font properties")
    print("4. Run shape clustering")
    print("5. Run mftraining/cntraining")
    print("6. Combine data")
    
    print("\n💡 Recommendation: Use pre-trained Tesseract for now,")
    print("   and focus on PaddleOCR for custom training.")
    
    return True

if __name__ == "__main__":
    try:
        from PIL import Image
        train_tesseract()
    except ImportError:
        print("Installing Pillow for image processing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image
        train_tesseract()