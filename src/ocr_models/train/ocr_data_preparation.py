"""
STEP 3: Prepare labeled data for OCR training
This creates the files needed for PaddleOCR, Tesseract, and DocTR
"""
import os
import random
import shutil
from pathlib import Path
from PIL import Image

def prepare_all_ocr_data():
    print("=" * 60)
    print("📦 PREPARE OCR TRAINING DATA")
    print("=" * 60)
    
    # Configuration
    base_dir = Path("datasets/ocr")
    all_crops_dir = base_dir / "all_crops"  # From Step 1
    labels_file = base_dir / "manual_labels.txt"  # From Step 2
    
    # Check if files exist
    if not all_crops_dir.exists():
        print(f"❌ Crops folder not found: {all_crops_dir}")
        print("   Run Step 1 (01_generate_crops.py) first!")
        return
    
    if not labels_file.exists():
        print(f"❌ Labels file not found: {labels_file}")
        print("   Run Step 2 (02_quick_labeler.py) first!")
        return
    
    # Create directory structure
    dirs_to_create = [
        base_dir / "images" / "train",
        base_dir / "images" / "val",
        base_dir / "images" / "test",
        base_dir / "tesseract_data" / "train",
        base_dir / "tesseract_data" / "val"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Step 3.1: Load manual labels
    print("\n📖 Loading manual labels...")
    image_label_map = {}
    
    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_name = parts[0]
                label = parts[1]
                image_label_map[img_name] = label
    
    print(f"  Loaded {len(image_label_map)} labeled images")
    
    # Step 3.2: Split data (70% train, 15% val, 15% test)
    print("\n📊 Splitting data...")
    all_items = list(image_label_map.items())
    random.shuffle(all_items)
    
    total = len(all_items)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)
    
    train_items = all_items[:train_end]
    val_items = all_items[train_end:val_end]
    test_items = all_items[val_end:]
    
    print(f"  Training set: {len(train_items)} images")
    print(f"  Validation set: {len(val_items)} images")
    print(f"  Test set: {len(test_items)} images")
    
    # Step 3.3: Create PaddleOCR format files
    print("\n📝 Creating PaddleOCR format...")
    create_paddleocr_files(train_items, val_items, test_items, base_dir)
    
    # Step 3.4: Create Tesseract format files
    print("\n🔤 Creating Tesseract format...")
    create_tesseract_files(train_items, val_items, base_dir)
    
    # Step 3.5: Create DocTR format files
    print("\n📄 Creating DocTR format...")
    create_doctr_files(train_items, val_items, test_items, base_dir)
    
    # Step 3.6: Copy images to train/val/test folders
    print("\n🖼️  Organizing image files...")
    organize_image_files(train_items, val_items, test_items, all_crops_dir, base_dir)
    
    # Step 3.7: Create digits dictionary
    print("\n🔢 Creating digits dictionary...")
    with open(base_dir / "digits_dict.txt", 'w') as f:
        for i in range(10):
            f.write(f"{i}\n")
    
    print("\n" + "=" * 60)
    print("✅ STEP 3 COMPLETE: Data ready for all OCR engines!")
    print("=" * 60)
    print("\n📁 Generated files:")
    print(f"  • {base_dir}/train_list.txt (PaddleOCR training)")
    print(f"  • {base_dir}/val_list.txt (PaddleOCR validation)")
    print(f"  • {base_dir}/test_list.txt (PaddleOCR testing)")
    print(f"  • {base_dir}/doctr_train.txt (DocTR training)")
    print(f"  • {base_dir}/doctr_val.txt (DocTR validation)")
    print(f"  • {base_dir}/tesseract_data/ (Tesseract training data)")
    print(f"  • {base_dir}/digits_dict.txt (Character dictionary)")
    print(f"  • {base_dir}/images/train/, /val/, /test/ (Organized images)")

def create_paddleocr_files(train_items, val_items, test_items, base_dir):
    """Create PaddleOCR format label files"""
    # Training file
    with open(base_dir / "train_list.txt", 'w') as f:
        for img_name, label in train_items:
            f.write(f"images/train/{img_name}\t{label}\n")
    
    # Validation file
    with open(base_dir / "val_list.txt", 'w') as f:
        for img_name, label in val_items:
            f.write(f"images/val/{img_name}\t{label}\n")
    
    # Test file
    with open(base_dir / "test_list.txt", 'w') as f:
        for img_name, label in test_items:
            f.write(f"images/test/{img_name}\t{label}\n")

def create_tesseract_files(train_items, val_items, base_dir):
    """Create Tesseract format (.tif + .gt.txt)"""
    tesseract_dir = base_dir / "tesseract_data"
    
    # Create training data
    for i, (img_name, label) in enumerate(train_items):
        # Convert/copy image to .tif if needed
        src_path = base_dir / "all_crops" / img_name
        
        if src_path.exists():
            # For simplicity, we'll use .png for Tesseract
            dest_name = f"train_{i:04d}"
            dest_img = tesseract_dir / "train" / f"{dest_name}.png"
            
            # Copy/convert image
            img = Image.open(src_path)
            img.save(dest_img)
            
            # Create ground truth file
            gt_file = tesseract_dir / "train" / f"{dest_name}.gt.txt"
            with open(gt_file, 'w') as f:
                f.write(label)
    
    # Create validation data
    for i, (img_name, label) in enumerate(val_items):
        src_path = base_dir / "all_crops" / img_name
        
        if src_path.exists():
            dest_name = f"val_{i:04d}"
            dest_img = tesseract_dir / "val" / f"{dest_name}.png"
            
            img = Image.open(src_path)
            img.save(dest_img)
            
            gt_file = tesseract_dir / "val" / f"{dest_name}.gt.txt"
            with open(gt_file, 'w') as f:
                f.write(label)

def create_doctr_files(train_items, val_items, test_items, base_dir):
    """Create DocTR format files"""
    # Training file (list of image paths and labels)
    with open(base_dir / "doctr_train.txt", 'w') as f:
        for img_name, label in train_items:
            f.write(f"images/train/{img_name} {label}\n")
    
    # Validation file
    with open(base_dir / "doctr_val.txt", 'w') as f:
        for img_name, label in val_items:
            f.write(f"images/val/{img_name} {label}\n")
    
    # Test file
    with open(base_dir / "doctr_test.txt", 'w') as f:
        for img_name, label in test_items:
            f.write(f"images/test/{img_name} {label}\n")

def organize_image_files(train_items, val_items, test_items, all_crops_dir, base_dir):
    """Copy images to organized folders"""
    # Copy training images
    for img_name, _ in train_items:
        src = all_crops_dir / img_name
        dst = base_dir / "images" / "train" / img_name
        if src.exists():
            shutil.copy2(src, dst)
    
    # Copy validation images
    for img_name, _ in val_items:
        src = all_crops_dir / img_name
        dst = base_dir / "images" / "val" / img_name
        if src.exists():
            shutil.copy2(src, dst)
    
    # Copy test images
    for img_name, _ in test_items:
        src = all_crops_dir / img_name
        dst = base_dir / "images" / "test" / img_name
        if src.exists():
            shutil.copy2(src, dst)

if __name__ == "__main__":
    prepare_all_ocr_data()