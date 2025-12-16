#!/usr/bin/env python3
"""
EASYOCR TESTER - Test pre-trained EasyOCR on digit crops
Usage: python test_easyocr.py --image path/to/image.jpg
"""
import cv2
import argparse
from pathlib import Path
import json
import time

def test_easyocr_single(image_path):
    """Test EasyOCR on a single image"""
    print("🧪 Testing EasyOCR")
    print("=" * 50)
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return
    
    print(f"📷 Image: {Path(image_path).name}")
    print(f"   Size: {img.shape[1]}x{img.shape[0]}")
    
    # Initialize EasyOCR (this downloads models on first run)
    print("🚀 Initializing EasyOCR...")
    import easyocr
    
    # Create reader - use CPU if no GPU
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Run OCR
    print("🔤 Recognizing text...")
    start_time = time.time()
    result = reader.readtext(img, detail=1)
    elapsed = time.time() - start_time
    
    # Display results
    if not result:
        print("⚠️  No text detected")
        return
    
    print(f"\n⏱️  Processing time: {elapsed:.3f} seconds")
    print("\n✅ RESULTS:")
    
    for i, detection in enumerate(result):
        bbox = detection[0]      # Bounding box
        text = detection[1]      # Text
        confidence = detection[2]  # Confidence
        
        # Extract only digits
        digits = ''.join(filter(str.isdigit, text))
        
        print(f"\n  Detection {i+1}:")
        print(f"    📝 Raw text: '{text}'")
        print(f"    🔢 Digits only: '{digits}'")
        print(f"    📊 Confidence: {confidence:.3f}")
        print(f"    📐 Bounding box: {bbox}")
    
    print("\n" + "=" * 50)
    print("💡 EasyOCR Characteristics:")
    print("   • Good with noisy/blurry images")
    print("   • Slower than Tesseract but more accurate")
    print("   • First run downloads models (~100MB)")

def test_easyocr_batch(folder_path, max_images=10):
    """Test EasyOCR on multiple images"""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return
    
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    
    if not images:
        print(f"❌ No images found in: {folder_path}")
        return
    
    print(f"📁 Testing {min(len(images), max_images)} images from {folder_path}")
    print("=" * 50)
    
    # Initialize once
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    
    results = []
    for i, img_path in enumerate(images[:max_images]):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        print(f"[{i+1}/{min(len(images), max_images)}] {img_path.name}", end=" ", flush=True)
        
        result = reader.readtext(img, detail=1)
        
        if result:
            text = result[0][1]
            confidence = result[0][2]
            digits = ''.join(filter(str.isdigit, text))
            
            results.append({
                'file': img_path.name,
                'text': text,
                'digits': digits,
                'confidence': confidence
            })
            
            print(f"→ '{digits}' ({confidence:.3f})")
        else:
            print("→ No text")
    
    # Save results
    if results:
        with open('easyocr_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: easyocr_results.json")

def main():
    parser = argparse.ArgumentParser(description='Test EasyOCR on digit images')
    parser.add_argument('--image', help='Path to single test image')
    parser.add_argument('--folder', help='Path to folder of test images')
    parser.add_argument('--batch', type=int, default=10, help='Max images for batch test')
    
    args = parser.parse_args()
    
    if args.image:
        test_easyocr_single(args.image)
    elif args.folder:
        test_easyocr_batch(args.folder, args.batch)
    else:
        # Try default location
        default_path = "datasets/ocr/images/test/"
        if Path(default_path).exists():
            images = list(Path(default_path).glob("*.jpg")) + list(Path(default_path).glob("*.png"))
            if images:
                print("🔍 Testing on first sample image...")
                test_easyocr_single(str(images[0]))
            else:
                print("❌ No images found at default location")
        print("\nUsage:")
        print("  python test_easyocr.py --image path/to/image.jpg")
        print("  python test_easyocr.py --folder datasets/ocr/images/test/ --batch 5")

if __name__ == "__main__":
    main()