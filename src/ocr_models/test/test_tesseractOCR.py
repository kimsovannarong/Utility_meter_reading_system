#!/usr/bin/env python3
"""
TESSERACT TESTER - Test Tesseract OCR on digit crops
Note: Requires Tesseract installed separately
Usage: python test_tesseract.py --image path/to/image.jpg
"""
import cv2
import pytesseract
import argparse
from pathlib import Path
import json
import sys

def setup_tesseract():
    """Configure Tesseract path for Windows"""
    if sys.platform == "win32":
        # Common Tesseract installation paths
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"✅ Tesseract found at: {path}")
                return True
        
        print("❌ Tesseract not found!")
        print("\n💡 Install Tesseract from:")
        print("   https://github.com/UB-Mannheim/tesseract/wiki")
        print("\nThen update the path in this script.")
        return False
    return True

def test_tesseract_single(image_path):
    """Test Tesseract on a single image"""
    print("🧪 Testing Tesseract OCR")
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
    
    # Convert to grayscale (Tesseract works better)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding for better contrast
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Tesseract configuration for digits
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    # --oem 3: Default OCR engine
    # --psm 7: Treat image as single text line
    # tessedit_char_whitelist: Only recognize digits
    
    print("🔤 Recognizing text...")
    text = pytesseract.image_to_string(thresh, config=config)
    
    # Alternative: Try with original image too
    text_original = pytesseract.image_to_string(gray, config=config)
    
    # Clean results
    digits_thresh = ''.join(filter(str.isdigit, text))
    digits_original = ''.join(filter(str.isdigit, text_original))
    
    print("\n✅ RESULTS:")
    print(f"\n  With thresholding:")
    print(f"    📝 Raw text: '{text.strip()}'")
    print(f"    🔢 Digits only: '{digits_thresh}'")
    
    print(f"\n  Without thresholding:")
    print(f"    📝 Raw text: '{text_original.strip()}'")
    print(f"    🔢 Digits only: '{digits_original}'")
    
    # Try different page segmentation modes
    print("\n🔧 Trying different configurations...")
    
    configs = {
        'Single line (psm 7)': r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789',
        'Single char (psm 10)': r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789',
        'Sparse text (psm 11)': r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789',
    }
    
    for name, cfg in configs.items():
        result = pytesseract.image_to_string(thresh, config=cfg)
        digits = ''.join(filter(str.isdigit, result))
        if digits:
            print(f"    {name:20} → '{digits}'")
    
    print("\n" + "=" * 50)
    print("💡 Tesseract Tips:")
    print("   • Works best with black text on white background")
    print("   • Try different --psm modes for difficult images")
    print("   • Preprocess images (grayscale + threshold) helps")

def test_tesseract_batch(folder_path):
    """Test Tesseract on multiple images"""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return
    
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    
    if not images:
        print(f"❌ No images found in: {folder_path}")
        return
    
    print(f"📁 Testing {len(images)} images from {folder_path}")
    print("=" * 50)
    
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    results = []
    
    for i, img_path in enumerate(images[:15]):  # Test first 15
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Preprocess
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Run OCR
        text = pytesseract.image_to_string(thresh, config=config)
        digits = ''.join(filter(str.isdigit, text))
        
        results.append({
            'file': img_path.name,
            'text': text.strip(),
            'digits': digits
        })
        
        print(f"{img_path.name:30} → '{digits}'")
    
    # Save results
    if results:
        with open('tesseract_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: tesseract_results.json")

def main():
    # Setup Tesseract first
    if not setup_tesseract():
        return
    
    parser = argparse.ArgumentParser(description='Test Tesseract OCR on digit images')
    parser.add_argument('--image', help='Path to single test image')
    parser.add_argument('--folder', help='Path to folder of test images')
    
    args = parser.parse_args()
    
    if args.image:
        test_tesseract_single(args.image)
    elif args.folder:
        test_tesseract_batch(args.folder)
    else:
        # Default test
        default_path = "datasets/ocr/images/test/"
        if Path(default_path).exists():
            images = list(Path(default_path).glob("*.jpg")) + list(Path(default_path).glob("*.png"))
            if images:
                print("🔍 Testing on first sample image...")
                test_tesseract_single(str(images[0]))
            else:
                print("❌ No images found at default location")
        print("\nUsage:")
        print("  python test_tesseract.py --image path/to/image.jpg")
        print("  python test_tesseract.py --folder datasets/ocr/images/test/")

if __name__ == "__main__":
    main()