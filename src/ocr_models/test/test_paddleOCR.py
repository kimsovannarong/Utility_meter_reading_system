#!/usr/bin/env python3
"""
PADDLEOCR TESTER - Test pre-trained PaddleOCR on digit crops
Usage: python test_paddleocr.py --image path/to/image.jpg
       python test_paddleocr.py --folder path/to/folder
"""
import cv2
import argparse
from pathlib import Path
import json

def test_paddleocr_single(image_path):
    """Test PaddleOCR on a single image"""
    print("🧪 Testing PaddleOCR")
    print("=" * 50)
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return
    
    print(f"📷 Image: {Path(image_path).name}")
    print(f"   Size: {img.shape[1]}x{img.shape[0]}")
    
    # Initialize PaddleOCR (this downloads model on first run)
    print("🚀 Initializing PaddleOCR...")
    from paddleocr import PaddleOCR
    
    # Create OCR engine - optimized for digits
    ocr = PaddleOCR(
        use_angle_cls=True,  # Enable angle classification
        lang='en',           # English language
        det=False,           # NO detection (we already have YOLO crops)
        rec=True,            # YES recognition
        show_log=False       # Disable verbose logging
    )
    
    # Run OCR
    print("🔤 Recognizing text...")
    result = ocr.ocr(img, cls=True)
    
    # Display results
    if result is None or len(result) == 0:
        print("⚠️  No text detected")
        return
    
    print("\n✅ RESULTS:")
    for i, line in enumerate(result[0]):
        text = line[1][0]
        confidence = line[1][1]
        
        # Extract only digits
        digits = ''.join(filter(str.isdigit, text))
        
        print(f"\n  Detection {i+1}:")
        print(f"    📝 Raw text: '{text}'")
        print(f"    🔢 Digits only: '{digits}'")
        print(f"    📊 Confidence: {confidence:.3f}")
        
        # Show bounding box if available
        if line[0]:
            bbox = line[0]
            print(f"    📐 Bounding box: {bbox}")
    
    print("\n" + "=" * 50)
    print("💡 Tips:")
    print("   • Confidence > 0.9 is excellent")
    print("   • If digits are wrong, try better image quality")
    print("   • Works best with clear, high-contrast images")

def test_paddleocr_folder(folder_path):
    """Test PaddleOCR on all images in a folder"""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return
    
    # Get all images
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    
    if not images:
        print(f"❌ No images found in: {folder_path}")
        return
    
    print(f"📁 Found {len(images)} images in {folder_path}")
    print("=" * 50)
    
    # Initialize PaddleOCR once
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', det=False, rec=True, show_log=False)
    
    results = []
    for img_path in images[:10]:  # Test first 10 images
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        result = ocr.ocr(img, cls=True)
        
        if result and result[0]:
            text = result[0][0][1][0]
            confidence = result[0][0][1][1]
            digits = ''.join(filter(str.isdigit, text))
            
            results.append({
                'file': img_path.name,
                'text': text,
                'digits': digits,
                'confidence': confidence
            })
            
            print(f"{img_path.name:30} → '{digits}' ({confidence:.3f})")
    
    # Summary
    if results:
        print("-" * 50)
        avg_conf = sum(r['confidence'] for r in results) / len(results)
        success_rate = sum(1 for r in results if r['digits']) / len(results) * 100
        
        print(f"📊 Summary:")
        print(f"   Average confidence: {avg_conf:.3f}")
        print(f"   Success rate: {success_rate:.1f}%")
        
        # Save to JSON
        with open('paddleocr_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: paddleocr_results.json")

def main():
    parser = argparse.ArgumentParser(description='Test PaddleOCR on digit images')
    parser.add_argument('--image', help='Path to single test image')
    parser.add_argument('--folder', help='Path to folder of test images')
    
    args = parser.parse_args()
    
    if args.image:
        test_paddleocr_single(args.image)
    elif args.folder:
        test_paddleocr_folder(args.folder)
    else:
        # Default test on sample
        sample_path = "datasets/ocr/images/test/"
        if Path(sample_path).exists():
            images = list(Path(sample_path).glob("*.jpg")) + list(Path(sample_path).glob("*.png"))
            if images:
                print("🔍 Testing on first sample image...")
                test_paddleocr_single(str(images[0]))
            else:
                print("❌ No sample images found")
                print("\nUsage:")
                print("  python test_paddleocr.py --image path/to/image.jpg")
                print("  python test_paddleocr.py --folder path/to/folder/")
        else:
            print("❌ No test images found")
            print("\nPlease specify an image or folder:")

if __name__ == "__main__":
    main()