#!/usr/bin/env python3
"""
SETUP and TEST EasyOCR
Note: EasyOCR doesn't support custom training officially.
This script sets it up for benchmarking.
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_easyocr():
    print("=" * 60)
    print("🔄 SETTING UP EasyOCR FOR BENCHMARKING")
    print("=" * 60)
    
    # Install EasyOCR
    print("\n📦 Installing EasyOCR...")
    subprocess.run([sys.executable, "-m", "pip", "install", "easyocr"], check=False)
    
    # Test installation
    try:
        import easyocr
        print("✅ EasyOCR installed successfully")
        
        # Test with a sample
        print("\n🧪 Testing EasyOCR on sample digits...")
        test_easyocr_sample()
        
    except Exception as e:
        print(f"❌ EasyOCR setup failed: {e}")
        return False
    
    return True

def test_easyocr_sample():
    """Test EasyOCR with a sample digit image"""
    import easyocr
    import cv2
    import numpy as np
    
    # Create a test image with digits
    test_img = np.ones((50, 200, 3), dtype=np.uint8) * 255
    cv2.putText(test_img, "12345", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Initialize reader
    reader = easyocr.Reader(['en'])
    
    # Perform OCR
    results = reader.readtext(test_img)
    
    if results:
        print(f"✅ EasyOCR detected: '{results[0][1]}' (confidence: {results[0][2]:.3f})")
    else:
        print("❌ EasyOCR failed to detect text")
    
    # Show available languages/models
    print("\n🌐 Available EasyOCR languages: ['en', 'ch_sim', 'ja', 'ko', 'th', ...]")
    print("📝 Note: EasyOCR uses pre-trained models only")
    print("   Custom training is not officially supported")

def main():
    setup_easyocr()
    
    print("\n" + "=" * 60)
    print("📋 EasyOCR Limitations for Your Project")
    print("=" * 60)
    print("\n⚠️  IMPORTANT: EasyOCR has limitations:")
    print("   1. NO official custom training support")
    print("   2. Can't train on your digit dataset")
    print("   3. Uses generic pre-trained models")
    print("\n✅ What you CAN do:")
    print("   1. Use it as a baseline for comparison")
    print("   2. Benchmark against your custom models")
    print("   3. Good for general text, not optimized for digits")
    print("\n🎯 Recommendation:")
    print("   Focus training on PaddleOCR and Tesseract")
    print("   Use EasyOCR only as a comparison baseline")

if __name__ == "__main__":
    main()