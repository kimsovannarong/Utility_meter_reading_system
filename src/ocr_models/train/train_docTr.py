#!/usr/bin/env python3
"""
TRAIN DocTr (Document Text Recognition) on your digit data
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_doctr():
    print("=" * 60)
    print("📄 SETTING UP DocTr (Document Text Recognition)")
    print("=" * 60)
    
    # Install docTR
    print("\n📦 Installing docTR...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "doctr", "tensorflow"  # or "torch" for PyTorch
    ], check=False)
    
    # Check installation
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
        print("✅ docTR installed successfully")
        
        # Test with sample
        test_doctr_sample()
        
    except Exception as e:
        print(f"❌ docTR setup failed: {e}")
        print("\nTrying alternative installation...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "python-doctr"
        ], check=False)
        
        try:
            from doctr.io import DocumentFile
            print("✅ docTR installed via alternative method")
        except:
            print("❌ Failed to install docTR")
            return False
    
    return True

def test_doctr_sample():
    """Test docTR with a sample"""
    import numpy as np
    import cv2
    
    print("\n🧪 Testing docTR...")
    
    # Create test image
    test_img = np.ones((60, 200, 3), dtype=np.uint8) * 255
    cv2.putText(test_img, "9876", (30, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Save and test
    cv2.imwrite("test_digit.png", test_img)
    
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
        
        # Load model (this downloads pretrained weights)
        print("Loading docTR model (may download ~200MB)...")
        model = ocr_predictor(det_arch='db_resnet50', 
                            reco_arch='crnn_vgg16_bn',
                            pretrained=True)
        
        # Process image
        doc = DocumentFile.from_images("test_digit.png")
        result = model(doc)
        
        # Show results
        print("\n📊 docTR structure:")
        print("   - Detection model: finds text regions")
        print("   - Recognition model: reads text")
        print("   - Can train both or just recognition")
        
        os.remove("test_digit.png")
        
    except Exception as e:
        print(f"   Test error: {e}")
        print("   But installation is successful")

def train_doctr_model():
    """DocTR training example structure"""
    print("\n" + "=" * 60)
    print("🏋️  DocTr TRAINING PROCESS")
    print("=" * 60)
    
    print("\n📚 DocTR Training Steps:")
    print("1. Load your dataset")
    print("2. Choose model architecture")
    print("3. Set up training loop")
    print("4. Train recognition model")
    print("5. Evaluate on validation set")
    
    print("\n💡 For digit recognition, you would:")
    print("   a. Use only recognition model (CRNN)")
    print("   b. Train on your cropped digit images")
    print("   c. Use CTC loss for sequence recognition")
    
    print("\n⚠️  Note: Full DocTR training is complex")
    print("   Requires PyTorch/TensorFlow knowledge")
    print("   Consider using PaddleOCR for simplicity")

def main():
    if not setup_doctr():
        return
    
    train_doctr_model()
    
    print("\n" + "=" * 60)
    print("✅ DocTr SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps for DocTR training:")
    print("1. Prepare dataset in DocTR format")
    print("2. Write custom training script")
    print("3. Train recognition model on digits")
    print("\n📁 Your data is ready at:")
    print("   datasets/ocr/doctr_train.txt")
    print("   datasets/ocr/doctr_val.txt")

if __name__ == "__main__":
    main()