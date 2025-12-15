#!/usr/bin/env python3
"""
BENCHMARK ALL OCR MODELS
Tests PaddleOCR, Tesseract, EasyOCR, and DocTr on your test set
"""
import cv2
import pandas as pd
import time
import numpy as np
from pathlib import Path
import json

class OCRBenchmark:
    def __init__(self):
        self.test_dir = Path("datasets/ocr/images/test")
        self.test_labels = self.load_test_labels()
        
        # Results storage
        self.results = []
        
    def load_test_labels(self):
        """Load ground truth labels from test_list.txt"""
        labels = {}
        test_file = Path("datasets/ocr/test_list.txt")
        
        if not test_file.exists():
            print("❌ Test labels not found. Run data preparation first!")
            return {}
        
        with open(test_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_path = parts[0]
                    label = parts[1]
                    img_name = img_path.split('/')[-1]
                    labels[img_name] = label
        
        print(f"📋 Loaded {len(labels)} test images")
        return labels
    
    def run_all_benchmarks(self, max_images=50):
        """Run benchmarks for all OCR engines"""
        print("=" * 70)
        print("📊 COMPREHENSIVE OCR MODELS BENCHMARK")
        print("=" * 70)
        
        # Limit number of test images for speed
        test_items = list(self.test_labels.items())[:max_images]
        
        # 1. PaddleOCR (Custom trained)
        print("\n1️⃣  PADDLEOCR (Your Custom Model)")
        paddle_results = self.benchmark_paddleocr(test_items)
        self.results.append(("PaddleOCR (Custom)", *paddle_results))
        
        # 2. EasyOCR (Pretrained)
        print("\n2️⃣  EASYOCR (Pretrained)")
        easy_results = self.benchmark_easyocr(test_items)
        self.results.append(("EasyOCR (Pretrained)", *easy_results))
        
        # 3. Tesseract OCR
        print("\n3️⃣  TESSERACT OCR (Default)")
        tesseract_results = self.benchmark_tesseract(test_items)
        self.results.append(("Tesseract OCR", *tesseract_results))
        
        # 4. DocTr (Pretrained)
        print("\n4️⃣  DOCTr (Pretrained)")
        doctr_results = self.benchmark_doctr(test_items)
        self.results.append(("DocTr (Pretrained)", *doctr_results))
        
        # Display results
        self.display_results()
        
        # Save detailed results
        self.save_detailed_results(test_items)
    
    def benchmark_paddleocr(self, test_items):
        """Benchmark PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            
            print("   Loading PaddleOCR model...")
            # Try custom model first, fall back to default
            custom_model_path = Path("PaddleOCR/inference/rec_digit")
            
            if custom_model_path.exists():
                ocr = PaddleOCR(
                    rec_model_dir=str(custom_model_path),
                    use_angle_cls=False,
                    lang='en',
                    show_log=False
                )
                print("   ✓ Using your custom trained model")
            else:
                ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                print("   ⚠️  Using default model (custom not found)")
            
            return self._run_ocr_test(ocr, test_items, engine='paddle')
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0, 0, 0, 0, "0/0"
    
    def benchmark_easyocr(self, test_items):
        """Benchmark EasyOCR"""
        try:
            import easyocr
            
            print("   Loading EasyOCR model...")
            reader = easyocr.Reader(['en'])
            
            return self._run_ocr_test(reader, test_items, engine='easy')
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0, 0, 0, 0, "0/0"
    
    def benchmark_tesseract(self, test_items):
        """Benchmark Tesseract"""
        try:
            import pytesseract
            
            print("   Testing Tesseract OCR...")
            return self._run_ocr_test(pytesseract, test_items, engine='tesseract')
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0, 0, 0, 0, "0/0"
    
    def benchmark_doctr(self, test_items):
        """Benchmark DocTr"""
        try:
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
            
            print("   Loading DocTr model (may be slow)...")
            model = ocr_predictor(pretrained=True)
            
            return self._run_ocr_test(model, test_items, engine='doctr')
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 0, 0, 0, 0, "0/0"
    
    def _run_ocr_test(self, ocr_engine, test_items, engine='paddle'):
        """Run OCR test and calculate metrics"""
        correct = 0
        total_time = 0
        cer_total = 0
        wer_total = 0
        
        for i, (img_name, true_label) in enumerate(test_items):
            img_path = self.test_dir / img_name
            
            if not img_path.exists():
                continue
            
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Run OCR with timing
            start_time = time.time()
            
            if engine == 'paddle':
                result = ocr_engine.ocr(img, cls=False)
                pred = ""
                if result and result[0]:
                    text = result[0][0][1][0]
                    pred = ''.join(filter(str.isdigit, text))
            
            elif engine == 'easy':
                result = ocr_engine.readtext(img)
                pred = ""
                if result:
                    text = result[0][1]
                    pred = ''.join(filter(str.isdigit, text))
            
            elif engine == 'tesseract':
                # Configure for digits
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
                text = ocr_engine.image_to_string(img, config=custom_config)
                pred = ''.join(filter(str.isdigit, text))
            
            elif engine == 'doctr':
                # Save temp file for doctr
                temp_path = f"temp_{i}.png"
                cv2.imwrite(temp_path, img)
                doc = DocumentFile.from_images(temp_path)
                result = ocr_engine(doc)
                
                # Extract text from result
                pred = ""
                if hasattr(result, 'pages') and result.pages:
                    for block in result.pages[0].blocks:
                        for line in block.lines:
                            pred += line.value
                pred = ''.join(filter(str.isdigit, pred))
                
                # Clean up
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Calculate CER and WER
            cer = self.calculate_cer(true_label, pred)
            wer = self.calculate_wer(true_label, pred)
            
            cer_total += cer
            wer_total += wer
            
            # Check if completely correct
            if pred == true_label:
                correct += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"     Processed {i+1}/{len(test_items)} images...")
        
        # Calculate averages
        n = len(test_items)
        accuracy = (correct / n * 100) if n > 0 else 0
        avg_time = total_time / n if n > 0 else 0
        avg_cer = cer_total / n if n > 0 else 100
        avg_wer = wer_total / n if n > 0 else 100
        
        print(f"     ✓ Accuracy: {accuracy:.1f}%, Time: {avg_time:.3f}s/img")
        
        return accuracy, avg_time, avg_cer, avg_wer, f"{correct}/{n}"
    
    def calculate_cer(self, truth, prediction):
        """Calculate Character Error Rate"""
        # Simple implementation
        if not truth:
            return 100 if prediction else 0
        
        # Count errors (simplified)
        errors = 0
        min_len = min(len(truth), len(prediction))
        max_len = max(len(truth), len(prediction))
        
        for i in range(min_len):
            if truth[i] != prediction[i]:
                errors += 1
        
        errors += (max_len - min_len)  # Missing or extra characters
        
        return (errors / len(truth)) * 100 if truth else 100
    
    def calculate_wer(self, truth, prediction):
        """Calculate Word Error Rate (for digits, each digit is a 'word')"""
        # For digits, we can treat each digit as a word
        truth_words = list(truth)
        pred_words = list(prediction)
        
        if not truth_words:
            return 100 if pred_words else 0
        
        # Simplified WER calculation
        errors = 0
        min_len = min(len(truth_words), len(pred_words))
        max_len = max(len(truth_words), len(pred_words))
        
        for i in range(min_len):
            if truth_words[i] != pred_words[i]:
                errors += 1
        
        errors += (max_len - min_len)
        
        return (errors / len(truth_words)) * 100 if truth_words else 100
    
    def display_results(self):
        """Display benchmark results in a nice table"""
        print("\n" + "=" * 70)
        print("📈 FINAL BENCHMARK RESULTS")
        print("=" * 70)
        
        # Create DataFrame
        df = pd.DataFrame(
            self.results,
            columns=["Model", "Accuracy %", "Time (s/img)", "CER %", "WER %", "Correct/Total"]
        )
        
        # Format for display
        pd.set_option('display.width', 100)
        pd.set_option('display.max_columns', 6)
        pd.set_option('display.float_format', '{:.2f}'.format)
        
        print("\n" + df.to_string(index=False))
        
        # Find best model
        if not df.empty:
            best_acc = df.loc[df['Accuracy %'].idxmax()]
            print(f"\n🏆 BEST MODEL: {best_acc['Model']} ({best_acc['Accuracy %']:.1f}% accuracy)")
            
            fastest = df.loc[df['Time (s/img)'].idxmin()]
            print(f"⚡ FASTEST: {fastest['Model']} ({fastest['Time (s/img)']:.3f} seconds/image)")
    
    def save_detailed_results(self, test_items):
        """Save detailed results to files"""
        # Save summary CSV
        df = pd.DataFrame(
            self.results,
            columns=["Model", "Accuracy %", "Time (s/img)", "CER %", "WER %", "Correct/Total"]
        )
        df.to_csv("benchmark_results.csv", index=False)
        
        # Save JSON with more details
        results_dict = {
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_set_size": len(test_items),
            "models": []
        }
        
        for model_name, acc, avg_time, cer, wer, correct_str in self.results:
            results_dict["models"].append({
                "name": model_name,
                "accuracy": acc,
                "avg_inference_time": avg_time,
                "character_error_rate": cer,
                "word_error_rate": wer,
                "correct": correct_str
            })
        
        with open("benchmark_detailed.json", "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n💾 Results saved to:")
        print(f"   - benchmark_results.csv (summary table)")
        print(f"   - benchmark_detailed.json (detailed results)")

def main():
    print("🚀 OCR MODELS COMPARISON TOOL")
    print("This will benchmark PaddleOCR, EasyOCR, Tesseract, and DocTr")
    print("-" * 60)
    
    # Check if test data exists
    test_dir = Path("datasets/ocr/images/test")
    if not test_dir.exists() or len(list(test_dir.glob("*"))) == 0:
        print("❌ Test data not found!")
        print("\nPlease run these steps first:")
        print("1. python 01_generate_crops.py")
        print("2. python 02_quick_labeler.py")
        print("3. python 03_prepare_ocr_data.py")
        return
    
    # Ask how many images to test
    print(f"Found {len(list(test_dir.glob('*.jpg')))} test images")
    try:
        max_images = int(input("\nHow many images to test? (Press Enter for 50): ") or "50")
    except:
        max_images = 50
    
    # Run benchmark
    benchmark = OCRBenchmark()
    benchmark.run_all_benchmarks(max_images=max_images)
    
    print("\n" + "=" * 70)
    print("🎯 RECOMMENDATIONS")
    print("=" * 70)
    print("\nBased on your meter reading project:")
    print("1. Use PaddleOCR if you need highest accuracy")
    print("2. Use Tesseract if you want open-source and good speed")
    print("3. Consider EasyOCR for general text (not digit-optimized)")
    print("4. DocTr is powerful but complex for digit-only tasks")
    print("\nNext: Integrate the best model into your pipeline!")

if __name__ == "__main__":
    main()