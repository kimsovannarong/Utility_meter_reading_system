"""
"""
import cv2
import os
from pathlib import Path

def label_crops():
    """Label all crop images interactively."""
    # Configuration
    crops_dir = Path("./ocr_dataset/images/train/")  # Label training crops
    output_file = Path("./ocr_dataset/train_list.txt")
    
    # Get all crop images
    images = sorted([f for f in crops_dir.glob("*.jpg")] + 
                    [f for f in crops_dir.glob("*.png")])
    
    if not images:
        print("❌ No images found! Run 01_generate_crops.py first.")
        return
    
    print(f"📝 Labeling {len(images)} images")
    print("Controls:")
    print("  • Type digits and press ENTER to label")
    print("  • Press 's' to skip an image")
    print("  • Press 'q' to quit and save")
    print("-" * 40)
    
    labels = []
    skipped = 0
    
    for i, img_path in enumerate(images):
        # Load and display image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Could not load {img_path.name}, skipping")
            continue
        
        # Resize for display if too large
        h, w = img.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        # Show image with info
        display = img.copy()
        cv2.putText(display, f"Image {i+1}/{len(images)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Enter digits:", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Digit Labeler", display)
        
        # Get user input via OpenCV (simple)
        cv2.waitKey(1)  # Small delay to ensure window is ready
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            print("\n⏹️  Saving and exiting...")
            break
        elif key == ord('s'):
            print(f"  Skipped: {img_path.name}")
            skipped += 1
            continue
        
        # Get digits via console input (more reliable)
        cv2.destroyWindow("Digit Labeler")
        digits = input(f"\n[{i+1}/{len(images)}] Enter digits for {img_path.name}: ").strip()
        
        # Validate input
        if not digits.isdigit():
            print(f"  ❌ '{digits}' is not valid digits. Skipping.")
            skipped += 1
            cv2.destroyAllWindows()
            continue
        
        # Save label
        rel_path = f"images/train/{img_path.name}"
        labels.append(f"{rel_path}\t{digits}")
        print(f"  ✅ Labeled: {digits}")
        
        # Recreate window for next image
        cv2.destroyAllWindows()
    
    # Save all labels
    if labels:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(labels))
        
        print(f"\n✅ Saved {len(labels)} labels to {output_file}")
        if skipped > 0:
            print(f"   Skipped {skipped} images")
    else:
        print("❌ No labels were saved!")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    label_crops()