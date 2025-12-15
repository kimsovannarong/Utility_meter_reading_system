"""
STEP 4: Train PaddleOCR model on your digit data
"""
import os
import sys
import subprocess
from pathlib import Path

def train_paddleocr():
    print("=" * 60)
    print("🚀 TRAINING PADDLEOCR MODEL")
    print("=" * 60)
    
    # Step 4.1: Setup PaddleOCR
    print("\n🔧 Setting up PaddleOCR...")
    
    paddleocr_dir = Path("PaddleOCR")
    if not paddleocr_dir.exists():
        print("Cloning PaddleOCR repository...")
        subprocess.run(["git", "clone", "https://github.com/PaddlePaddle/PaddleOCR.git"], 
                      check=True)
    
    # Step 4.2: Install requirements
    print("\n📦 Installing requirements...")
    req_file = paddleocr_dir / "requirements.txt"
    if req_file.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], 
                      check=False)
    
    # Step 4.3: Download pretrained model
    print("\n⬇️  Downloading pretrained model...")
    pretrained_dir = paddleocr_dir / "pretrained_models"
    pretrained_dir.mkdir(exist_ok=True)
    
    import urllib.request
    import tarfile
    
    model_url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar"
    tar_path = pretrained_dir / "en_PP-OCRv3_rec_train.tar"
    
    if not tar_path.exists():
        print("Downloading...")
        urllib.request.urlretrieve(model_url, tar_path)
        
        print("Extracting...")
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(pretrained_dir)
        
        os.remove(tar_path)
    
    # Step 4.4: Create config file
    print("\n⚙️  Creating training configuration...")
    create_paddleocr_config()
    
    # Step 4.5: Start training
    print("\n🏋️  Starting training...")
    print("This will take 30-60 minutes. Check logs in output/rec_digit/")
    
    original_dir = os.getcwd()
    os.chdir(paddleocr_dir)
    
    try:
        # Training command
        cmd = [
            sys.executable, "tools/train.py",
            "-c", "../configs/rec_digit.yml",
            "-o", "Global.epoch_num=50",
            "Global.save_model_dir=./output/rec_digit",
            f"Global.pretrained_model=./pretrained_models/en_PP-OCRv3_rec_train/best_accuracy"
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training error: {e}")
    finally:
        os.chdir(original_dir)
    
    print("\n" + "=" * 60)
    print("✅ PADDLEOCR TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test your model with: python test_paddleocr.py")
    print("2. Export for inference: python export_paddleocr.py")

def create_paddleocr_config():
    """Create PaddleOCR configuration file"""
    config_content = """
Global:
  use_visualdl: false
  save_model_dir: ./output/rec_digit
  save_epoch_step: 10
  eval_batch_step: [0, 100]
  cal_metric_during_train: true
  pretrained_model: ./pretrained_models/en_PP-OCRv3_rec_train/best_accuracy.pdparams
  checkpoints: null
  character_dict_path: ../datasets/ocr/digits_dict.txt
  character_type: EN
  max_text_length: 10
  infer_mode: false
  use_space_char: false

Architecture:
  model_type: rec
  algorithm: CRNN
  Transform: null
  Backbone:
    name: MobileNetV3
    scale: 0.5
    model_name: small
  Neck:
    name: SequenceEncoder
    encoder_type: rnn
    hidden_size: 48
  Head:
    name: CTCHead

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ../datasets/ocr
    label_file_list:
      - ../datasets/ocr/train_list.txt
  loader:
    shuffle: true
    batch_size_per_card: 64
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ../datasets/ocr
    label_file_list:
      - ../datasets/ocr/val_list.txt
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 64
    num_workers: 4
"""
    
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "rec_digit.yml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"Configuration saved to: {config_file}")

if __name__ == "__main__":
    train_paddleocr()