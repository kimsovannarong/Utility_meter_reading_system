#!/usr/bin/env python3
"""
DETR Training Script for YOLO format datasets
"""

import os
import sys
import json
import yaml
import torch
import time
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from transformers import DetrForObjectDetection, DetrImageProcessor
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False
    print("Please install DETR dependencies: pip install transformers torchvision")

class YOLODataset(Dataset):
    """
    Custom dataset that reads YOLO format directly for DETR training
    """
    
    def __init__(self, dataset_path, split='train', processor=None, img_size=800):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.processor = processor
        self.img_size = img_size
        
        # Load data.yaml
        with open(self.dataset_path / 'data.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config['names']
        self.num_classes = self.config['nc']
        
        # Get image and label paths
        self.images_dir = self.dataset_path / split / 'images'
        self.labels_dir = self.dataset_path / split / 'labels'
        
        # Get all image files
        self.image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        
        print(f"Loaded {len(self.image_files)} images from {split} split")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        try:
            # Get image path
            img_path = self.image_files[idx]
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            orig_width, orig_height = image.size
            
            # Load corresponding label file
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            
            boxes = []
            labels = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert YOLO format to COCO format bboxes [x_min, y_min, width, height]
                        x_min = (x_center - width/2) * orig_width
                        y_min = (y_center - height/2) * orig_height
                        box_width = width * orig_width
                        box_height = height * orig_height
                        
                        boxes.append([x_min, y_min, box_width, box_height])
                        labels.append(class_id)
            
            # Convert to tensors
            if len(boxes) > 0:
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.long)
            else:
                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                labels_tensor = torch.zeros(0, dtype=torch.long)
            
            # Process image only with processor
            encoding = self.processor(
                images=image, 
                return_tensors="pt",
                size={"height": self.img_size, "width": self.img_size}
            )
            
            # Prepare target for DETR
            target = {
                'image_id': torch.tensor([idx], dtype=torch.int64),
                'class_labels': labels_tensor,
                'boxes': boxes_tensor,
                'size': torch.tensor([orig_height, orig_width]),  # [height, width]
                'orig_size': torch.tensor([orig_height, orig_width])
            }
            
            return {
                'pixel_values': encoding['pixel_values'].squeeze(0),
                'pixel_mask': encoding['pixel_mask'].squeeze(0),
                'labels': target
            }
            
        except Exception as e:
            print(f"Error loading image {idx}: {e}")
            # Return dummy data for this image
            return {
                'pixel_values': torch.zeros((3, self.img_size, self.img_size)),
                'pixel_mask': torch.zeros((self.img_size, self.img_size)),
                'labels': {
                    'image_id': torch.tensor([idx]),
                    'class_labels': torch.zeros(0, dtype=torch.long),
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'size': torch.tensor([self.img_size, self.img_size]),
                    'orig_size': torch.tensor([self.img_size, self.img_size])
                }
            }

class DETRTrainerYOLO:
    """DETR Trainer that works directly with YOLO format"""
    
    def __init__(self, config_path=None):
        self.model_name = 'detr'
        self.device = self._setup_device()
        self.config = self._load_config(config_path)
        self._setup_directories()
        
        if not DETR_AVAILABLE:
            print("Dependencies missing. Install: pip install transformers torchvision")
            return
        
        # Get number of classes from YOLO config
        self.num_classes = self._get_num_classes()
        
        # Initialize DETR
        print(f"Loading DETR model for {self.num_classes} classes...")
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=self.num_classes,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        
        print(f"✓ DETR model loaded successfully")
    
    def _setup_device(self):
        """Setup training device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("✓ Using Apple MPS (Metal)")
        else:
            device = torch.device('cpu')
            print("✓ Using CPU for training")
        return device
    
    def _load_config(self, config_path):
        """Load configuration"""
        default_config = {
            'dataset_path': 'datasets/detection',
            'epochs': 25,
            'batch_size': 2,
            'learning_rate': 1e-4,
            'project': 'models/detectors/detr',
            'img_size': 800,
            'save_checkpoint_every': 5
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            'models/detectors/detr',
            'experiments/detr',
            'results/detr'
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _get_num_classes(self):
        """Get number of classes from YOLO dataset"""
        dataset_path = self.config.get('dataset_path', 'datasets/detection')
        config_file = Path(dataset_path) / 'data.yaml'
        
        if not config_file.exists():
            print(f"Error: data.yaml not found at {config_file}")
            return 4  # Default for your meter detection (1, digits, electricity, water)
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        return config.get('nc', 4)
    
    def check_dataset(self):
        """Check if YOLO dataset exists and is valid"""
        print(f"\n{'='*60}")
        print("DATASET VERIFICATION")
        print(f"{'='*60}")
        
        dataset_path = self.config.get('dataset_path', 'datasets/detection')
        
        if not os.path.exists(dataset_path):
            print(f"✗ Dataset not found: {dataset_path}")
            return False
        
        # Check data.yaml
        config_file = Path(dataset_path) / 'data.yaml'
        if not config_file.exists():
            print(f"✗ data.yaml not found: {config_file}")
            return False
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Dataset: {dataset_path}")
        print(f"  Classes: {config.get('names', [])}")
        print(f"  Num classes: {config.get('nc', 0)}")
        
        # Check splits
        splits = ['train', 'valid']
        for split in splits:
            img_dir = Path(dataset_path) / split / 'images'
            lbl_dir = Path(dataset_path) / split / 'labels'
            
            if img_dir.exists() and lbl_dir.exists():
                images = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
                labels = len(list(lbl_dir.glob("*.txt")))
                print(f"✓ {split}: {images} images, {labels} labels")
            else:
                print(f"✗ {split} directory missing")
                return False
        
        return True
    
    def create_datasets(self):
        """Create train and validation datasets from YOLO format"""
        dataset_path = self.config.get('dataset_path', 'datasets/detection')
        
        print(f"\nCreating datasets from YOLO format...")
        
        train_dataset = YOLODataset(
            dataset_path=dataset_path,
            split='train',
            processor=self.processor,
            img_size=self.config.get('img_size', 800)
        )
        
        val_dataset = YOLODataset(
            dataset_path=dataset_path,
            split='valid',
            processor=self.processor,
            img_size=self.config.get('img_size', 800)
        )
        
        print(f"✓ Datasets created:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def collate_fn(self, batch):
        """Custom collate function for DETR training"""
        # Filter out None or invalid entries
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
        
        # Handle labels
        labels = []
        for item in batch:
            label_dict = item["labels"]
            labels.append({
                'image_id': label_dict['image_id'],
                'class_labels': label_dict['class_labels'],
                'boxes': label_dict['boxes'],
                'size': label_dict['size'],
                'orig_size': label_dict['orig_size']
            })
        
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels
        }
    
    def train(self, epochs=None, batch_size=None):
        """Train DETR model directly with YOLO format"""
        print(f"\n{'='*60}")
        print(f"DETR TRAINING (YOLO Format)")
        print(f"{'='*60}")
        
        if not DETR_AVAILABLE:
            print("DETR not available. Install transformers first.")
            return False
        
        # Use provided values or config defaults
        epochs = epochs or self.config.get('epochs', 25)
        batch_size = batch_size or self.config.get('batch_size', 2)
        
        # Check dataset
        if not self.check_dataset():
            return False
        
        # Create datasets
        train_dataset, val_dataset = self.create_datasets()
        if len(train_dataset) == 0:
            print("✗ No training data found!")
            return False
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=0
        )
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Num classes: {self.num_classes}")
        print(f"  Learning rate: {self.config.get('learning_rate', 1e-4)}")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=1e-4
        )
        
        # Training loop
        print(f"\nStarting DETR training for {epochs} epochs...")
        start_time = time.time()
        
        self.model.train()
        training_losses = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            epoch_loss = 0
            batch_count = 0
            
            # Progress bar
            progress_bar = tqdm(train_loader, desc="Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                if batch is None:
                    continue
                
                # Move batch to device
                pixel_values = batch["pixel_values"].to(self.device)
                pixel_mask = batch["pixel_mask"].to(self.device)
                labels = batch["labels"]
                
                # Prepare labels for DETR
                formatted_labels = []
                for label_set in labels:
                    # Convert boxes from [x, y, width, height] to [x_min, y_min, x_max, y_max]
                    if len(label_set['boxes']) > 0:
                        boxes = label_set['boxes']
                        # Convert COCO format to DETR format
                        boxes_detr = torch.zeros_like(boxes)
                        boxes_detr[:, 0] = boxes[:, 0]  # x_min
                        boxes_detr[:, 1] = boxes[:, 1]  # y_min
                        boxes_detr[:, 2] = boxes[:, 0] + boxes[:, 2]  # x_max
                        boxes_detr[:, 3] = boxes[:, 1] + boxes[:, 3]  # y_max
                        
                        formatted_labels.append({
                            'class_labels': label_set['class_labels'].to(self.device),
                            'boxes': boxes_detr.to(self.device),
                            'size': label_set['size'].to(self.device),
                            'orig_size': label_set['orig_size'].to(self.device)
                        })
                    else:
                        # Empty labels
                        formatted_labels.append({
                            'class_labels': torch.zeros(0, dtype=torch.long).to(self.device),
                            'boxes': torch.zeros((0, 4), dtype=torch.float32).to(self.device),
                            'size': label_set['size'].to(self.device),
                            'orig_size': label_set['orig_size'].to(self.device)
                        })
                
                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=formatted_labels
                )
                
                loss = outputs.loss
                epoch_loss += loss.item()
                batch_count += 1
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})
            
            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                training_losses.append(avg_epoch_loss)
                
                print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")
                
                # Save checkpoint
                if (epoch + 1) % self.config.get('save_checkpoint_every', 5) == 0:
                    checkpoint_path = f"models/detectors/detr/checkpoint_epoch{epoch+1}.pth"
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_epoch_loss,
                        'num_classes': self.num_classes
                    }, checkpoint_path)
                    print(f"✓ Checkpoint saved: {checkpoint_path}")
            else:
                print(f"Epoch {epoch + 1}: No valid batches processed")
        
        training_time = time.time() - start_time
        
        # Save final model if we have any training
        if training_losses:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_path = f"models/detectors/detr/meter_detr_yolo_e{epochs}_{timestamp}.pth"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'num_classes': self.num_classes,
                'epochs_trained': epochs,
                'training_time': training_time,
                'training_losses': training_losses,
                'config': self.config
            }, final_path)
            
            # Save experiment log
            self.save_experiment_log(
                epochs=epochs,
                batch_size=batch_size,
                training_time=training_time,
                final_loss=training_losses[-1] if training_losses else 0,
                model_path=final_path
            )
            
            print(f"\n{'='*60}")
            print(f"DETR TRAINING COMPLETE!")
            print(f"{'='*60}")
            print(f"✓ Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
            print(f"✓ Final model: {final_path}")
            print(f"✓ Final loss: {training_losses[-1]:.4f}" if training_losses else "")
        else:
            print("✗ Training failed - no valid batches processed")
            return False
        
        return True
    
    def save_experiment_log(self, **metrics):
        """Save experiment results"""
        result_data = {
            'model': 'detr_yolo',
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'device': str(self.device),
            'metrics': metrics
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = f"experiments/detr/exp_yolo_{timestamp}.json"
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        print(f"✓ Experiment log saved to: {result_file}")

def main():
    parser = argparse.ArgumentParser(description='Train DETR directly with YOLO format dataset')
    parser.add_argument('--config', type=str, default='configs/detr.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', '-e', type=int, default=None,
                       help='Number of epochs (default: 25)')
    parser.add_argument('--batch', '-b', type=int, default=None,
                       help='Batch size (default: 2)')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                       help='Path to YOLO format dataset')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DETR TRAINING - YOLO FORMAT")
    print("="*60)
    
    trainer = DETRTrainerYOLO(args.config)
    
    # Override dataset path if provided
    if args.dataset:
        trainer.config['dataset_path'] = args.dataset
    
    if DETR_AVAILABLE:
        success = trainer.train(epochs=args.epochs, batch_size=args.batch)
        if not success:
            print("\n✗ Training failed. Check the error messages above.")
    else:
        print("\nPlease install DETR dependencies:")
        print("pip install transformers torchvision")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()