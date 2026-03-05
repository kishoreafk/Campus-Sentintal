#!/usr/bin/env python3
"""
YOLO-ACT Training Script

Complete training pipeline for YOLO-ACT model with AVA dataset.

Usage:
    python scripts/train_yolo_act.py
    python scripts/train_yolo_act.py --epochs 50 --batch-size 8
    python scripts/train_yolo_act.py --resume models/checkpoints/best_model.pth
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.ava_classes import AVA_CLASSES, get_class_ids_from_names
from config.training_config import (
    DATASET_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, PATHS, HARDWARE_CONFIG,
    get_class_weights
)
from models.yolo_act import YOLOActModel, YOLOActLoss, create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AVADataset(Dataset):
    """AVA Dataset for action detection"""
    
    def __init__(
        self,
        frame_dir: str,
        annotation_file: str = None,
        img_size: int = 224,
        num_frames: int = 16,
        transform=None,
        selected_classes: List[int] = None,
        use_proposals: bool = False,
        proposal_file: str = None
    ):
        self.frame_dir = Path(frame_dir)
        self.img_size = img_size
        self.num_frames = num_frames
        self.transform = transform
        self.selected_classes = selected_classes
        
        # Load annotations
        self.samples = self._load_annotations(annotation_file)
        
        # Build class mapping for selected classes
        if selected_classes:
            self.class_to_idx = {c: i for i, c in enumerate(sorted(selected_classes))}
            self.num_classes = len(selected_classes)
        else:
            self.class_to_idx = {c: c - 1 for c in range(1, 81)}
            self.num_classes = 80
        
        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _load_annotations(self, annotation_file: str) -> List[Dict]:
        """Load annotations from file"""
        samples = []
        
        if annotation_file and Path(annotation_file).exists():
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        samples.append({
                            "filename": parts[0],
                            "action_id": int(parts[1]) if len(parts) > 1 else 1,
                            "bbox": [float(x) for x in parts[2:6]] if len(parts) > 5 else [0, 0, 1, 1]
                        })
        
        # If no annotations, find all frame files
        if not samples:
            for frame_file in self.frame_dir.glob("*.jpg"):
                samples.append({
                    "filename": frame_file.name,
                    "action_id": 1,
                    "bbox": [0, 0, 1, 1]
                })
        
        logger.info(f"Loaded {len(samples)} samples from {self.frame_dir}")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_path = self.frame_dir / sample["filename"]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Get action label
        action_id = sample["action_id"]
        
        # Map to selected class index
        if self.selected_classes and action_id in self.selected_classes:
            label = self.class_to_idx[action_id]
        else:
            label = 0  # Background
        
        # Get bbox
        bbox = sample["bbox"]
        
        return {
            "image": img,
            "label": label,
            "bbox": torch.tensor(bbox),
            "filename": sample["filename"]
        }


class Trainer:
    """Trainer for YOLO-ACT"""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: str = "cuda",
        lr: float = 0.0001,
        weight_decay: float = 0.0001,
        accumulation_steps: int = 4,
        use_amp: bool = True,
        keep_best_only: bool = True,
        keep_last_only: bool = False,
        max_checkpoints: int = 5,
        encrypt_checkpoints: bool = False,
        encryption_key: str = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.keep_best_only = keep_best_only
        self.keep_last_only = keep_last_only
        self.max_checkpoints = max_checkpoints
        self.encrypt_checkpoints = encrypt_checkpoints
        self.encryption_key = encryption_key or "default_key_change_in_production"
        self.saved_checkpoints = []  # Track saved checkpoints for cleanup
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        
        total_loss = 0
        total_cls_loss = 0
        total_bbox_loss = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            bboxes = batch["bbox"].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    
                    # Compute loss
                    loss_dict = self.loss_fn(outputs, {
                        "labels": labels,
                        "bboxes": bboxes
                    })
                    loss = loss_dict["total_loss"]
                    cls_loss = loss_dict["action_loss"].item()
                    bbox_loss = loss_dict["det_loss"].item()
                
                # Backward pass
                self.scaler.scale(loss / self.accumulation_steps).backward()
                
                # Update weights
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss_dict = self.loss_fn(outputs, {
                    "labels": labels,
                    "bboxes": bboxes
                })
                loss = loss_dict["total_loss"]
                cls_loss = loss_dict["action_loss"].item()
                bbox_loss = loss_dict["det_loss"].item()
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            total_cls_loss += cls_loss
            total_bbox_loss += bbox_loss
            self.global_step += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        
        return {
            "loss": avg_loss,
            "cls_loss": total_cls_loss / num_batches,
            "bbox_loss": total_bbox_loss / num_batches
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        
        self.model.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                bboxes = batch["bbox"].to(self.device)
                
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)
                        loss_dict = self.loss_fn(outputs, {
                            "labels": labels,
                            "bboxes": bboxes
                        })
                else:
                    outputs = self.model(images)
                    loss_dict = self.loss_fn(outputs, {
                        "labels": labels,
                        "bboxes": bboxes
                    })
                
                total_loss += loss_dict["total_loss"].item()
        
        avg_loss = total_loss / len(dataloader)
        
        return {
            "val_loss": avg_loss
        }
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        is_best: bool = False
    ):
        """Save model checkpoint with optional encryption and cleanup"""
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "global_step": self.global_step,
            "keep_best_only": self.keep_best_only,
            "keep_last_only": self.keep_last_only
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save last checkpoint regardless of settings
        last_path = str(checkpoint_path.parent / "last_model.pth")
        torch.save(checkpoint, last_path)
        logger.info(f"Saved last model to {last_path}")
        
        # Handle checkpoint saving based on options
        if self.keep_last_only:
            # Only keep the last checkpoint (already saved above)
            pass
        elif self.keep_best_only and not is_best:
            # Only save if this is the best model
            pass
        else:
            # Save regular checkpoint
            torch.save(checkpoint, path)
            self.saved_checkpoints.append(path)
            logger.info(f"Saved checkpoint to {path}")
        
        # Save best model
        if is_best:
            best_path = str(checkpoint_path.parent / "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path} (loss: {self.best_loss:.4f})")
            # Also track best checkpoint
            best_checkpoint_path = str(checkpoint_path.parent / "best_model.pth")
            if best_checkpoint_path not in self.saved_checkpoints:
                self.saved_checkpoints.append(best_checkpoint_path)
        
        # Auto-cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones"""
        if self.keep_best_only or self.keep_last_only:
            return  # No cleanup needed
        
        if len(self.saved_checkpoints) > self.max_checkpoints:
            # Remove oldest checkpoints
            checkpoints_to_remove = self.saved_checkpoints[:-self.max_checkpoints]
            for ckpt_path in checkpoints_to_remove:
                try:
                    path_obj = Path(ckpt_path)
                    # Don't remove best_model.pth or last_model.pth
                    if path_obj.name not in ["best_model.pth", "last_model.pth"]:
                        if path_obj.exists():
                            path_obj.unlink()
                            logger.info(f"Removed old checkpoint: {ckpt_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {ckpt_path}: {e}")
            
            # Keep only the most recent checkpoints
            self.saved_checkpoints = self.saved_checkpoints[-self.max_checkpoints:]
    
    def save_encrypted_checkpoint(
        self,
        path: str,
        epoch: int,
        is_best: bool = False
    ):
        """Save an encrypted model checkpoint"""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            logger.warning("cryptography not installed, saving unencrypted checkpoint")
            return self.save_checkpoint(path, epoch, is_best)
        
        # Generate or use provided key
        key = self.encryption_key.encode() if isinstance(self.encryption_key, str) else self.encryption_key
        fernet = Fernet(key)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "global_step": self.global_step,
            "encrypted": True
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # Serialize and encrypt
        import io
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        encrypted_data = fernet.encrypt(buffer.getvalue())
        
        # Save encrypted checkpoint
        encrypted_path = Path(path)
        encrypted_path.parent.mkdir(parents=True, exist_ok=True)
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
        
        logger.info(f"Saved encrypted checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint, supporting encrypted checkpoints"""
        path_obj = Path(path)
        
        # Check if file is encrypted
        try:
            with open(path_obj, 'rb') as f:
                first_bytes = f.read(10)
            
            # Check for Fernet encryption magic bytes
            if first_bytes.startswith(b'gAAAAAB'):
                logger.info("Loading encrypted checkpoint...")
                return self._load_encrypted_checkpoint(path)
        except Exception as e:
            logger.warning(f"Error checking encryption: {e}")
        
        # Load regular checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss = checkpoint["best_loss"]
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path}, epoch {self.current_epoch}")
    
    def _load_encrypted_checkpoint(self, path: str):
        """Load an encrypted checkpoint"""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            logger.error("cryptography not installed, cannot load encrypted checkpoint")
            return
        
        key = self.encryption_key.encode() if isinstance(self.encryption_key, str) else self.encryption_key
        fernet = Fernet(key)
        
        with open(path, 'rb') as f:
            encrypted_data = f.read()
        
        try:
            decrypted_data = fernet.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Failed to decrypt checkpoint: {e}")
            return
        
        import io
        buffer = io.BytesIO(decrypted_data)
        checkpoint = torch.load(buffer, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss = checkpoint["best_loss"]
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Loaded encrypted checkpoint from {path}, epoch {self.current_epoch}")


def debug_batch(dataloader: DataLoader):
    """Debug function to verify batch shapes and formats"""
    logger.info("Running batch debug check...")
    
    batch = next(iter(dataloader))
    
    print(f"  Clip shape: {batch['image'].shape}")
    print(f"    Expected: (B, C, T, H, W)")
    print(f"  Labels shape: {batch['label'].shape}")
    print(f"    Expected: (B, num_classes)")
    print(f"  Boxes shape: {batch['bbox'].shape}")
    print(f"    Expected: (B, 4)")
    
    # Verify normalization
    clip_min = batch['image'].min().item()
    clip_max = batch['image'].max().item()
    print(f"  Clip value range: [{clip_min:.3f}, {clip_max:.3f}]")
    
    if clip_min < 0 or clip_max > 1:
        print("  WARNING: Clip not normalized to [0, 1]!")
    
    # Check for multi-label (only valid when label is 2-D one-hot)
    label_tensor = batch['label']
    if label_tensor.dim() > 1:
        labels_sum = label_tensor.sum(dim=1)
        multi_label = (labels_sum > 1).sum().item()
        print(f"  Multi-label samples: {multi_label}/{label_tensor.shape[0]}")
    else:
        print(f"  Labels are scalar class indices (not multi-hot)")
    
    print("  Debug check complete!")


def create_dataloaders(
    frame_dir: str = "data/frames",
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: int = 224,
    num_frames: int = 16,
    selected_classes: List[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders"""
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = AVADataset(
        frame_dir=os.path.join(frame_dir, "train"),
        img_size=img_size,
        num_frames=num_frames,
        transform=train_transform,
        selected_classes=selected_classes
    )
    
    val_dataset = AVADataset(
        frame_dir=os.path.join(frame_dir, "val"),
        img_size=img_size,
        num_frames=num_frames,
        transform=val_transform,
        selected_classes=selected_classes
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    device: str = "cuda",
    checkpoint_dir: str = "models/checkpoints",
    log_dir: str = "logs",
    lr: float = 0.0001,
    weight_decay: float = 0.0001,
    save_interval: int = 5,
    use_amp: bool = True,
    start_epoch: int = 0
) -> Dict:
    """Main training function"""
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=YOLOActLoss(num_classes=80),
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        use_amp=use_amp,
        keep_best_only=True,  # Default to keep best only
        keep_last_only=False,
        max_checkpoints=5,
        encrypt_checkpoints=False
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        trainer.optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Create directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": []
    }
    
    logger.info("Starting training...")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Mixed precision: {use_amp}")
    
    for epoch in range(start_epoch, num_epochs):
        trainer.current_epoch = epoch
        
        epoch_start = time.time()
        
        # Train
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        
        # Update learning rate
        scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Log metrics
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Save history
        history["train_loss"].append(train_metrics['loss'])
        history["val_loss"].append(val_metrics['val_loss'])
        history["learning_rate"].append(current_lr)
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        
        is_best = val_metrics['val_loss'] < trainer.best_loss
        if is_best:
            trainer.best_loss = val_metrics['val_loss']
        
        if (epoch + 1) % save_interval == 0 or is_best:
            trainer.save_checkpoint(
                checkpoint_path,
                epoch=epoch + 1,
                is_best=is_best
            )
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch time: {epoch_time:.2f}s")
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {trainer.best_loss:.4f}")
    
    return history


def parse_args(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train YOLO-ACT model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python scripts/train_yolo_act.py
  
  # Train with custom settings
  python scripts/train_yolo_act.py --epochs 50 --batch-size 8 --lr 0.0001
  
  # Resume training from checkpoint
  python scripts/train_yolo_act.py --resume models/checkpoints/checkpoint_epoch_10.pth
        """
    )
    
    parser.add_argument(
        '--frame-dir',
        type=str,
        default='data/frames',
        help='Directory containing extracted frames'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='models/checkpoints',
        help='Directory to save checkpoints'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for logs'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=4,
        help='Batch size'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0001,
        help='Weight decay'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=416,
        help='Input image size'
    )
    
    parser.add_argument(
        '--num-frames',
        type=int,
        default=16,
        help='Number of frames per sample'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    parser.add_argument(
        '--save-interval',
        type=int,
        default=5,
        help='Save checkpoint every N epochs'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        default=True,
        help='Auto-resume from last checkpoint if available (default: True)'
    )
    
    parser.add_argument(
        '--resume-from-best',
        action='store_true',
        default=False,
        help='Resume training from best model checkpoint'
    )
    
    parser.add_argument(
        '--backbone',
        type=str,
        default='yolov8m.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='YOLO backbone variant'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable mixed precision training'
    )
    
    parser.add_argument(
        '--classes',
        type=str,
        default=None,
        help='Comma-separated list of action classes to train on'
    )
    
    # Checkpoint management options
    parser.add_argument(
        '--keep-best-only',
        action='store_true',
        default=True,
        help='Only keep the best model checkpoint (default: True)'
    )
    
    parser.add_argument(
        '--keep-last-only',
        action='store_true',
        default=False,
        help='Only keep the last checkpoint, delete others'
    )
    
    parser.add_argument(
        '--max-checkpoints',
        type=int,
        default=5,
        help='Maximum number of checkpoints to keep when not using best/last only'
    )
    
    parser.add_argument(
        '--encrypt-checkpoints',
        action='store_true',
        default=False,
        help='Enable checkpoint encryption'
    )
    
    parser.add_argument(
        '--encryption-key',
        type=str,
        default=None,
        help='Encryption key for checkpoints (required if --encrypt-checkpoints is set)'
    )
    
    return parser.parse_args(args)


def main():
    """Main function"""
    args = parse_args()
    
    # Determine device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Parse selected classes
    selected_classes = None
    if args.classes:
        selected_classes = get_class_ids_from_names(
            [c.strip() for c in args.classes.split(',')]
        )
        logger.info(f"Training on {len(selected_classes)} classes: {selected_classes}")
    
    # Create model
    logger.info("Creating model...")
    model, loss_fn = create_model(
        num_classes=80,
        yolo_variant=args.backbone,
        pretrained=True,
        clip_len=args.num_frames,
        device=args.device
    )
    
    # Create trainer for managing training state
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp,
        keep_best_only=args.keep_best_only,
        keep_last_only=args.keep_last_only,
        max_checkpoints=args.max_checkpoints,
        encrypt_checkpoints=args.encrypt_checkpoints,
        encryption_key=args.encryption_key
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        start_epoch = trainer.current_epoch
    elif args.auto_resume:
        # Auto-resume from last checkpoint if available
        checkpoint_dir = Path(args.checkpoint_dir)
        last_checkpoint = checkpoint_dir / "last_model.pth"
        best_checkpoint = checkpoint_dir / "best_model.pth"
        
        if args.resume_from_best and best_checkpoint.exists():
            logger.info(f"Auto-resuming from best checkpoint: {best_checkpoint}")
            trainer.load_checkpoint(str(best_checkpoint))
            start_epoch = trainer.current_epoch
        elif last_checkpoint.exists():
            logger.info(f"Auto-resuming from last checkpoint: {last_checkpoint}")
            trainer.load_checkpoint(str(last_checkpoint))
            start_epoch = trainer.current_epoch
        else:
            logger.info("No checkpoint found, starting training from scratch")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        frame_dir=args.frame_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        num_frames=args.num_frames,
        selected_classes=selected_classes
    )
    
    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_interval=args.save_interval,
        use_amp=not args.no_amp,
        start_epoch=start_epoch
    )
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Final learning rate: {history['learning_rate'][-1]:.6f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
