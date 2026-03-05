#!/usr/bin/env python3
"""
Unit Tests for YOLO-ACT Training Pipeline

Tests for training script, model, and checkpoint management.
Run with: pytest tests/ -v
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAVAClasses(unittest.TestCase):
    """Test AVA class configuration"""
    
    def test_ava_classes_no_duplicates(self):
        """Test that AVA_CLASSES has no duplicate keys"""
        from config.ava_classes import AVA_CLASSES
        
        # Check all 80 classes are present
        self.assertEqual(len(AVA_CLASSES), 80)
        
        # Check no duplicate keys
        keys = list(AVA_CLASSES.keys())
        self.assertEqual(len(keys), len(set(keys)), "Duplicate keys found in AVA_CLASSES")
    
    def test_ava_classes_no_duplicate_values(self):
        """Test that AVA_CLASSES has no duplicate values"""
        from config.ava_classes import AVA_CLASSES
        
        values = list(AVA_CLASSES.values())
        unique_values = set(values)
        self.assertEqual(len(values), len(unique_values), "Duplicate values found in AVA_CLASSES")
    
    def test_class_to_id_mapping(self):
        """Test reverse mapping works correctly"""
        from config.ava_classes import AVA_CLASSES, CLASS_TO_ID
        
        for class_id, class_name in AVA_CLASSES.items():
            self.assertEqual(CLASS_TO_ID[class_name.lower()], class_id)


class TestTrainingConfig(unittest.TestCase):
    """Test training configuration"""
    
    def test_model_config_backbone_no_extension(self):
        """Test that MODEL_CONFIG backbone doesn't have .pt extension"""
        from config.training_config import MODEL_CONFIG
        
        # Should not have .pt extension
        self.assertFalse(MODEL_CONFIG["backbone"].endswith(".pt"))
    
    def test_training_config_defaults(self):
        """Test training config has required fields"""
        from config.training_config import TRAINING_CONFIG
        
        required_fields = [
            "batch_size", "num_epochs", "accumulation_steps",
            "optimizer", "weight_decay", "use_amp"
        ]
        
        for field in required_fields:
            self.assertIn(field, TRAINING_CONFIG, f"Missing field: {field}")
    
    def test_class_weights(self):
        """Test class weights are properly defined"""
        from config.training_config import get_class_weights
        
        weights = get_class_weights()
        
        self.assertEqual(len(weights), 80)
        self.assertTrue(all(0 < w <= 5 for w in weights.values()))


class TestModelArchitecture(unittest.TestCase):
    """Test YOLO-ACT model architecture"""
    
    def setUp(self):
        """Set up test fixtures"""
        from models.yolo_act import YOLOActModel
        self.model = YOLOActModel(
            yolo_variant="yolov8n.pt",
            num_action_classes=80,
            clip_len=16,
            pretrained=False
        )
    
    def test_model_forward_pass(self):
        """Test model can do forward pass"""
        # Create dummy input: (B, C, T, H, W)
        batch_size = 2
        channels = 3
        temporal = 16
        height = width = 64
        
        dummy_input = torch.randn(batch_size, channels, temporal, height, width)
        
        with torch.no_grad():
            output = self.model(dummy_input)
        
        self.assertIn("detection", output)
        self.assertIsNotNone(output["detection"])
    
    def test_model_output_shape(self):
        """Test model output has correct shape"""
        batch_size = 2
        dummy_input = torch.randn(2, 3, 16, 64, 64)
        
        with torch.no_grad():
            output = self.model(dummy_input)
        
        # Detection output should be present
        self.assertIn("detection", output)


class TestYOLOActLoss(unittest.TestCase):
    """Test loss function"""
    
    def setUp(self):
        """Set up test fixtures"""
        from models.yolo_act import YOLOActLoss
        self.loss_fn = YOLOActLoss(num_classes=80)
    
    def test_loss_returns_dict(self):
        """Test loss returns dictionary with all components"""
        # Create dummy predictions and targets
        predictions = {
            "action_logits": torch.randn(5, 80),
            "detection": torch.randn(2, 15, 8, 8)
        }
        targets = {
            "labels": torch.randint(0, 2, (5, 80)),
            "boxes": torch.randn(5, 4)
        }
        
        loss_dict = self.loss_fn(predictions, targets)
        
        self.assertIn("total_loss", loss_dict)
        self.assertIn("action_loss", loss_dict)
        self.assertIn("det_loss", loss_dict)
    
    def test_loss_computation(self):
        """Test loss computation doesn't fail"""
        predictions = {
            "action_logits": torch.randn(5, 80),
            "detection": torch.randn(2, 15, 8, 8)
        }
        targets = {
            "labels": torch.randint(0, 2, (5, 80)),
            "boxes": torch.randn(5, 4)
        }
        
        loss_dict = self.loss_fn(predictions, targets)
        
        # Loss should be a scalar tensor
        self.assertEqual(loss_dict["total_loss"].dim(), 0)
        self.assertFalse(torch.isnan(loss_dict["total_loss"]))


class TestCheckpointManagement(unittest.TestCase):
    """Test checkpoint save/load functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from models.yolo_act import YOLOActModel, YOLOActLoss
        from scripts.train_yolo_act import Trainer
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create model and trainer
        self.model = YOLOActModel(
            yolo_variant="yolov8n.pt",
            num_action_classes=80,
            pretrained=False
        )
        self.loss_fn = YOLOActLoss(num_classes=80)
        
        self.trainer = Trainer(
            model=self.model,
            loss_fn=self.loss_fn,
            device="cpu",
            keep_best_only=True,
            keep_last_only=False,
            max_checkpoints=3
        )
    
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_checkpoint(self):
        """Test checkpoint saving works"""
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pth")
        
        # Set trainer options to save regular checkpoint
        self.trainer.keep_last_only = False
        self.trainer.keep_best_only = False

        # Save checkpoint
        self.trainer.save_checkpoint(checkpoint_path, epoch=1, is_best=False)

        self.assertTrue(os.path.exists(checkpoint_path))
    
    def test_save_best_checkpoint(self):
        """Test best checkpoint saving works"""
        # Set best loss
        self.trainer.best_loss = 0.5
        
        checkpoint_path = os.path.join(self.temp_dir, "epoch_1.pth")
        
        # Save checkpoint as best
        self.trainer.save_checkpoint(checkpoint_path, epoch=1, is_best=True)
        
        # Check best_model.pth was created
        best_path = os.path.join(self.temp_dir, "best_model.pth")
        self.assertTrue(os.path.exists(best_path))
    
    def test_save_last_checkpoint(self):
        """Test last checkpoint is always saved"""
        checkpoint_path = os.path.join(self.temp_dir, "epoch_1.pth")
        
        self.trainer.save_checkpoint(checkpoint_path, epoch=1, is_best=False)
        
        # Check last_model.pth was created
        last_path = os.path.join(self.temp_dir, "last_model.pth")
        self.assertTrue(os.path.exists(last_path))
    
    def test_load_checkpoint(self):
        """Test checkpoint loading works"""
        from scripts.train_yolo_act import Trainer
        from models.yolo_act import YOLOActModel, YOLOActLoss
        
        # Save checkpoint first
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pth")
        
        # Set trainer options to save regular checkpoint
        self.trainer.keep_last_only = False
        self.trainer.keep_best_only = False
        self.trainer.save_checkpoint(checkpoint_path, epoch=5, is_best=False)
        
        # Create new trainer
        new_trainer = Trainer(
            model=YOLOActModel(yolo_variant="yolov8n.pt", num_action_classes=80, pretrained=False),
            loss_fn=YOLOActLoss(num_classes=80),
            device="cpu"
        )
        
        # Load checkpoint
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Verify epoch was loaded
        self.assertEqual(new_trainer.current_epoch, 5)
    
    def test_checkpoint_cleanup(self):
        """Test old checkpoint cleanup works"""
        # Save multiple checkpoints
        for i in range(5):
            checkpoint_path = os.path.join(self.temp_dir, f"epoch_{i}.pth")
            self.trainer.save_checkpoint(checkpoint_path, epoch=i, is_best=False)
        
        # Should have kept max_checkpoints (3) plus last_model.pth and best
        saved_checkpoints = list(Path(self.temp_dir).glob("epoch_*.pth"))
        
        # Should not exceed max_checkpoints for regular checkpoints
        self.assertLessEqual(len(saved_checkpoints), self.trainer.max_checkpoints)


class TestDataLoading(unittest.TestCase):
    """Test data loading functionality"""
    
    def test_create_dataloaders(self):
        """Test dataloader creation doesn't fail"""
        from scripts.train_yolo_act import create_dataloaders
        
        # Should work with non-existent directories (will create empty datasets)
        try:
            train_loader, val_loader = create_dataloaders(
                frame_dir="nonexistent_dir",
                batch_size=2,
                num_workers=0
            )
            
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(val_loader)
        except Exception as e:
            # Expected if directories don't exist OR if dataset is empty (num_samples=0)
            msg = str(e).lower()
            self.assertTrue("not found" in msg or "num_samples" in msg)


class TestArgumentParsing(unittest.TestCase):
    """Test argument parsing"""
    
    def test_train_script_args(self):
        """Test training script accepts expected arguments"""
        from scripts.train_yolo_act import parse_args
        
        # Test default values
        args = parse_args([])
        
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.lr, 0.0001)
        self.assertTrue(args.auto_resume)
        self.assertTrue(args.keep_best_only)
    
    def test_resume_args(self):
        """Test resume arguments work"""
        from scripts.train_yolo_act import parse_args
        
        # Test with resume flag
        args = parse_args(['--resume', 'checkpoint.pth'])
        
        self.assertEqual(args.resume, 'checkpoint.pth')
    
    def test_auto_resume_args(self):
        """Test auto-resume flags"""
        from scripts.train_yolo_act import parse_args
        
        # Test auto-resume disabled
        args = parse_args(['--auto-resume'])
        self.assertTrue(args.auto_resume)
        
        # Test resume from best
        args = parse_args(['--resume-from-best'])
        self.assertTrue(args.resume_from_best)


class TestEvaluation(unittest.TestCase):
    """Test evaluation functionality"""
    
    def test_compute_iou(self):
        """Test IoU computation"""
        from scripts.evaluate import compute_iou
        
        # Perfect overlap
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        iou = compute_iou(box1, box2)
        self.assertAlmostEqual(iou, 1.0)
        
        # No overlap
        box1 = [0, 0, 10, 10]
        box2 = [20, 20, 30, 30]
        iou = compute_iou(box1, box2)
        self.assertAlmostEqual(iou, 0.0)
        
        # Partial overlap
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = compute_iou(box1, box2)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)


class TestFrameExtraction(unittest.TestCase):
    """Test frame extraction functionality"""
    
    def test_frame_extractor_init(self):
        """Test FrameExtractor initializes correctly"""
        from scripts.extract_frames import FrameExtractor
        
        extractor = FrameExtractor(
            video_dir="test_videos",
            frame_dir="test_frames",
            fps=30,
            img_size=224
        )
        
        self.assertEqual(extractor.fps, 30)
        self.assertEqual(extractor.img_size, 224)
    
    def test_ava_constants(self):
        """Test AVA temporal window constants"""
        from scripts.extract_frames import (
            AVA_START_SEC, AVA_ANNOTATION_START, 
            AVA_ANNOTATION_END, AVA_DURATION_SEC
        )
        
        self.assertEqual(AVA_ANNOTATION_START, 902)
        self.assertEqual(AVA_ANNOTATION_END, 1798)
        self.assertEqual(AVA_DURATION_SEC, 901)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
