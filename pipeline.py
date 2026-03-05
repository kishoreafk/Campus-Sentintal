#!/usr/bin/env python3
"""
YOLO-ACT Training Pipeline

Complete pipeline to train YOLO-ACT model with AVA dataset:
1. Download annotations
2. Download videos with multi-source fallback
3. Validate dataset (completeness, integrity, duplicates)
4. Extract frames from videos
5. Train the model

Usage:
    python pipeline.py
    python pipeline.py --classes kiss,hug,walk,run,sit,jump,fight
    python pipeline.py --skip-download --skip-extract
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.ava_classes import USER_SELECTED_CLASSES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline orchestrator"""
    
    def __init__(
        self,
        selected_classes: Optional[List[str]] = None,
        video_dir: str = "data/raw_videos",
        frame_dir: str = "data/frames",
        annotation_dir: str = "data/annotations",
        checkpoint_dir: str = "models/checkpoints",
        log_dir: str = "logs"
    ):
        self.selected_classes = selected_classes or USER_SELECTED_CLASSES
        self.video_dir = video_dir
        self.frame_dir = frame_dir
        self.annotation_dir = annotation_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.video_dir,
            self.frame_dir,
            self.annotation_dir,
            "data/ava/proposals",  # For proposal files
            self.checkpoint_dir,
            self.log_dir
        ]
        
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    def run_step(self, step_name: str, command: List[str], env: dict = None):
        """Run a pipeline step"""
        logger.info(f"\n{'='*60}")
        logger.info(f"STEP: {step_name}")
        logger.info(f"{'='*60}")
        
        try:
            result = subprocess.run(
                command,
                env=env or os.environ.copy(),
                check=True,
                capture_output=False
            )
            logger.info(f"Step '{step_name}' completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Step '{step_name}' failed with code {e.returncode}")
            return False
    
    def download_videos_orchestrator(self, max_videos: Optional[int] = None):
        """Step 1: Download videos using multi-source orchestrator"""
        
        cmd = [
            sys.executable,
            "scripts/download_orchestrator.py",
            "--video-dir", self.video_dir,
            "--annotation-dir", self.annotation_dir,
            "--max-workers", "8"
        ]
        
        if self.selected_classes:
            cmd.extend(["--classes", ",".join(self.selected_classes)])
        
        if max_videos:
            cmd.extend(["--max-videos", str(max_videos)])
        
        return self.run_step("Download Videos (Multi-Source)", cmd)
    
    def validate_dataset(self):
        """Step 2: Validate dataset integrity"""
        
        cmd = [
            sys.executable,
            "scripts/validate_dataset.py",
            "--video-dir", self.video_dir,
            "--annotation-dir", self.annotation_dir
        ]
        
        return self.run_step("Validate Dataset", cmd)
    
    def extract_frames(self, fps: int = 30, img_size: int = 416, max_videos: Optional[int] = None):
        """Step 3: Extract frames from videos"""
        
        cmd = [
            sys.executable,
            "scripts/extract_frames.py",
            "--video-dir", self.video_dir,
            "--frame-dir", self.frame_dir,
            "--annotation-dir", self.annotation_dir,
            "--fps", str(fps),
            "--img-size", str(img_size),
            "--generate-annotations"
        ]
        
        if max_videos:
            cmd.extend(["--max-videos", str(max_videos)])
        
        if self.selected_classes:
            cmd.extend(["--classes", ",".join(self.selected_classes)])
        
        return self.run_step("Extract Frames", cmd)
    
    def train(
        self,
        epochs: int = 50,
        batch_size: int = 4,
        lr: float = 0.001,
        img_size: int = 416,
        backbone: str = "yolov8m.pt",
        device: str = "cuda"
    ):
        """Step 4: Train the model"""
        
        cmd = [
            sys.executable,
            "scripts/train_yolo_act.py",
            "--frame-dir", self.frame_dir,
            "--checkpoint-dir", self.checkpoint_dir,
            "--log-dir", self.log_dir,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--img-size", str(img_size),
            "--backbone", backbone,
            "--device", device,
        ]
        
        if self.selected_classes:
            cmd.extend(["--classes", ",".join(self.selected_classes)])
        
        return self.run_step("Train Model", cmd)
    
    def run(
        self,
        skip_download: bool = False,
        skip_validate: bool = False,
        skip_extract: bool = False,
        skip_train: bool = False,
        max_videos: Optional[int] = None,
        epochs: int = 50,
        batch_size: int = 4,
        **train_kwargs
    ):
        """Run the complete pipeline"""
        
        logger.info("="*60)
        logger.info("YOLO-ACT TRAINING PIPELINE")
        logger.info("="*60)
        logger.info(f"Selected classes: {', '.join(self.selected_classes)}")
        logger.info(f"Video directory: {self.video_dir}")
        logger.info(f"Frame directory: {self.frame_dir}")
        logger.info("="*60)
        
        # Step 1: Download videos
        if not skip_download:
            logger.info("\n[Step 1/5] Downloading videos (multi-source)...")
            self.download_videos_orchestrator(max_videos=max_videos)
        else:
            logger.info("\n[Step 1/5] Skipping video download...")
        
        # Step 2: Validate dataset
        if not skip_validate and not skip_download:
            logger.info("\n[Step 2/5] Validating dataset...")
            self.validate_dataset()
        else:
            logger.info("\n[Step 2/5] Skipping validation...")
        
        # Step 3: Extract frames
        if not skip_extract:
            logger.info("\n[Step 3/5] Extracting frames...")
            self.extract_frames(max_videos=max_videos)
        else:
            logger.info("\n[Step 3/5] Skipping frame extraction...")
        
        # Step 4: Train
        if not skip_train:
            logger.info("\n[Step 4/5] Training model...")
            self.train(
                epochs=epochs,
                batch_size=batch_size,
                **train_kwargs
            )
        else:
            logger.info("\n[Step 4/5] Skipping training...")
        
        # Step 5: Evaluation
        logger.info("\n[Step 5/5] Pipeline complete!")
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)
        
        return True


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="YOLO-ACT Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default settings
  python pipeline.py
  
  # Download videos for specific classes
  python pipeline.py --classes kiss,hug,walk,run,sit,jump,fight
  
  # Quick test with limited videos
  python pipeline.py --max-videos 10 --epochs 10 --batch-size 2
  
  # Skip download and extraction, just train
  python pipeline.py --skip-download --skip-validate --skip-extract
        """
    )
    
    parser.add_argument(
        '--classes', '-c',
        type=str,
        default=None,
        help='Comma-separated action classes'
    )
    
    parser.add_argument(
        '--video-dir',
        type=str,
        default='data/raw_videos',
        help='Directory for videos'
    )
    
    parser.add_argument(
        '--frame-dir',
        type=str,
        default='data/frames',
        help='Directory for extracted frames'
    )
    
    parser.add_argument(
        '--annotation-dir',
        type=str,
        default='data/annotations',
        help='Directory for annotations'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='models/checkpoints',
        help='Directory for model checkpoints'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for logs'
    )
    
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Maximum number of videos to download'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=416,
        help='Input image size'
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
        help='Device (cuda or cpu)'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip video download step'
    )
    
    parser.add_argument(
        '--skip-validate',
        action='store_true',
        help='Skip dataset validation step'
    )
    
    parser.add_argument(
        '--skip-extract',
        action='store_true',
        help='Skip frame extraction step'
    )
    
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip training step'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse classes
    selected_classes = None
    if args.classes:
        selected_classes = [c.strip() for c in args.classes.split(',')]
    else:
        selected_classes = USER_SELECTED_CLASSES
    
    logger.info(f"Selected classes: {selected_classes}")
    
    # Create pipeline
    pipeline = Pipeline(
        selected_classes=selected_classes,
        video_dir=args.video_dir,
        frame_dir=args.frame_dir,
        annotation_dir=args.annotation_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Run pipeline
    success = pipeline.run(
        skip_download=args.skip_download,
        skip_validate=args.skip_validate,
        skip_extract=args.skip_extract,
        skip_train=args.skip_train,
        max_videos=args.max_videos,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        backbone=args.backbone,
        device=args.device
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
