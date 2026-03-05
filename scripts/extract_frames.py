#!/usr/bin/env python3
"""
AVA Frame Extraction Script

Extracts frames from AVA videos for training.
AVA-specific: Videos are 30min, but annotations only cover 902-1798s (15min window).

Key optimizations:
- Extract only the annotated temporal window (902-1798 seconds)
- Use Decord for faster frame loading (10x faster than OpenCV)
- Frame-accurate extraction
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from config.ava_classes import AVA_CLASSES, get_class_ids_from_names, ANNOTATION_FILES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# AVA temporal window constants
AVA_START_SEC = 900  # 15 minutes before annotation start
AVA_ANNOTATION_START = 902  # First annotated second
AVA_ANNOTATION_END = 1798   # Last annotated second
AVA_DURATION_SEC = 901  # 15 minutes 1 second (902 to 1798 inclusive)


class FrameExtractor:
    """Extract frames from AVA videos"""
    
    def __init__(
        self,
        video_dir: str = "data/raw_videos",
        frame_dir: str = "data/frames",
        annotation_dir: str = "data/annotations",
        fps: int = 30,
        img_size: int = 224,
        selected_classes: List[str] = None,
        use_decord: bool = True
    ):
        self.video_dir = Path(video_dir)
        self.frame_dir = Path(frame_dir)
        self.annotation_dir = Path(annotation_dir)
        self.fps = fps
        self.img_size = img_size
        self.selected_classes = selected_classes
        self.use_decord = use_decord
        
        # Create directories
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        (self.frame_dir / "train").mkdir(exist_ok=True)
        (self.frame_dir / "val").mkdir(exist_ok=True)
        
        # Try to import decord for faster video reading
        self.decord_available = False
        if use_decord:
            try:
                import decord
                from decord import VideoReader, cpu
                self.decord_available = True
                logger.info("Using Decord for faster frame extraction")
            except ImportError:
                logger.warning("Decord not available, using OpenCV")
        
        # Filter class IDs
        self.filter_class_ids = set()
        if selected_classes:
            self.filter_class_ids = set(get_class_ids_from_names(selected_classes))
    
    def get_video_files(self) -> List[Path]:
        """Get list of video files in the video directory"""
        if not self.video_dir.exists():
            logger.error(f"Video directory does not exist: {self.video_dir}")
            return []
        
        video_extensions = {'.mp4', '.webm', '.avi', '.mkv', '.mov'}
        
        video_files = []
        for video_file in self.video_dir.iterdir():
            if video_file.suffix.lower() in video_extensions:
                video_files.append(video_file)
        
        return sorted(video_files)
    
    def verify_video_duration(self, video_path: Path) -> Tuple[bool, float]:
        """Verify video has the required 30-minute duration (AVA videos)"""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                # AVA videos should be ~30 minutes (1800 seconds)
                if duration >= 1800:
                    return True, duration
                else:
                    return False, duration
            return False, 0.0
        except Exception as e:
            logger.error(f"Error checking video duration: {e}")
            return False, 0.0
    
    def extract_frames_ffmpeg(
        self,
        video_path: Path,
        output_dir: Path,
        start_sec: int = AVA_ANNOTATION_START,
        duration: int = AVA_DURATION_SEC
    ) -> int:
        """Extract frames using ffmpeg (frame-accurate)"""
        
        video_id = video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already extracted (expect ~27000 frames at 30 FPS for 901 seconds)
        existing_frames = list(output_dir.glob(f"{video_id}_*.jpg"))
        expected_frames = int(duration * self.fps)
        if len(existing_frames) >= expected_frames * 0.95:  # 95% threshold
            logger.info(f"Frames already exist for {video_id}, skipping")
            return len(existing_frames)
        
        # Extract frames using ffmpeg with frame-accurate seeking
        # Use -ss after -i for accurate frame selection
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),  # Seek to start
            "-i", str(video_path),
            "-t", str(duration),    # Duration
            "-vf", f"fps={self.fps},scale={self.img_size}:-1",
            "-q:v", "2",            # High quality JPEG
            "-threads", "2",
            str(output_dir / f"{video_id}_%06d.jpg")
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                frames = list(output_dir.glob(f"{video_id}_*.jpg"))
                return len(frames)
            else:
                logger.error(f"FFmpeg error: {result.stderr[:200]}")
                return 0
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return 0
    
    def extract_frames_decord(
        self,
        video_path: Path,
        output_dir: Path,
        start_sec: int = AVA_ANNOTATION_START,
        duration: int = AVA_DURATION_SEC
    ) -> int:
        """Extract frames using Decord (faster)"""
        from decord import VideoReader, cpu
        
        video_id = video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already extracted
        existing_frames = list(output_dir.glob(f"{video_id}_*.jpg"))
        if len(existing_frames) >= 800:
            logger.info(f"Frames already exist for {video_id}, skipping")
            return len(existing_frames)
        
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            
            # Calculate frame indices for the annotation window
            video_fps = vr.get_avg_fps()
            start_frame = int(start_sec * video_fps)
            end_frame = start_frame + int(duration * video_fps)
            total_frames = len(vr)
            
            # Clamp to available frames
            start_frame = min(start_frame, total_frames - 1)
            end_frame = min(end_frame, total_frames)
            
            # Get frames at specified FPS
            frame_indices = np.arange(start_frame, end_frame, video_fps / self.fps, dtype=int)
            frame_indices = frame_indices[:duration * self.fps]  # Limit to expected frames
            
            frames = vr.get_batch(frame_indices).asnumpy()
            
            # Save frames (convert RGB -> BGR for OpenCV)
            for i, frame in enumerate(frames):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_resized = cv2.resize(frame_bgr, (self.img_size, self.img_size))
                frame_path = output_dir / f"{video_id}_{i:06d}.jpg"
                cv2.imwrite(str(frame_path), frame_resized)
            
            return len(frames)
            
        except Exception as e:
            logger.error(f"Error extracting frames with Decord: {e}")
            return 0
    
    def extract_frames(self, video_path: Path, output_dir: Path) -> int:
        """Extract frames from a single video"""
        
        video_id = video_path.stem
        
        # Verify video duration first
        is_valid, duration = self.verify_video_duration(video_path)
        
        if not is_valid:
            logger.warning(f"Video {video_id} duration {duration}s < 1800s, may be incomplete")
            # Still try to extract what's available
        
        # Choose extraction method
        if self.decord_available:
            return self.extract_frames_decord(video_path, output_dir)
        else:
            return self.extract_frames_ffmpeg(video_path, output_dir)
    
    def extract_all_frames(
        self,
        train_split: float = 0.8,
        max_videos: int = None
    ) -> Dict:
        """Extract frames from all videos with train/val split"""
        
        # Get video files
        video_files = self.get_video_files()
        
        if not video_files:
            logger.error("No video files found")
            return {"train": 0, "val": 0, "total": 0}
        
        logger.info(f"Found {len(video_files)} videos")
        
        # Shuffle and split
        random.seed(42)
        random.shuffle(video_files)
        
        if max_videos:
            video_files = video_files[:max_videos]
        
        split_idx = int(len(video_files) * train_split)
        train_videos = video_files[:split_idx]
        val_videos = video_files[split_idx:]
        
        logger.info(f"Training videos: {len(train_videos)}")
        logger.info(f"Validation videos: {len(val_videos)}")
        
        # Extract training frames
        train_output = self.frame_dir / "train"
        train_count = 0
        
        logger.info("Extracting training frames...")
        for i, video_file in enumerate(train_videos):
            logger.info(f"Processing train video {i+1}/{len(train_videos)}: {video_file.name}")
            count = self.extract_frames(video_file, train_output)
            train_count += count
        
        # Extract validation frames
        val_output = self.frame_dir / "val"
        val_count = 0
        
        logger.info("Extracting validation frames...")
        for i, video_file in enumerate(val_videos):
            logger.info(f"Processing val video {i+1}/{len(val_videos)}: {video_file.name}")
            count = self.extract_frames(video_file, val_output)
            val_count += count
        
        logger.info("Frame extraction complete!")
        
        return {
            "train_frames": train_count,
            "val_frames": val_count,
            "total_frames": train_count + val_count,
        }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract frames from AVA videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all frames
  python scripts/extract_frames.py
  
  # Extract with custom FPS
  python scripts/extract_frames.py --fps 30
  
  # Extract from limited videos for testing
  python scripts/extract_frames.py --max-videos 10
        """
    )
    
    parser.add_argument(
        '--video-dir',
        type=str,
        default='data/raw_videos',
        help='Directory containing videos'
    )
    
    parser.add_argument(
        '--frame-dir',
        type=str,
        default='data/frames',
        help='Directory to save extracted frames'
    )
    
    parser.add_argument(
        '--annotation-dir',
        type=str,
        default='data/annotations',
        help='Directory containing annotations'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Target FPS for frame extraction'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=224,
        help='Output image size'
    )
    
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.8,
        help='Training/validation split ratio'
    )
    
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Maximum number of videos to process'
    )
    
    parser.add_argument(
        '--classes', '-c',
        type=str,
        help='Comma-separated list of action classes to filter'
    )
    
    parser.add_argument(
        '--no-decord',
        action='store_true',
        help='Disable Decord (use ffmpeg instead)'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Parse classes
    selected_classes = None
    if args.classes:
        selected_classes = [c.strip() for c in args.classes.split(',')]
        logger.info(f"Selected classes: {selected_classes}")
    
    # Initialize extractor
    extractor = FrameExtractor(
        video_dir=args.video_dir,
        frame_dir=args.frame_dir,
        annotation_dir=args.annotation_dir,
        fps=args.fps,
        img_size=args.img_size,
        selected_classes=selected_classes,
        use_decord=not args.no_decord
    )
    
    # Extract frames
    logger.info("Starting frame extraction...")
    logger.info(f"AVA temporal window: {AVA_ANNOTATION_START}-{AVA_ANNOTATION_END} seconds")
    
    results = extractor.extract_all_frames(
        train_split=args.split_ratio,
        max_videos=args.max_videos
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("FRAME EXTRACTION RESULTS")
    print("=" * 50)
    print(f"Training frames:     {results['train_frames']}")
    print(f"Validation frames:   {results['val_frames']}")
    print(f"Total frames:        {results['total_frames']}")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
