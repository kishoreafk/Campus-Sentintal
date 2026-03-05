#!/usr/bin/env python3
"""
AVA Video Download Orchestrator

Multi-source video download with priority:
1. CVDF / Facebook Mirror (pre-cut 15min clips) - Primary
2. Hugging Face / Academic Mirrors
3. YouTube direct download (with trimming to AVA's 15-minute segments)

This script tries each source in priority order for every video.
"""

import os
import sys
import argparse
import logging
import subprocess
import json
import time
import hashlib
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.ava_classes import AVA_CLASSES, get_class_ids_from_names, ANNOTATION_FILES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Download sources
AVAILABLE_SOURCES = {
    "cvdf": "https://s3.amazonaws.com/ava-dataset/trainval",
    "huggingface": "https://huggingface.co/datasets/Loie/AVA-dataset/resolve/main/videos/",
    "youtube": "https://www.youtube.com/watch?v=",
}


class AVADownloadOrchestrator:
    """Multi-source video downloader with fallback"""
    
    def __init__(
        self,
        video_dir: str = "data/raw_videos",
        annotation_dir: str = "data/annotations",
        selected_classes: Optional[List[str]] = None,
        max_workers: int = 8,
        skip_existing: bool = True
    ):
        self.video_dir = Path(video_dir)
        self.annotation_dir = Path(annotation_dir)
        self.selected_classes = selected_classes
        self.max_workers = max_workers
        self.skip_existing = skip_existing
        
        # Create directories
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        # Status tracking
        self.status_file = self.video_dir / "download_status.json"
        self.status = self._load_status()
        
        # Class filter
        self.filter_class_ids = set()
        if selected_classes:
            self.filter_class_ids = set(get_class_ids_from_names(selected_classes))
    
    def _load_status(self) -> Dict:
        """Load previous download status"""
        if self.status_file.exists():
            with open(self.status_file) as f:
                return json.load(f)
        return {}
    
    def _save_status(self):
        """Save download status"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def download_annotations(self):
        """Download AVA annotations"""
        logger.info("Downloading AVA annotations...")
        
        # Try official Google source first
        google_base = "https://research.google.com/ava/download"
        
        # Also try FAIR mirror
        fair_base = "https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations"
        
        files_to_download = [
            ("ava_train_v2.2.csv", google_base),
            ("ava_val_v2.2.csv", google_base),
            ("ava_action_list_v2.2_for_activitynet_2019.pbtxt", google_base),
            ("ava_train_excluded_timestamps_v2.2.csv", google_base),
            ("ava_val_excluded_timestamps_v2.2.csv", google_base),
            ("ava_file_names_trainval_v2.1.txt", google_base),
        ]
        
        for filename, base_url in files_to_download:
            dest = self.annotation_dir / filename
            if dest.exists():
                continue
                
            urls = [
                f"{base_url}/{filename}",
                f"{fair_base}/{filename}",
            ]
            
            for url in urls:
                try:
                    logger.info(f"Downloading {filename} from {url}")
                    result = subprocess.run(
                        ["curl", "-L", "-o", str(dest), url],
                        capture_output=True, timeout=60
                    )
                    if result.returncode == 0 and dest.exists():
                        logger.info(f"Downloaded {filename}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to download {filename}: {e}")
        
        # Download proposal files
        proposal_urls = [
            ("https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl",
             "data/ava/proposals/"),
            ("https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl",
             "data/ava/proposals/"),
        ]
        
        for url, dest_dir in proposal_urls:
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
            filename = Path(url).name
            dest = Path(dest_dir) / filename
            if dest.exists():
                continue
            try:
                subprocess.run(["curl", "-L", "-o", str(dest), url], timeout=300)
            except Exception as e:
                logger.warning(f"Failed to download proposal: {e}")
    
    def get_video_list(self) -> List[str]:
        """Get list of video IDs from annotations"""
        list_file = self.annotation_dir / "ava_file_names_trainval_v2.1.txt"
        
        if not list_file.exists():
            logger.error("Video list file not found")
            return []
        
        with open(list_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
        
        # Filter by selected classes if specified
        if self.filter_class_ids:
            train_file = self.annotation_dir / "ava_train_v2.2.csv"
            if train_file.exists():
                filtered = set()
                with open(train_file) as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 4:
                            video_id = parts[0]
                            action_id = int(parts[3])
                            if action_id in self.filter_class_ids:
                                filtered.add(video_id)
                video_ids = [v for v in video_ids if v in filtered]
        
        return video_ids
    
    def video_exists(self, video_id: str) -> bool:
        """Check if video already exists"""
        for ext in ['.mkv', '.mp4', '.webm', '.avi']:
            if (self.video_dir / f"{video_id}{ext}").exists():
                return True
        return False
    
    def try_cvdf_source(self, video_id: str) -> bool:
        """Try downloading from CVDF / Facebook S3 mirror"""
        base = AVAILABLE_SOURCES["cvdf"]
        
        # Try different extensions
        for ext in ['mkv', 'mp4', 'webm']:
            url = f"{base}/{video_id}.{ext}"
            output = self.video_dir / f"{video_id}.{ext}"
            
            try:
                # First check if URL exists
                result = subprocess.run(
                    ["curl", "-s", "-I", "-L", url],
                    capture_output=True, text=True, timeout=30
                )
                
                if "HTTP" in result.stdout and "200" in result.stdout:
                    # Download with aria2c for faster speed
                    subprocess.run([
                        "aria2c", "-x", "16", "-s", "16", "-k", "1M",
                        "-d", str(self.video_dir),
                        "-o", f"{video_id}.{ext}",
                        url
                    ], timeout=600)
                    
                    if output.exists():
                        return True
            except Exception as e:
                logger.debug(f"CVDF failed for {video_id}: {e}")
                continue
        
        return False
    
    def try_huggingface_source(self, video_id: str) -> bool:
        """Try downloading from Hugging Face mirror"""
        base = AVAILABLE_SOURCES["huggingface"]
        
        for ext in ['mkv', 'mp4']:
            url = f"{base}{video_id}.{ext}"
            output = self.video_dir / f"{video_id}.{ext}"
            
            try:
                result = subprocess.run(
                    ["curl", "-s", "-I", "-L", url],
                    capture_output=True, text=True, timeout=30
                )
                
                if "HTTP" in result.stdout and "200" in result.stdout:
                    subprocess.run([
                        "curl", "-L", "-o", str(output), url
                    ], timeout=600)
                    
                    if output.exists():
                        return True
            except Exception as e:
                logger.debug(f"HuggingFace failed for {video_id}: {e}")
                continue
        
        return False
    
    def try_youtube_source(self, video_id: str, start_sec: int = 900, duration: int = 901) -> bool:
        """Try downloading from YouTube and trimming to AVA's 15-minute segment
        
        AVA uses seconds 902-1798 of each video (15 minutes centered at midpoint)
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        temp_path = self.video_dir / f"_temp_{video_id}.mp4"
        output_path = self.video_dir / f"{video_id}.mp4"
        
        try:
            # Step 1: Download video
            cmd_download = [
                "yt-dlp",
                "-f", "best[height<=720][ext=mp4]/best[height<=720]",
                "--merge-output-format", "mp4",
                "-o", str(temp_path),
                "--retries", "3",
                "--socket-timeout", "30",
                url
            ]
            
            result = subprocess.run(
                cmd_download,
                capture_output=True, text=True, timeout=900
            )
            
            if result.returncode != 0 or not temp_path.exists():
                return False
            
            # Step 2: Trim to AVA temporal window
            cmd_trim = [
                "ffmpeg", "-y",
                "-ss", str(start_sec),
                "-i", str(temp_path),
                "-t", str(duration),
                "-c:v", "libx264", "-crf", "18",
                "-c:a", "aac",
                "-threads", "4",
                str(output_path)
            ]
            
            subprocess.run(cmd_trim, capture_output=True, timeout=300)
            
            # Cleanup temp file
            if temp_path.exists():
                temp_path.unlink()
            
            return output_path.exists()
        
        except Exception as e:
            logger.debug(f"YouTube failed for {video_id}: {e}")
            # Cleanup on failure
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def download_single(self, video_id: str) -> Tuple[str, str]:
        """Try all sources for a single video"""
        
        # Check if already exists
        if self.skip_existing and self.video_exists(video_id):
            self.status[video_id] = "exists"
            return video_id, "exists"
        
        # Skip if already successfully downloaded
        if video_id in self.status and self.status[video_id].startswith("ok:"):
            return video_id, self.status[video_id]
        
        sources = [
            ("cvdf", self.try_cvdf_source),
            ("huggingface", self.try_huggingface_source),
            ("youtube", self.try_youtube_source),
        ]
        
        for source_name, source_fn in sources:
            try:
                if source_fn(video_id):
                    if self.video_exists(video_id):
                        self.status[video_id] = f"ok:{source_name}"
                        return video_id, f"ok:{source_name}"
            except Exception as e:
                logger.debug(f"Source {source_name} error for {video_id}: {e}")
                continue
        
        self.status[video_id] = "all_failed"
        return video_id, "all_failed"
    
    def run(self, max_videos: Optional[int] = None):
        """Run the download orchestrator"""
        
        # Get video list
        video_ids = self.get_video_list()
        
        if not video_ids:
            logger.error("No videos to download")
            return
        
        # Filter to pending videos
        pending = [
            v for v in video_ids
            if self.status.get(v, "").split(":")[0] not in ("ok", "exists")
        ]
        
        if max_videos:
            pending = pending[:max_videos]
        
        logger.info(f"Total videos: {len(video_ids)}")
        logger.info(f"Pending downloads: {len(pending)}")
        
        # Download with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_single, v): v 
                for v in pending
            }
            
            for future in as_completed(futures):
                vid, status = future.result()
                logger.info(f"  {vid}: {status}")
                
                # Save progress periodically
                self._save_status()
        
        # Final summary
        ok_count = sum(1 for s in self.status.values() if "ok" in s or "exists" in s)
        logger.info(f"\nFinal: {ok_count}/{len(video_ids)} videos available")
        
        self._save_status()


def parse_args():
    parser = argparse.ArgumentParser(description="AVA Video Download Orchestrator")
    
    parser.add_argument('--video-dir', type=str, default='data/raw_videos')
    parser.add_argument('--annotation-dir', type=str, default='data/annotations')
    parser.add_argument('--classes', type=str, help='Comma-separated action classes')
    parser.add_argument('--max-workers', type=int, default=8)
    parser.add_argument('--max-videos', type=int, default=None)
    parser.add_argument('--skip-existing', action='store_true', default=True)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    selected_classes = None
    if args.classes:
        selected_classes = [c.strip() for c in args.classes.split(',')]
    
    orchestrator = AVADownloadOrchestrator(
        video_dir=args.video_dir,
        annotation_dir=args.annotation_dir,
        selected_classes=selected_classes,
        max_workers=args.max_workers,
        skip_existing=args.skip_existing
    )
    
    # Download annotations first
    orchestrator.download_annotations()
    
    # Run downloads
    orchestrator.run(max_videos=args.max_videos)


if __name__ == "__main__":
    main()
