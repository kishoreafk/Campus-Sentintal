#!/usr/bin/env python3
"""
AVA Video Downloader with Class Filtering

Downloads videos from AVA dataset official sources with support for:
- Class/action filtering (only download videos containing selected actions)
- Duplicate avoidance (skip existing videos)
- Multiple download sources (CVDF mirror, Google Cloud Storage, etc.)

IMPORTANT: This script downloads from official AVA dataset mirrors, NOT YouTube.
The AVA dataset videos are hosted on the CVDF mirror and other official sources.

Usage:
    python scripts/download_videos.py --classes kiss,hug,walk,run
    python scripts/download_videos.py --classes kiss,hug,walk,run,sit,jump,fight
    python scripts/download_videos.py --all  # Download all classes
"""

import os
import sys
import argparse
import logging
import subprocess
import hashlib
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional
import urllib.request
import zipfile
import shutil
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.ava_classes import (
    AVA_CLASSES, COMMON_CLASSES, get_class_ids_from_names,
    ANNOTATION_FILES, AVA_ANNOTATIONS_URL
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Official AVA Video Download Sources (NOT YouTube)
# Verified working pattern: https://s3.amazonaws.com/ava-dataset/trainval/<video_id>.mkv
AVA_VIDEO_SOURCES = {
    # CVDF / official S3 mirror - Primary source (trainval/ prefix, .mkv format)
    "cvdf_base": "https://s3.amazonaws.com/ava-dataset/",
    "aws_base": "https://ava-dataset.s3.amazonaws.com/",
    # Fallback mirrors
    "gcs_base": "https://storage.googleapis.com/ava-dataset/",
}

ANNOTATION_MIRRORS = [
    "https://s3.amazonaws.com/ava-dataset/annotations",
    "https://ava-dataset.s3.amazonaws.com/annotations",
    "https://storage.googleapis.com/ava-dataset/annotations",
]

# AVA video file naming convention
# Videos are named by their YouTube ID but downloaded from official sources
# The actual videos are available in the AVA bucket


class AVAVideoDownloader:
    """Downloader for AVA dataset videos with class filtering"""
    
    def __init__(
        self,
        video_dir: str = "data/raw_videos",
        annotation_dir: str = "data/annotations",
        selected_classes: Optional[List[str]] = None,
        quality: str = "best[height<=720]",
        skip_existing: bool = True,
        download_source: str = "cvdf"  # cvdf, aws, gcs, github, original, all
    ):
        self.video_dir = Path(video_dir)
        self.annotation_dir = Path(annotation_dir)
        self.selected_classes = selected_classes
        self.quality = quality
        self.skip_existing = skip_existing
        self.download_source = download_source
        
        # Create directories
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        
        # Class ID to filter by
        self.filter_class_ids: Set[int] = set()
        if selected_classes:
            self.filter_class_ids = set(get_class_ids_from_names(selected_classes))
            logger.info(f"Filtering for classes: {self.filter_class_ids}")
        
        # Download statistics
        self.download_stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "existing": 0
        }
    
    def download_annotations(self, force: bool = False) -> Tuple[Path, Path]:
        """Download AVA annotations if not present"""
        train_file = self.annotation_dir / ANNOTATION_FILES["train"]
        val_file = self.annotation_dir / ANNOTATION_FILES["val"]
        
        if train_file.exists() and val_file.exists() and not force:
            logger.info("Annotations already exist, skipping download")
            return train_file, val_file
        
        logger.info("Downloading AVA annotations...")
        
        # Download annotation files
        for filename in ["ava_train_v2.2.csv", "ava_val_v2.2.csv"]:
            dest = self.annotation_dir / filename

            downloaded = False
            for base_url in [AVA_ANNOTATIONS_URL, *ANNOTATION_MIRRORS]:
                url = f"{base_url.rstrip('/')}/{filename}"
                try:
                    logger.info(f"Downloading {filename} from {url}...")
                    request = urllib.request.Request(url)
                    request.add_header('User-Agent', 'Mozilla/5.0')
                    with urllib.request.urlopen(request, timeout=30) as response:
                        with open(dest, 'wb') as f:
                            f.write(response.read())
                    logger.info(f"Downloaded {filename}")
                    downloaded = True
                    break
                except Exception as e:
                    logger.warning(f"Could not download {filename} from {url}: {e}")

            if not downloaded:
                self._create_sample_annotations(dest, filename)
        
        return train_file, val_file
    
    def _create_sample_annotations(self, dest: Path, filename: str):
        """Create a sample annotation file if download fails"""
        logger.warning(f"Creating sample annotation file: {dest}")
        
        # Sample format for AVA annotations:
        # video_id, timestamp, person_id, action_id, action_label, x1, y1, x2, y2
        sample_data = """-5KQ66BRF-w,0900,1,30,kiss,0.342,0.356,0.472,0.726
-5KQ66BRF-w,0900,2,30,kiss,0.572,0.398,0.692,0.768
3e9x--0KILw,0300,1,27,hug (a person),0.312,0.421,0.478,0.741
3e9x--0KILw,0300,2,27,hug (a person),0.589,0.398,0.721,0.689
"""
        dest.write_text(sample_data)
    
    def load_annotations(self) -> Dict[str, List[Dict]]:
        """Load AVA annotations and filter by selected classes"""
        train_file = self.annotation_dir / ANNOTATION_FILES["train"]
        
        if not train_file.exists():
            logger.error(f"Training annotations not found: {train_file}")
            return {"train": [], "val": []}
        
        annotations = {"train": [], "val": []}
        
        # Parse annotations
        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    video_id = parts[0]
                    timestamp = parts[1]
                    # AVA CSV: video_id(0), ts(1), x1(2), y1(3), x2(4), y2(5), action_id(6), person_id(7)
                    try:
                        action_id = int(parts[6])
                    except ValueError:
                        continue
                    
                    # Filter by class if specified
                    if self.filter_class_ids and action_id not in self.filter_class_ids:
                        continue
                    
                    annotations["train"].append({
                        "video_id": video_id,
                        "timestamp": timestamp,
                        "person_id": parts[7] if len(parts) > 7 else "",
                        "action_id": action_id,
                        "action_name": AVA_CLASSES.get(action_id, "unknown")
                    })
        
        logger.info(f"Loaded {len(annotations['train'])} annotations")
        return annotations
    
    def get_unique_videos(self, annotations: Dict) -> Set[str]:
        """Get unique video IDs from annotations"""
        video_ids = set()
        for ann in annotations["train"]:
            video_ids.add(ann["video_id"])
        return video_ids
    
    def check_existing_videos(self) -> Set[str]:
        """Check which videos already exist in the video directory"""
        existing = set()
        
        if not self.video_dir.exists():
            return existing
        
        for video_file in self.video_dir.iterdir():
            if video_file.suffix.lower() in ['.mp4', '.webm', '.avi', '.mkv', '.mov']:
                # Video ID is the filename without extension
                video_id = video_file.stem
                existing.add(video_id)
        
        logger.info(f"Found {len(existing)} existing videos")
        return existing
    
    def get_videos_to_download(self, annotations: Dict) -> Set[str]:
        """Get list of videos that need to be downloaded"""
        all_videos = self.get_unique_videos(annotations)
        existing = self.check_existing_videos()
        
        if self.skip_existing:
            videos_to_download = all_videos - existing
        else:
            videos_to_download = all_videos
        
        logger.info(f"Videos to download: {len(videos_to_download)}")
        logger.info(f"Existing videos: {len(existing)}")
        
        return videos_to_download
    
    def get_video_url(self, video_id: str) -> Optional[str]:
        """Get the official AVA video URL for a given video ID.

        Verified working pattern:
            https://s3.amazonaws.com/ava-dataset/trainval/<video_id>.mkv

        Videos are predominantly .mkv under the trainval/ prefix on S3.
        We try .mkv first, then .mp4 as a fallback, across all mirrors.
        """
        # Build candidate list – .mkv under trainval/ is the confirmed pattern.
        sources_to_try: List[str] = []

        if self.download_source in ("cvdf", "all"):
            for ext in ("mkv", "mp4"):
                sources_to_try.append(
                    f"{AVA_VIDEO_SOURCES['cvdf_base']}trainval/{video_id}.{ext}"
                )

        if self.download_source in ("aws", "all"):
            for ext in ("mkv", "mp4"):
                sources_to_try.append(
                    f"{AVA_VIDEO_SOURCES['aws_base']}trainval/{video_id}.{ext}"
                )

        if self.download_source in ("gcs", "all"):
            for ext in ("mkv", "mp4"):
                sources_to_try.append(
                    f"{AVA_VIDEO_SOURCES['gcs_base']}trainval/{video_id}.{ext}"
                )

        if self.download_source == "original":
            sources_to_try = self._get_original_video_url(video_id)

        # Default: if user picked 'cvdf' only, still include AWS alias
        if self.download_source == "cvdf":
            for ext in ("mkv", "mp4"):
                sources_to_try.append(
                    f"{AVA_VIDEO_SOURCES['aws_base']}trainval/{video_id}.{ext}"
                )

        # Check which source is accessible
        for url in sources_to_try:
            if self._check_url_exists(url):
                return url

        return None
    
    def _get_original_video_url(self, video_id: str) -> List[str]:
        """Get video URLs from original sources (movie/TV clips)"""
        # AVA videos come from various movies and TV shows
        # This is a placeholder - actual URLs would require the original source mapping
        return []
    
    def _check_url_exists(self, url: str) -> bool:
        """Check if a URL is accessible"""
        try:
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'Mozilla/5.0')
            request.method = 'HEAD'
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status == 200
        except Exception:
            return False
    
    def download_video(self, video_id: str, max_retries: int = 3) -> bool:
        """Download a single video from official AVA sources"""
        
        # Check if video already exists
        if self.skip_existing:
            for ext in ['.mkv', '.mp4', '.webm', '.avi', '.mov']:
                if (self.video_dir / f"{video_id}{ext}").exists():
                    logger.info(f"Video {video_id} already exists, skipping")
                    self.download_stats["existing"] += 1
                    return True
        
        # Get video URL from official sources
        video_url = self.get_video_url(video_id)
        
        if video_url is None:
            logger.warning(f"No official source found for video {video_id}")
            logger.info(f"Video {video_id} may need manual download or use --source original")
            self.download_stats["failed"] += 1
            return False
        
        # Download the video
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {video_id} (attempt {attempt + 1}/{max_retries})")
                logger.info(f"Source: {video_url}")
                
                # Derive file extension from the source URL
                url_ext = Path(video_url.split("?")[0]).suffix or ".mkv"
                output_path = self.video_dir / f"{video_id}{url_ext}"
                
                # Download with progress
                self._download_with_progress(video_url, output_path)
                
                logger.info(f"Successfully downloaded {video_id}")
                self.download_stats["success"] += 1
                return True
                
            except Exception as e:
                logger.error(f"Error downloading {video_id}: {e}")
                if attempt < max_retries - 1:
                    logger.info("Retrying...")
                    time.sleep(2)
        
        logger.error(f"Failed to download {video_id} after {max_retries} attempts")
        self.download_stats["failed"] += 1
        return False
    
    def _download_with_progress(self, url: str, output_path: Path):
        """Download file with progress indication"""
        
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            if block_num % 100 == 0:
                logger.info(f"Downloaded: {percent:.1f}%")
        
        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'Mozilla/5.0')
        
        with urllib.request.urlopen(request, timeout=60) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            
            with open(output_path, 'wb') as f:
                shutil.copyfileobj(response, f, length=1024*1024)
    
    def download_batch(
        self,
        video_ids: List[str],
        delay_between: float = 1.0
    ) -> Dict[str, bool]:
        """Download multiple videos"""
        
        results = {}
        total = len(video_ids)
        self.download_stats["total"] = total
        
        for i, video_id in enumerate(video_ids):
            logger.info(f"Processing video {i + 1}/{total}: {video_id}")
            
            success = self.download_video(video_id)
            results[video_id] = success
            
            # Rate limiting
            if i < total - 1:
                time.sleep(delay_between)
        
        # Summary
        logger.info(f"Download complete: {self.download_stats['success']}/{total} successful")
        
        return results
    
    def print_download_summary(self):
        """Print download statistics"""
        print("\n" + "=" * 50)
        print("DOWNLOAD SUMMARY")
        print("=" * 50)
        print(f"Total videos:     {self.download_stats['total']}")
        print(f"Downloaded:      {self.download_stats['success']}")
        print(f"Failed:          {self.download_stats['failed']}")
        print(f"Skipped:         {self.download_stats['skipped']}")
        print(f"Already exists:  {self.download_stats['existing']}")
        print("=" * 50)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Download AVA dataset videos from official sources (NOT YouTube)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download videos for specific classes from the CVDF mirror
  python scripts/download_videos.py --classes kiss,hug,walk,run
  
  # Force Google Cloud fallback
  python scripts/download_videos.py --classes kiss,hug,walk,run --source gcs
  
  # Check existing videos without downloading
  python scripts/download_videos.py --check-only
  
  # Download all classes
  python scripts/download_videos.py --all

NOTE: AVA videos are downloaded from official sources (CVDF, AWS, Google Cloud Storage),
NOT from YouTube. Some videos may not be available from mirrors and may need
to be obtained from the original movie/TV sources.
        """
    )
    
    parser.add_argument(
        '--classes', '-c',
        type=str,
        help='Comma-separated list of action classes to filter'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all classes (no filtering)'
    )
    
    parser.add_argument(
        '--video-dir',
        type=str,
        default='data/raw_videos',
        help='Directory to save videos'
    )
    
    parser.add_argument(
        '--annotation-dir',
        type=str,
        default='data/annotations',
        help='Directory containing annotations'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='cvdf',
        choices=['cvdf', 'gcs', 'aws', 'original', 'all'],
        help='Video download source (cvdf=primary mirror, gcs=Google Cloud Storage, aws=AWS mirror, original=direct sources)'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip videos that already exist'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check existing videos without downloading'
    )
    
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Maximum number of videos to download'
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
        logger.info(f"Selected classes: {selected_classes}")
    elif not args.all:
        logger.error("Please specify --classes or --all")
        return 1
    
    # Initialize downloader
    downloader = AVAVideoDownloader(
        video_dir=args.video_dir,
        annotation_dir=args.annotation_dir,
        selected_classes=selected_classes,
        skip_existing=args.skip_existing,
        download_source=args.source
    )
    
    # Check existing videos
    if args.check_only:
        existing = downloader.check_existing_videos()
        print(f"\nExisting videos: {len(existing)}")
        for vid in sorted(existing):
            print(f"  - {vid}")
        return 0
    
    # Download annotations
    logger.info("Step 1: Downloading annotations...")
    downloader.download_annotations()
    
    # Load and filter annotations
    logger.info("Step 2: Loading annotations and filtering by class...")
    annotations = downloader.load_annotations()
    
    # Get videos to download
    videos_to_download = downloader.get_videos_to_download(annotations)
    video_ids = sorted(videos_to_download)
    
    # Limit number of videos if requested
    if args.max_videos and len(video_ids) > args.max_videos:
        logger.info(f"Limiting to {args.max_videos} videos")
        video_ids = video_ids[:args.max_videos]
    
    if not video_ids:
        logger.info("No videos to download!")
        return 0
    
    # Download videos
    logger.info(f"Step 3: Downloading {len(video_ids)} videos from official AVA sources...")
    results = downloader.download_batch(video_ids)
    
    # Print summary
    downloader.print_download_summary()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
