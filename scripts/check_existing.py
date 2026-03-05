#!/usr/bin/env python3
"""
Check Existing Videos - Duplicate Avoidance Utility

Checks which videos already exist in the video directory and compares
with AVA annotations to determine what needs to be downloaded.

Usage:
    python scripts/check_existing.py
    python scripts/check_existing.py --video-dir data/raw_videos
    python scripts/check_existing.py --annotation-dir data/annotations
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Set, Dict, List, Tuple
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.ava_classes import AVA_CLASSES, get_class_ids_from_names, ANNOTATION_FILES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoChecker:
    """Check for existing videos and analyze coverage"""
    
    def __init__(
        self,
        video_dir: str = "data/raw_videos",
        annotation_dir: str = "data/annotations"
    ):
        self.video_dir = Path(video_dir)
        self.annotation_dir = Path(annotation_dir)
        
        # Ensure directories exist
        if not self.video_dir.exists():
            logger.warning(f"Video directory does not exist: {self.video_dir}")
            self.video_dir.mkdir(parents=True, exist_ok=True)
    
    def get_existing_videos(self) -> Set[str]:
        """Get set of existing video IDs"""
        existing = set()
        
        if not self.video_dir.exists():
            return existing
        
        # Common video extensions
        video_extensions = {'.mp4', '.webm', '.avi', '.mkv', '.mov', '.flv', '.wmv'}
        
        for video_file in self.video_dir.iterdir():
            if video_file.suffix.lower() in video_extensions:
                # Video ID is the filename without extension
                video_id = video_file.stem
                existing.add(video_id)
        
        return existing
    
    def get_video_files_info(self) -> List[Dict]:
        """Get detailed info about existing video files"""
        info = []
        
        if not self.video_dir.exists():
            return info
        
        video_extensions = {'.mp4', '.webm', '.avi', '.mkv', '.mov', '.flv', '.wmv'}
        
        for video_file in self.video_dir.iterdir():
            if video_file.suffix.lower() in video_extensions:
                stat = video_file.stat()
                info.append({
                    "video_id": video_file.stem,
                    "filename": video_file.name,
                    "size_mb": stat.st_size / (1024 * 1024),
                    "extension": video_file.suffix,
                })
        
        return sorted(info, key=lambda x: x["size_mb"], reverse=True)
    
    def load_annotations(self, selected_classes: List[str] = None) -> Dict:
        """Load AVA annotations"""
        train_file = self.annotation_dir / ANNOTATION_FILES["train"]
        
        if not train_file.exists():
            logger.error(f"Training annotations not found: {train_file}")
            return {"videos": set(), "by_class": defaultdict(set), "total": 0}
        
        # Filter by selected classes if provided
        filter_class_ids = set()
        if selected_classes:
            filter_class_ids = set(get_class_ids_from_names(selected_classes))
        
        videos = set()
        by_class = defaultdict(set)
        
        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    video_id = parts[0]
                    action_id = int(parts[3])
                    
                    # Filter by class if specified
                    if filter_class_ids and action_id not in filter_class_ids:
                        continue
                    
                    videos.add(video_id)
                    by_class[action_id].add(video_id)
        
        return {
            "videos": videos,
            "by_class": by_class,
            "total": len(videos)
        }
    
    def analyze_coverage(
        self,
        annotations: Dict,
        selected_classes: List[str] = None
    ) -> Dict:
        """Analyze coverage of existing videos"""
        existing = self.get_existing_videos()
        required_videos = annotations["videos"]
        
        # Calculate coverage
        covered = existing & required_videos
        missing = required_videos - existing
        
        result = {
            "existing_count": len(existing),
            "required_count": len(required_videos),
            "covered_count": len(covered),
            "missing_count": len(missing),
            "coverage_percent": (len(covered) / len(required_videos) * 100) if required_videos else 0,
            "existing_videos": existing,
            "covered_videos": covered,
            "missing_videos": missing,
        }
        
        # By class analysis
        if selected_classes:
            filter_class_ids = set(get_class_ids_from_names(selected_classes))
            by_class_coverage = {}
            
            for class_id in filter_class_ids:
                class_videos = annotations["by_class"].get(class_id, set())
                class_covered = class_videos & existing
                class_missing = class_videos - existing
                
                by_class_coverage[class_id] = {
                    "name": AVA_CLASSES.get(class_id, "unknown"),
                    "total": len(class_videos),
                    "covered": len(class_covered),
                    "missing": len(class_missing),
                    "coverage_percent": (len(class_covered) / len(class_videos) * 100) if class_videos else 0,
                }
            
            result["by_class"] = by_class_coverage
        
        return result
    
    def generate_download_list(
        self,
        annotations: Dict,
        selected_classes: List[str] = None,
        priority_classes: List[str] = None
    ) -> List[Tuple[str, int]]:
        """Generate prioritized list of videos to download"""
        existing = self.get_existing_videos()
        required = annotations["videos"]
        
        # Get videos not yet downloaded
        to_download = required - existing
        
        # If priority classes specified, prioritize those
        if priority_classes:
            priority_class_ids = set(get_class_ids_from_names(priority_classes))
            
            # Priority: videos with priority classes first
            priority_videos = []
            other_videos = []
            
            for video_id in to_download:
                # Check if this video has annotations in priority classes
                is_priority = False
                for ann in annotations.get("by_video", {}).get(video_id, []):
                    if ann in priority_class_ids:
                        is_priority = True
                        break
                
                if is_priority:
                    priority_videos.append(video_id)
                else:
                    other_videos.append(video_id)
            
            return [(vid, 1) for vid in priority_videos] + [(vid, 2) for vid in other_videos]
        
        return [(vid, 1) for vid in sorted(to_download)]
    
    def print_report(self, coverage: Dict, selected_classes: List[str] = None):
        """Print coverage report"""
        print("\n" + "=" * 60)
        print("VIDEO COVERAGE REPORT")
        print("=" * 60)
        
        print(f"\nOverall Statistics:")
        print(f"  Existing videos:    {coverage['existing_count']}")
        print(f"  Required videos:    {coverage['required_count']}")
        print(f"  Covered videos:     {coverage['covered_count']}")
        print(f"  Missing videos:    {coverage['missing_count']}")
        print(f"  Coverage:          {coverage['coverage_percent']:.1f}%")
        
        if coverage['missing_count'] > 0:
            print(f"\nVideos to download ({coverage['missing_count']}):")
            for vid in sorted(coverage['missing_videos'])[:20]:
                print(f"  - {vid}")
            if coverage['missing_count'] > 20:
                print(f"  ... and {coverage['missing_count'] - 20} more")
        
        # By class breakdown
        if selected_classes and 'by_class' in coverage:
            print(f"\nCoverage by Class:")
            print("-" * 60)
            
            for class_id, class_info in sorted(coverage['by_class'].items()):
                name = class_info['name'][:30]
                total = class_info['total']
                covered = class_info['covered']
                pct = class_info['coverage_percent']
                
                print(f"  {class_id:2d}. {name:30s} {covered:4d}/{total:4d} ({pct:5.1f}%)")
        
        print("\n" + "=" * 60)
    
    def find_duplicate_files(self) -> List[Tuple[Path, Path]]:
        """Find potential duplicate files based on name"""
        duplicates = []
        files_by_name = defaultdict(list)
        
        if not self.video_dir.exists():
            return duplicates
        
        for video_file in self.video_dir.iterdir():
            name = video_file.stem.lower()
            files_by_name[name].append(video_file)
        
        for name, files in files_by_name.items():
            if len(files) > 1:
                # Keep the first, mark others as duplicates
                for f in files[1:]:
                    duplicates.append((files[0], f))
        
        return duplicates


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Check existing videos and analyze coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check existing videos
  python scripts/check_existing.py
  
  # Check with specific classes
  python scripts/check_existing.py --classes kiss,hug,walk,run
  
  # Generate download list
  python scripts/check_existing.py --generate-list
  
  # Show detailed file info
  python scripts/check_existing.py --show-files
        """
    )
    
    parser.add_argument(
        '--video-dir',
        type=str,
        default='data/raw_videos',
        help='Directory containing videos'
    )
    
    parser.add_argument(
        '--annotation-dir',
        type=str,
        default='data/annotations',
        help='Directory containing annotations'
    )
    
    parser.add_argument(
        '--classes', '-c',
        type=str,
        help='Comma-separated list of action classes'
    )
    
    parser.add_argument(
        '--priority-classes', '-p',
        type=str,
        help='Priority classes for download'
    )
    
    parser.add_argument(
        '--generate-list',
        action='store_true',
        help='Generate download list for missing videos'
    )
    
    parser.add_argument(
        '--show-files',
        action='store_true',
        help='Show detailed file information'
    )
    
    parser.add_argument(
        '--find-duplicates',
        action='store_true',
        help='Find duplicate video files'
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
    
    priority_classes = None
    if args.priority_classes:
        priority_classes = [c.strip() for c in args.priority_classes.split(',')]
    
    # Initialize checker
    checker = VideoChecker(
        video_dir=args.video_dir,
        annotation_dir=args.annotation_dir
    )
    
    # Show file info if requested
    if args.show_files:
        print("\nExisting Video Files:")
        print("-" * 60)
        
        files_info = checker.get_video_files_info()
        if not files_info:
            print("  No video files found")
        else:
            total_size = 0
            for info in files_info[:20]:
                print(f"  {info['video_id']:15s} {info['size_mb']:8.2f} MB  {info['extension']}")
                total_size += info['size_mb']
            
            print(f"\n  Total files: {len(files_info)}")
            print(f"  Total size: {total_size:.2f} MB")
            
            if len(files_info) > 20:
                remaining = sum(f['size_mb'] for f in files_info[20:])
                print(f"  Remaining size: {remaining:.2f} MB")
        
        return 0
    
    # Find duplicates if requested
    if args.find_duplicates:
        print("\nFinding duplicate files...")
        duplicates = checker.find_duplicate_files()
        
        if duplicates:
            print(f"\nFound {len(duplicates)} potential duplicates:")
            for orig, dup in duplicates:
                print(f"  {orig.name} <-> {dup.name}")
        else:
            print("No duplicates found")
        
        return 0
    
    # Load annotations
    logger.info("Loading annotations...")
    annotations = checker.load_annotations(selected_classes)
    
    if annotations['total'] == 0:
        logger.error("No annotations found. Please download annotations first.")
        return 1
    
    # Analyze coverage
    logger.info("Analyzing coverage...")
    coverage = checker.analyze_coverage(annotations, selected_classes)
    
    # Print report
    checker.print_report(coverage, selected_classes)
    
    # Generate download list if requested
    if args.generate_list and coverage['missing_count'] > 0:
        print("\nGenerating download list...")
        
        download_list = checker.generate_download_list(
            annotations,
            selected_classes,
            priority_classes
        )
        
        print(f"\nDownload list ({len(download_list)} videos):")
        print("-" * 40)
        
        priority_count = sum(1 for _, p in download_list if p == 1)
        print(f"Priority videos: {priority_count}")
        print(f"Other videos: {len(download_list) - priority_count}")
        
        # Save to file
        output_file = "data/videos_to_download.txt"
        with open(output_file, 'w') as f:
            for vid, priority in download_list:
                f.write(f"{vid}\n")
        
        print(f"\nSaved download list to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
