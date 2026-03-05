#!/usr/bin/env python3
"""
AVA Dataset Validation Script

Comprehensive validation:
1. Check completeness (expected vs downloaded videos)
2. Check video integrity (not corrupted)
3. Detect duplicates
4. Check annotation-video alignment
"""

import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AVADatasetValidator:
    """Validate AVA dataset integrity"""
    
    def __init__(self, video_dir: str, annotation_dir: str):
        self.video_dir = Path(video_dir)
        self.annotation_dir = Path(annotation_dir)
        
        self.report = {
            "total_expected": 0,
            "total_found": 0,
            "missing": [],
            "corrupted": [],
            "duplicates": {},
            "video_info": {},
        }
    
    def check_completeness(self) -> List[str]:
        """Compare downloaded videos against expected list"""
        list_file = self.annotation_dir / "ava_file_names_trainval_v2.1.txt"
        
        if not list_file.exists():
            print(f"Warning: Video list file not found: {list_file}")
            return []
        
        with open(list_file) as f:
            expected = set(line.strip() for line in f if line.strip())
        
        self.report["total_expected"] = len(expected)
        
        # Find available videos
        found_ids = set()
        for vfile in self.video_dir.iterdir():
            if vfile.suffix.lower() in ('.mp4', '.mkv', '.webm', '.avi'):
                found_ids.add(vfile.stem)
        
        self.report["total_found"] = len(found_ids)
        self.report["missing"] = list(expected - found_ids)
        
        coverage = len(found_ids) / len(expected) * 100 if expected else 0
        print(f"Completeness: {len(found_ids)}/{len(expected)} ({coverage:.1f}%)")
        print(f"Missing: {len(self.report['missing'])} videos")
        
        return self.report["missing"]
    
    @staticmethod
    def probe_video(video_path: str) -> Tuple[str, Dict]:
        """Use ffprobe to verify video is not corrupted"""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration,size",
                "-show_entries", "stream=codec_type,width,height,r_frame_rate",
                "-of", "json",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                return video_path, {"status": "corrupted", "error": result.stderr[:200]}
            
            info = json.loads(result.stdout)
            fmt = info.get("format", {})
            streams = info.get("streams", [])
            
            video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
            
            if not video_stream:
                return video_path, {"status": "no_video_stream"}
            
            duration = float(fmt.get("duration", 0))
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))
            
            issues = []
            # AVA clips should be ~901 seconds (15 minutes)
            if duration < 800:
                issues.append(f"short:{duration:.0f}s")
            if duration > 1000:
                issues.append(f"long:{duration:.0f}s")
            if width < 320 or height < 240:
                issues.append(f"low_res:{width}x{height}")
            
            return video_path, {
                "status": "ok" if not issues else "warning",
                "duration": duration,
                "resolution": f"{width}x{height}",
                "issues": issues,
            }
            
        except subprocess.TimeoutExpired:
            return video_path, {"status": "timeout"}
        except Exception as e:
            return video_path, {"status": "error", "error": str(e)}
    
    def check_integrity(self, max_workers: int = 8) -> List[str]:
        """Verify all videos are valid and not corrupted"""
        videos = list(self.video_dir.glob("*.*"))
        videos = [v for v in videos if v.suffix.lower() in ('.mp4', '.mkv', '.webm')]
        
        print(f"Checking integrity of {len(videos)} videos...")
        corrupted = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.probe_video, str(v)): v for v in videos}
            
            for future in as_completed(futures):
                path, info = future.result()
                vid = Path(path).stem
                self.report["video_info"][vid] = info
                
                if info["status"] in ("corrupted", "no_video_stream", "error", "timeout"):
                    corrupted.append(vid)
                    print(f"  [CORRUPT] {vid}: {info}")
        
        self.report["corrupted"] = corrupted
        print(f"Corrupted: {len(corrupted)}/{len(videos)}")
        return corrupted
    
    @staticmethod
    def compute_hash(video_path: str) -> Tuple[str, str, int]:
        """Compute partial hash (first 50MB + last 10MB) for duplicate detection"""
        h = hashlib.sha256()
        fsize = os.path.getsize(video_path)
        chunk = 50 * 1024 * 1024  # 50MB
        
        with open(video_path, 'rb') as f:
            h.update(f.read(min(chunk, fsize)))
            if fsize > chunk:
                f.seek(max(0, fsize - 10 * 1024 * 1024))
                h.update(f.read())
        
        return str(video_path), h.hexdigest(), fsize
    
    def check_duplicates(self, max_workers: int = 8) -> Dict:
        """Detect duplicates via filename and content"""
        videos = list(self.video_dir.glob("*.*"))
        videos = [v for v in videos if v.suffix.lower() in ('.mp4', '.mkv', '.webm', '.avi')]
        
        # Method 1: Filename duplicates (same stem, different extension)
        stem_map = defaultdict(list)
        for v in videos:
            stem_map[v.stem].append(str(v))
        
        filename_dups = {k: v for k, v in stem_map.items() if len(v) > 1}
        
        # Method 2: Content hash duplicates
        print("Computing content hashes...")
        hash_map = defaultdict(list)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(self.compute_hash, [str(v) for v in videos])
            for path, file_hash, fsize in results:
                hash_map[file_hash].append({"path": path, "size": fsize})
        
        content_dups = {h: v for h, v in hash_map.items() if len(v) > 1}
        
        self.report["duplicates"] = {
            "filename_duplicates": filename_dups,
            "content_duplicates": {h: [x["path"] for x in v] for h, v in content_dups.items()},
        }
        
        print(f"Filename duplicates: {len(filename_dups)} groups")
        print(f"Content duplicates: {len(content_dups)} groups")
        
        return self.report["duplicates"]
    
    def check_annotation_alignment(self) -> Dict:
        """Ensure every annotated video has a corresponding video file"""
        import pandas as pd
        
        train_csv = self.annotation_dir / "ava_train_v2.2.csv"
        val_csv = self.annotation_dir / "ava_val_v2.2.csv"
        
        available = set()
        for v in self.video_dir.iterdir():
            if v.suffix.lower() in ('.mp4', '.mkv', '.webm'):
                available.add(v.stem)
        
        alignment = {}
        
        if train_csv.exists():
            try:
                train_df = pd.read_csv(train_csv, header=None)
                annotated_train = set(train_df[0].unique())
                missing_train = annotated_train - available
                alignment["train"] = {
                    "total": len(annotated_train),
                    "missing": list(missing_train)
                }
                print(f"Train: {len(annotated_train) - len(missing_train)}/{len(annotated_train)} available")
            except Exception as e:
                print(f"Error reading train CSV: {e}")
        
        if val_csv.exists():
            try:
                val_df = pd.read_csv(val_csv, header=None)
                annotated_val = set(val_df[0].unique())
                missing_val = annotated_val - available
                alignment["val"] = {
                    "total": len(annotated_val),
                    "missing": list(missing_val)
                }
                print(f"Val: {len(annotated_val) - len(missing_val)}/{len(annotated_val)} available")
            except Exception as e:
                print(f"Error reading val CSV: {e}")
        
        self.report["annotation_alignment"] = alignment
        return alignment
    
    def resolve_duplicates(self, strategy: str = "keep_largest"):
        """Remove duplicate files, keeping best version"""
        dups = self.report.get("duplicates", {})
        
        # Handle filename duplicates
        for stem, paths in dups.get("filename_duplicates", {}).items():
            if strategy == "keep_largest":
                paths_sorted = sorted(paths, key=lambda p: os.path.getsize(p), reverse=True)
            elif strategy == "prefer_mp4":
                paths_sorted = sorted(paths, key=lambda p: (
                    0 if p.endswith('.mp4') else 1, -os.path.getsize(p)
                ))
            
            keep = paths_sorted[0]
            for remove_path in paths_sorted[1:]:
                backup = remove_path + ".dup_backup"
                print(f"Removing duplicate: {remove_path} (keeping {keep})")
                os.rename(remove_path, backup)
    
    def run_full_validation(self) -> Dict:
        """Run complete validation"""
        print("=" * 60)
        print("AVA Dataset Validation Report")
        print("=" * 60)
        
        print("\n--- Step 1: Completeness Check ---")
        self.check_completeness()
        
        print("\n--- Step 2: Integrity Check ---")
        self.check_integrity()
        
        print("\n--- Step 3: Duplicate Detection ---")
        self.check_duplicates()
        
        print("\n--- Step 4: Annotation Alignment ---")
        self.check_annotation_alignment()
        
        # Save report
        report_path = self.annotation_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
        
        print(f"\nReport saved to: {report_path}")
        
        return self.report


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Validate AVA Dataset")
    parser.add_argument('--video-dir', type=str, default='data/raw_videos')
    parser.add_argument('--annotation-dir', type=str, default='data/annotations')
    parser.add_argument('--resolve-duplicates', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    
    validator = AVADatasetValidator(
        video_dir=args.video_dir,
        annotation_dir=args.annotation_dir
    )
    
    report = validator.run_full_validation()
    
    if args.resolve_duplicates:
        validator.resolve_duplicates()


if __name__ == "__main__":
    main()
