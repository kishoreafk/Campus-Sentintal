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
import shutil
import urllib.request
import urllib.error
import re
import zipfile
import tempfile
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
    "cvdf": "https://s3.amazonaws.com/ava-dataset",
    "huggingface": "https://huggingface.co/datasets/Loie/AVA-dataset/resolve/main/videos/",
    "youtube": "https://www.youtube.com/watch?v=",
}

OFFICIAL_ANNOTATION_ZIP_URL = "https://s3.amazonaws.com/ava-dataset/annotations/ava_v2.2.zip"

ANNOTATION_BASE_URLS = [
    # Prefer S3 mirror first.
    "https://s3.amazonaws.com/ava-dataset/annotations",
    "https://ava-dataset.s3.amazonaws.com/annotations",
    # Fallbacks
    "https://storage.googleapis.com/ava-dataset/annotations",
    "https://huggingface.co/datasets/Loie/AVA-dataset/resolve/main/annotations",
    "https://huggingface.co/datasets/Loie/AVA-dataset/resolve/main",
    # Legacy fallback only; should never be a hard dependency.
    "https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations",
]

PROPOSAL_CANDIDATE_URLS = {
    "ava_dense_proposals_train.FAIR.recall_93.9.pkl": [
        "https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl",
        "https://huggingface.co/datasets/Loie/AVA-dataset/resolve/main/annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl",
    ],
    "ava_dense_proposals_val.FAIR.recall_93.9.pkl": [
        "https://dl.fbaipublicfiles.com/video-long-term-feature-banks/data/ava/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl",
        "https://huggingface.co/datasets/Loie/AVA-dataset/resolve/main/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl",
    ],
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
        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        
        # Status tracking
        self.status_file = self.video_dir / "download_status.json"
        self.status = self._load_status()

        # Track raw/canonical filename variants from annotation lists
        self.video_id_variants: Dict[str, Set[str]] = {}
        
        # Class filter
        self.filter_class_ids = set()
        if selected_classes:
            self.filter_class_ids = set(get_class_ids_from_names(selected_classes))
    
    def _load_status(self) -> Dict:
        """Load previous download status, cleaning any garbage entries."""
        if self.status_file.exists():
            try:
                with open(self.status_file) as f:
                    raw = json.load(f)
                # Only keep entries with valid video-ID-shaped keys
                cleaned = {
                    k: v for k, v in raw.items()
                    if re.match(r"^[A-Za-z0-9_-]{8,20}$", k)
                }
                if len(cleaned) < len(raw):
                    logger.info(
                        f"Cleaned {len(raw) - len(cleaned)} invalid entries "
                        "from download_status.json"
                    )
                return cleaned
            except (json.JSONDecodeError, TypeError):
                logger.warning("Corrupt download_status.json — starting fresh.")
        return {}
    
    def _save_status(self):
        """Save download status"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def download_annotations(self):
        """Download AVA annotations"""
        logger.info("Downloading AVA annotations...")

        files_to_download = [
            "ava_train_v2.2.csv",
            "ava_val_v2.2.csv",
            "ava_action_list_v2.2_for_activitynet_2019.pbtxt",
            "ava_train_excluded_timestamps_v2.2.csv",
            "ava_val_excluded_timestamps_v2.2.csv",
            "ava_file_names_trainval_v2.1.txt",
        ]

        # Prefer the official Google bundle first. It is the most reliable path
        # for the core AVA v2.2 annotation files.
        self._download_official_annotation_bundle(files_to_download)

        for filename in files_to_download:
            dest = self.annotation_dir / filename
            if self._validate_annotation_file(dest, filename):
                logger.info(f"Annotation ready: {filename}")
                continue

            if self._download_and_validate_annotation(filename, dest):
                logger.info(f"Annotation ready: {filename}")
            else:
                logger.error(
                    f"Could not obtain valid annotation file: {filename}. "
                    "Subsequent steps may fail until this file is available."
                )

        self._download_proposals()

    def _download_official_annotation_bundle(self, required_filenames: List[str]) -> bool:
        """Download and extract the official AVA v2.2 annotation zip."""
        bundle_members = {
            "ava_train_v2.2.csv",
            "ava_val_v2.2.csv",
            "ava_action_list_v2.2_for_activitynet_2019.pbtxt",
            "ava_train_excluded_timestamps_v2.2.csv",
            "ava_val_excluded_timestamps_v2.2.csv",
        }
        missing = [
            name for name in required_filenames
            if name in bundle_members
            and not self._validate_annotation_file(self.annotation_dir / name, name)
        ]
        if not missing:
            return True

        with tempfile.TemporaryDirectory(prefix="ava_annotations_") as temp_dir:
            zip_path = Path(temp_dir) / "ava_v2.2.zip"
            logger.info(f"Downloading official AVA annotation bundle from {OFFICIAL_ANNOTATION_ZIP_URL}")
            if not self._download_file(OFFICIAL_ANNOTATION_ZIP_URL, zip_path, timeout=300):
                logger.warning("Official AVA annotation bundle download failed. Falling back to per-file mirrors.")
                return False

            try:
                with zipfile.ZipFile(zip_path) as zf:
                    available_names = set(zf.namelist())
                    for filename in missing:
                        if filename not in available_names:
                            logger.warning(f"{filename} not present in official AVA bundle.")
                            continue

                        dest = self.annotation_dir / filename
                        zf.extract(filename, path=self.annotation_dir)
                        if self._validate_annotation_file(dest, filename):
                            logger.info(f"Extracted {filename} from official AVA bundle")
                        else:
                            logger.warning(f"Extracted {filename} from official bundle but validation failed")
                            dest.unlink(missing_ok=True)
            except zipfile.BadZipFile:
                logger.warning("Official AVA annotation bundle is corrupt or unreadable.")
                return False

        return True

    def _build_annotation_candidates(self, filename: str) -> List[str]:
        """Build ordered candidate URLs for an annotation file."""
        return [f"{base.rstrip('/')}/{filename}" for base in ANNOTATION_BASE_URLS]

    def _looks_like_html_error(self, path: Path) -> bool:
        """Detect HTML/error payloads saved as data files."""
        try:
            with open(path, "rb") as f:
                header = f.read(4096).decode("utf-8", errors="ignore").lower()
        except Exception:
            return True

        return (
            "<!doctype html" in header
            or "<html" in header
            or "<title>error" in header
        )

    def _validate_annotation_file(self, path: Path, filename: str) -> bool:
        """Validate annotation file content to avoid accepting HTTP error payloads."""
        if not path.exists() or path.stat().st_size == 0:
            return False

        if self._looks_like_html_error(path):
            return False

        try:
            if filename in {"ava_train_v2.2.csv", "ava_val_v2.2.csv"}:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        row = line.strip()
                        if not row:
                            continue
                        parts = row.split(",")
                        if len(parts) < 7:
                            continue
                        if not re.match(r"^[A-Za-z0-9_-]{8,20}$", parts[0].strip()):
                            continue
                        if not parts[1].strip().isdigit():
                            continue
                        return True
                return False

            if filename in {
                "ava_train_excluded_timestamps_v2.2.csv",
                "ava_val_excluded_timestamps_v2.2.csv",
            }:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        row = line.strip()
                        if not row:
                            continue
                        parts = row.split(",")
                        if len(parts) >= 2 and parts[1].strip().isdigit():
                            return True
                return False

            if filename == "ava_file_names_trainval_v2.1.txt":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        raw = line.strip()
                        if not raw:
                            continue
                        normalized = self._normalize_video_id(raw)
                        if re.match(r"^[A-Za-z0-9_-]{8,20}$", normalized):
                            return True
                return False

            if filename.endswith(".pbtxt"):
                text = path.read_text(encoding="utf-8", errors="ignore")
                return ("item {" in text) and ("name:" in text) and ("id:" in text)

            return path.stat().st_size > 32
        except Exception:
            return False

    def _download_and_validate_annotation(self, filename: str, dest: Path) -> bool:
        """Download a single annotation file with strict validation and mirror fallback."""
        if dest.exists():
            if self._validate_annotation_file(dest, filename):
                return True
            logger.warning(f"Existing {filename} is invalid. Re-downloading.")
            dest.unlink(missing_ok=True)

        for url in self._build_annotation_candidates(filename):
            logger.info(f"Downloading {filename} from {url}")
            ok = self._download_file(url, dest, timeout=120)
            if not ok:
                continue

            if self._validate_annotation_file(dest, filename):
                return True

            logger.warning(
                f"Downloaded payload for {filename} from {url} is invalid "
                "(likely HTML/error content). Trying next mirror."
            )
            dest.unlink(missing_ok=True)

        return False

    def _download_proposals(self):
        """Best-effort proposal file downloads. Failures do not abort setup."""
        proposal_dir = Path("data/ava/proposals")
        proposal_dir.mkdir(parents=True, exist_ok=True)

        for filename, urls in PROPOSAL_CANDIDATE_URLS.items():
            dest = proposal_dir / filename
            if dest.exists() and dest.stat().st_size > 1024:
                continue

            if dest.exists() and dest.stat().st_size <= 1024:
                dest.unlink(missing_ok=True)

            downloaded = False
            for url in urls:
                logger.info(f"Downloading proposal {filename} from {url}")
                if self._download_file(url, dest, timeout=300):
                    if dest.exists() and dest.stat().st_size > 1024:
                        downloaded = True
                        break
                    logger.warning(
                        f"Downloaded proposal {filename} from {url} appears incomplete."
                    )
                    dest.unlink(missing_ok=True)

            if not downloaded:
                logger.warning(
                    f"Optional proposal file unavailable: {filename}. Continuing without it."
                )

    def _normalize_video_id(self, value: str) -> str:
        """Normalize video identifier to a canonical AVA YouTube ID-style stem."""
        cleaned = value.strip()
        if not cleaned:
            return ""

        # Remove URL/query fragments and path prefixes if present
        cleaned = cleaned.split("?")[0].split("#")[0].replace("\\", "/")
        cleaned = Path(cleaned).name

        # Strip known video extension if present
        if Path(cleaned).suffix.lower() in {".mp4", ".mkv", ".webm", ".avi", ".mov"}:
            cleaned = Path(cleaned).stem

        return cleaned

    def _register_video_variant(self, canonical_id: str, raw_value: str):
        """Keep raw and canonical variants to build robust mirror URL candidates."""
        if not canonical_id:
            return

        variants = self.video_id_variants.setdefault(canonical_id, set())
        candidates = {
            canonical_id,
            raw_value.strip(),
            Path(raw_value.strip().replace("\\", "/")).name,
            self._normalize_video_id(raw_value),
        }

        for candidate in candidates:
            if candidate:
                variants.add(candidate)

    def _url_is_available(self, url: str, timeout: int = 20) -> bool:
        """Check URL availability using strict HTTP status validation."""
        headers = {'User-Agent': 'Mozilla/5.0'}

        # Try HEAD first
        try:
            request = urllib.request.Request(url, method='HEAD', headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return 200 <= response.status < 300
        except urllib.error.HTTPError as e:
            # Some endpoints disallow HEAD. Retry with a tiny ranged GET.
            if e.code not in (403, 405):
                return False
        except Exception:
            return False

        # Fallback: ranged GET
        try:
            request = urllib.request.Request(url, headers={**headers, 'Range': 'bytes=0-0'})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status in (200, 206)
        except Exception:
            return False

    def _download_file(self, url: str, output: Path, timeout: int = 600) -> bool:
        """Download with aria2c fast-path, curl fallback, and stdlib HTTP fallback."""
        output.parent.mkdir(parents=True, exist_ok=True)

        # Try aria2c first if available
        if shutil.which("aria2c"):
            try:
                result = subprocess.run([
                    "aria2c", "-x", "16", "-s", "16", "-k", "1M",
                    "-d", str(output.parent),
                    "-o", output.name,
                    url
                ], timeout=timeout, capture_output=True, text=True)
                if result.returncode == 0 and output.exists() and output.stat().st_size > 0:
                    return True
                self._log_download_failure(url, result)
            except Exception:
                pass

        # Fallback to curl
        try:
            result = subprocess.run([
                "curl", "-L", "--fail",
                "--connect-timeout", "30",
                "--max-time", str(timeout),
                "--retry", "3",
                "--retry-delay", "5",
                "-o", str(output), url
            ], timeout=timeout + 60, capture_output=True, text=True)
            if result.returncode == 0 and output.exists() and output.stat().st_size > 0:
                return True
            self._log_download_failure(url, result)
        except Exception:
            pass

        # Final fallback: Python stdlib streaming download.
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with open(output, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

            if output.exists() and output.stat().st_size > 0:
                return True
        except Exception as exc:
            logger.warning(f"Python fallback download failed for {url}: {exc}")

        # Cleanup partial file if download failed
        if output.exists() and output.stat().st_size <= 0:
            output.unlink(missing_ok=True)

        return False

    def _log_download_failure(self, url: str, result: subprocess.CompletedProcess):
        """Emit actionable logs for common network/download failure classes."""
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        combined = f"{stderr}\n{stdout}".lower()

        if "curl: (28)" in combined or "timed out" in combined:
            logger.warning(f"Timeout/unreachable host while downloading {url}")
            return

        if "curl: (22)" in combined or "http response code" in combined or "requested url returned error" in combined:
            logger.warning(f"HTTP error while downloading {url}")
            return

        logger.warning(f"Download failed for {url} (exit={result.returncode})")

    def _build_cvdf_candidates(self, video_id: str) -> List[str]:
        """Build robust URL candidates for CVDF mirror.

        The verified working pattern is:
            https://s3.amazonaws.com/ava-dataset/trainval/<video_id>.mkv
        We put that first, then try other extension/prefix combinations.
        """
        base = AVAILABLE_SOURCES["cvdf"].rstrip("/")
        variants = self.video_id_variants.get(video_id, {video_id})
        known_exts = {".mkv", ".mp4", ".webm", ".avi"}
        # .mkv is the dominant format on the S3 mirror; try it first.
        exts_to_try = ["mkv", "mp4", "webm"]
        # trainval/ is the confirmed working prefix.
        prefixes = ["trainval/", "test/", ""]

        candidates: List[str] = []
        seen: set = set()

        def _add(url: str):
            if url not in seen:
                seen.add(url)
                candidates.append(url)

        # Highest-priority: trainval/<video_id>.mkv (verified pattern)
        _add(f"{base}/trainval/{video_id}.mkv")

        # Then iterate variants deterministically (sorted for stability)
        for variant in sorted(variants):
            name = Path(variant.strip().replace("\\", "/")).name
            if not name:
                continue

            has_ext = Path(name).suffix.lower() in known_exts
            filenames = [name] if has_ext else [f"{name}.{ext}" for ext in exts_to_try]

            for filename in filenames:
                for prefix in prefixes:
                    _add(f"{base}/{prefix}{filename}")

        return candidates

    def _build_huggingface_candidates(self, video_id: str) -> List[str]:
        """Build robust URL candidates for HuggingFace mirror."""
        base = AVAILABLE_SOURCES["huggingface"]
        variants = self.video_id_variants.get(video_id, {video_id})
        known_exts = {".mkv", ".mp4", ".webm", ".avi"}
        exts_to_try = ["mkv", "mp4"]

        candidates = []
        seen = set()

        for variant in variants:
            name = Path(variant.strip().replace("\\", "/")).name
            if not name:
                continue

            has_ext = Path(name).suffix.lower() in known_exts
            filenames = [name] if has_ext else [f"{name}.{ext}" for ext in exts_to_try]

            for filename in filenames:
                url = f"{base}{filename}"
                if url not in seen:
                    seen.add(url)
                    candidates.append(url)

        return candidates
    
    @staticmethod
    def _is_valid_video_id(video_id: str) -> bool:
        """Check if a string looks like a valid AVA YouTube-style video ID."""
        return bool(re.match(r"^[A-Za-z0-9_-]{8,20}$", video_id))

    def _video_ids_from_csv(self, csv_path: Path) -> List[str]:
        """Extract unique video IDs from an AVA annotation CSV file."""
        video_ids = []
        seen: set = set()
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                vid = self._normalize_video_id(parts[0])
                if vid and self._is_valid_video_id(vid) and vid not in seen:
                    seen.add(vid)
                    self._register_video_variant(vid, parts[0])
                    video_ids.append(vid)
        return video_ids

    def get_video_list(self) -> List[str]:
        """Get list of video IDs from annotations"""
        list_file = self.annotation_dir / "ava_file_names_trainval_v2.1.txt"
        
        video_ids: List[str] = []
        seen: set = set()

        if list_file.exists() and self._validate_annotation_file(list_file, list_file.name):
            with open(list_file) as f:
                for line in f:
                    raw_value = line.strip()
                    if not raw_value:
                        continue

                    normalized = self._normalize_video_id(raw_value)
                    if not normalized or not self._is_valid_video_id(normalized):
                        continue

                    self._register_video_variant(normalized, raw_value)

                    if normalized not in seen:
                        seen.add(normalized)
                        video_ids.append(normalized)
        else:
            # Fallback: extract video IDs from train/val CSV annotations
            logger.warning(
                "Video list file not found or invalid. "
                "Falling back to video IDs from annotation CSVs."
            )
            for csv_name in ["ava_train_v2.2.csv", "ava_val_v2.2.csv"]:
                csv_path = self.annotation_dir / csv_name
                if csv_path.exists():
                    for vid in self._video_ids_from_csv(csv_path):
                        if vid not in seen:
                            seen.add(vid)
                            video_ids.append(vid)

        if not video_ids:
            logger.error("No valid video IDs found from any source")
            return []

        # Filter by selected classes if specified
        if self.filter_class_ids:
            filtered = set()
            for csv_name in ["ava_train_v2.2.csv", "ava_val_v2.2.csv"]:
                csv_path = self.annotation_dir / csv_name
                if not csv_path.exists():
                    continue
                with open(csv_path) as f:
                    for line in f:
                        parts = line.strip().split(',')
                        # AVA CSV: video_id(0), ts(1), x1(2), y1(3), x2(4), y2(5), action_id(6), person_id(7)
                        if len(parts) < 7:
                            continue
                        video_id = self._normalize_video_id(parts[0])
                        try:
                            action_id = int(parts[6])
                        except ValueError:
                            continue
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
    
    def _validate_downloaded_video(self, path: Path, min_size_mb: float = 1.0) -> bool:
        """Check that a downloaded video file is large enough to be valid."""
        if not path.exists():
            return False
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb < min_size_mb:
            logger.warning(
                f"Downloaded file {path.name} is only {size_mb:.1f} MB "
                f"(min {min_size_mb} MB). Treating as incomplete."
            )
            path.unlink(missing_ok=True)
            return False
        return True

    def try_cvdf_source(self, video_id: str) -> bool:
        """Try downloading from CVDF / Facebook S3 mirror"""
        for url in self._build_cvdf_candidates(video_id):
            ext = Path(url).suffix
            output = self.video_dir / f"{video_id}{ext}"

            try:
                if not self._url_is_available(url):
                    continue

                logger.info(f"CVDF mirror candidate matched: {url}")
                if self._download_file(url, output, timeout=1800):
                    if self._validate_downloaded_video(output):
                        return True
            except Exception as e:
                logger.debug(f"CVDF failed for {video_id} via {url}: {e}")
                continue
        
        return False
    
    def try_huggingface_source(self, video_id: str) -> bool:
        """Try downloading from Hugging Face mirror"""
        for url in self._build_huggingface_candidates(video_id):
            ext = Path(url).suffix
            output = self.video_dir / f"{video_id}{ext}"

            try:
                if not self._url_is_available(url):
                    continue

                logger.info(f"HuggingFace mirror candidate matched: {url}")
                if self._download_file(url, output, timeout=1800):
                    if self._validate_downloaded_video(output):
                        return True
            except Exception as e:
                logger.debug(f"HuggingFace failed for {video_id} via {url}: {e}")
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
