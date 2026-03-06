#!/usr/bin/env python3
"""
AVA Video Downloader

Downloads AVA videos for a filtered class run.  Reads the video-ID list
produced by filter_annotations.py and downloads each video.

Download priority:
  1. S3 AVA mirror (trainval/<video_id>.mkv) — verified working
     Fallback extensions: .mkv, .mp4, .webm
  2. YouTube via yt-dlp + ffmpeg trim (15:00–30:01) when --source auto

Downloaded files are placed *both* in the shared --video-dir and
symlinked/copied into the class training folder.

Usage:
    # Download videos for a class using its video-ID list
    python scripts/download_videos.py --video-ids training_data/class_30_kiss/video_ids/train_video_ids.txt

    # Download all videos for a class (train + val) with YouTube fallback
    python scripts/download_videos.py \\
        --class-dir training_data/class_30_kiss \\
        --source auto

    # Limit number of videos
    python scripts/download_videos.py --class-dir training_data/class_30_kiss --max-videos 5
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# --- Logging setup -----------------------------------------------------------
# Console: clean, minimal (INFO level, short format)
# File:    detailed debug log under logs/
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = _LOG_DIR / f"download_{_run_ts}.log"

logger = logging.getLogger("ava_downloader")
logger.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_console_handler)

_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
)
logger.addHandler(_file_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
S3_BASE = "https://s3.amazonaws.com/ava-dataset"

# Extension priority for the S3 mirror (mkv is the dominant format).
MIRROR_EXTENSIONS = ["mkv", "mp4", "webm"]

# AVA temporal window used for YouTube fallback trimming.
AVA_TRIM_START = 900   # 15:00
AVA_TRIM_DURATION = 901  # 15:01  → covers 900–1801 s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _url_exists(url: str, timeout: int = 20) -> bool:
    """Return True if *url* responds with 2xx."""
    headers = {"User-Agent": "Mozilla/5.0"}
    # Try HEAD
    try:
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        if exc.code not in (403, 405):
            return False
    except Exception:
        return False
    # Fallback: ranged GET
    try:
        req = urllib.request.Request(url, headers={**headers, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 206)
    except Exception:
        return False


def _download_file(url: str, dest: Path, timeout: int = 1800) -> bool:
    """Download *url* → *dest*, trying aria2c → curl → Python stdlib."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # aria2c
    if shutil.which("aria2c"):
        try:
            r = subprocess.run(
                ["aria2c", "-x", "16", "-s", "16", "-k", "1M",
                 "-d", str(dest.parent), "-o", dest.name, url],
                timeout=timeout, capture_output=True, text=True,
            )
            if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
                return True
        except Exception:
            pass

    # curl
    try:
        r = subprocess.run(
            ["curl", "-L", "--fail", "--connect-timeout", "30",
             "--max-time", str(timeout), "--retry", "3", "--retry-delay", "5",
             "-o", str(dest), url],
            timeout=timeout + 120, capture_output=True, text=True,
        )
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return True
    except Exception:
        pass

    # Python stdlib
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        if dest.exists() and dest.stat().st_size > 0:
            return True
    except Exception as exc:
        logger.warning(f"Python download failed for {url}: {exc}")

    # Cleanup empty file
    if dest.exists() and dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
    return False


def _video_exists(video_dir: Path, video_id: str) -> Optional[Path]:
    """Return the path of an existing video file for *video_id*, or None."""
    for ext in MIRROR_EXTENSIONS:
        p = video_dir / f"{video_id}.{ext}"
        if p.exists() and p.stat().st_size > 1_000_000:  # >1 MB
            return p
    return None


# ---------------------------------------------------------------------------
# Source: S3 AVA mirror
# ---------------------------------------------------------------------------
def try_s3_mirror(video_id: str, video_dir: Path) -> Optional[Path]:
    """Try downloading from S3 trainval/ with each extension."""
    for ext in MIRROR_EXTENSIONS:
        url = f"{S3_BASE}/trainval/{video_id}.{ext}"
        dest = video_dir / f"{video_id}.{ext}"
        if _url_exists(url):
            logger.debug(f"S3 mirror hit: {url}")
            if _download_file(url, dest):
                size_mb = dest.stat().st_size / (1024 * 1024)
                if dest.stat().st_size > 1_000_000:
                    return dest
                else:
                    logger.debug(f"Downloaded file too small ({dest.stat().st_size} B) — discarding")
                    dest.unlink(missing_ok=True)
    return None


# ---------------------------------------------------------------------------
# Source: YouTube fallback (yt-dlp + ffmpeg trim)
# ---------------------------------------------------------------------------
def try_youtube(video_id: str, video_dir: Path) -> Optional[Path]:
    """Download from YouTube and trim to the AVA 15-min segment."""
    if not shutil.which("yt-dlp"):
        logger.debug("yt-dlp not installed — skipping YouTube source")
        return None

    url = f"https://www.youtube.com/watch?v={video_id}"
    temp_path = video_dir / f"_yt_temp_{video_id}.mp4"
    output_path = video_dir / f"{video_id}.mp4"

    try:
        # Step 1: download
        r = subprocess.run(
            ["yt-dlp",
             "-f", "best[height<=720][ext=mp4]/best[height<=720]",
             "--merge-output-format", "mp4",
             "-o", str(temp_path),
             "--retries", "3",
             "--socket-timeout", "30",
             url],
            capture_output=True, text=True, timeout=900,
        )
        if r.returncode != 0 or not temp_path.exists():
            return None

        # Step 2: trim to the AVA temporal window
        if not shutil.which("ffmpeg"):
            logger.warning("ffmpeg not found — keeping un-trimmed video")
            temp_path.rename(output_path)
            return output_path

        subprocess.run(
            ["ffmpeg", "-y",
             "-ss", str(AVA_TRIM_START),
             "-i", str(temp_path),
             "-t", str(AVA_TRIM_DURATION),
             "-c:v", "libx264", "-crf", "18",
             "-c:a", "aac",
             "-threads", "4",
             str(output_path)],
            capture_output=True, timeout=600,
        )

        if temp_path.exists():
            temp_path.unlink()

        if output_path.exists() and output_path.stat().st_size > 1_000_000:
            return output_path
        return None

    except Exception as exc:
        logger.debug(f"YouTube fallback failed for {video_id}: {exc}")
        for p in (temp_path, output_path):
            if p.exists():
                p.unlink(missing_ok=True)
        return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def download_video(
    video_id: str,
    video_dir: Path,
    source: str = "auto",
    skip_existing: bool = True,
) -> dict:
    """Download one video.

    Returns a result dict:
        {"video_id": str, "status": "success"|"exists"|"failed",
         "path": Path|None, "source": str|None, "size_mb": float,
         "error": str|None, "timestamp": str}
    """
    ts = datetime.now().isoformat(timespec="seconds")

    # Already present?
    if skip_existing:
        existing = _video_exists(video_dir, video_id)
        if existing:
            size_mb = existing.stat().st_size / (1024 * 1024)
            return {"video_id": video_id, "status": "exists",
                    "path": existing, "source": "local",
                    "size_mb": round(size_mb, 1), "error": None,
                    "timestamp": ts}

    # S3 mirror (always tried first)
    result = try_s3_mirror(video_id, video_dir)
    if result:
        size_mb = result.stat().st_size / (1024 * 1024)
        return {"video_id": video_id, "status": "success",
                "path": result, "source": "S3 mirror",
                "size_mb": round(size_mb, 1), "error": None,
                "timestamp": ts}

    # YouTube fallback
    if source == "auto":
        logger.debug(f"{video_id}: S3 miss — trying YouTube fallback")
        result = try_youtube(video_id, video_dir)
        if result:
            size_mb = result.stat().st_size / (1024 * 1024)
            return {"video_id": video_id, "status": "success",
                    "path": result, "source": "YouTube",
                    "size_mb": round(size_mb, 1), "error": None,
                    "timestamp": ts}

    err = "all sources exhausted (S3" + ("+YouTube" if source == "auto" else "") + ")"
    return {"video_id": video_id, "status": "failed",
            "path": None, "source": None,
            "size_mb": 0, "error": err,
            "timestamp": ts}


def download_videos(
    video_ids: List[str],
    video_dir: Path,
    source: str = "auto",
    skip_existing: bool = True,
    max_videos: Optional[int] = None,
    delay: float = 0.5,
    split: str = "",
    class_info: str = "",
) -> Dict[str, dict]:
    """Download a list of videos.

    Returns ``{video_id: result_dict}`` — see :func:`download_video` for the
    shape of each result dict.
    """
    video_dir.mkdir(parents=True, exist_ok=True)

    if max_videos and len(video_ids) > max_videos:
        video_ids = video_ids[:max_videos]

    results: Dict[str, dict] = {}
    total = len(video_ids)
    n_ok = n_exist = n_fail = 0

    header = "Download"
    if split:
        header += f" [{split}]"
    if class_info:
        header += f" — {class_info}"
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  {header}  ({total} videos)")
    logger.info(f"{'=' * 60}")

    for i, vid in enumerate(video_ids, 1):
        r = download_video(vid, video_dir, source=source,
                           skip_existing=skip_existing)
        results[vid] = r

        # --- clean one-liner per video ---
        idx = f"[{i:>{len(str(total))}}/{total}]"
        if r["status"] == "success":
            n_ok += 1
            logger.info(f"  {idx}  OK    {vid}  ({r['source']}, {r['size_mb']} MB)")
        elif r["status"] == "exists":
            n_exist += 1
            logger.info(f"  {idx}  SKIP  {vid}  (exists, {r['size_mb']} MB)")
        else:
            n_fail += 1
            logger.info(f"  {idx}  FAIL  {vid}  ({r['error']})")

        if i < total:
            time.sleep(delay)

    # --- summary block ---
    logger.info(f"{'─' * 60}")
    logger.info(f"  Summary: {n_ok} downloaded, {n_exist} skipped, {n_fail} failed  (total {total})")
    logger.info(f"{'─' * 60}")

    # --- write failure log ---
    failures = [r for r in results.values() if r["status"] == "failed"]
    if failures:
        fail_log = _LOG_DIR / f"failed_{_run_ts}.json"
        payload = {
            "run_timestamp": _run_ts,
            "split": split,
            "class_info": class_info,
            "source": source,
            "summary": {
                "total": total,
                "downloaded": n_ok,
                "skipped": n_exist,
                "failed": n_fail,
            },
            "failed_videos": [
                {"video_id": f["video_id"], "error": f["error"],
                 "timestamp": f["timestamp"]}
                for f in failures
            ],
        }
        with open(fail_log, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info(f"  Failed video IDs logged to: {fail_log}")

    logger.info(f"  Full log: {_log_file}\n")
    return results


def _link_videos_to_class_dir(
    results: Dict[str, dict],
    class_dir: Path,
    split: str,
):
    """Copy/link downloaded videos into the class training folder."""
    dest_dir = class_dir / "videos" / split
    dest_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for vid, r in results.items():
        src = r.get("path")
        if src is None or not Path(src).exists():
            continue
        src = Path(src)
        dest = dest_dir / src.name
        if dest.exists():
            continue
        try:
            os.link(src, dest)  # hard link (saves space)
            linked += 1
        except OSError:
            shutil.copy2(src, dest)
            linked += 1
    if linked:
        logger.info(f"  Linked {linked} videos → {dest_dir}")


def verify_and_repair_links(
    class_dir: Path,
    video_dir: Path,
) -> Dict[str, dict]:
    """Scan class dir, link any videos present in raw_videos but missing
    from the class videos folder. Returns a report dict per split."""
    report: Dict[str, dict] = {}

    for split in ("train", "val"):
        id_file = class_dir / "video_ids" / f"{split}_video_ids.txt"
        if not id_file.exists():
            continue
        ids = _read_id_file(id_file)
        dest_dir = class_dir / "videos" / split
        dest_dir.mkdir(parents=True, exist_ok=True)

        linked = already = missing = 0
        missing_ids: List[str] = []

        for vid in ids:
            # Already in class dir?
            existing_class = _video_exists(dest_dir, vid)
            if existing_class:
                already += 1
                continue

            # Exists in shared video dir?
            existing_raw = _video_exists(video_dir, vid)
            if existing_raw:
                dest = dest_dir / existing_raw.name
                try:
                    os.link(existing_raw, dest)
                except OSError:
                    shutil.copy2(existing_raw, dest)
                linked += 1
            else:
                missing += 1
                missing_ids.append(vid)

        report[split] = {
            "total": len(ids),
            "already_linked": already,
            "newly_linked": linked,
            "missing": missing,
            "missing_ids": missing_ids,
        }

        logger.info(f"  [{split}] {already} ok, {linked} newly linked, {missing} missing")
        if missing_ids:
            logger.debug(f"  Missing IDs ({split}): {missing_ids[:10]}{'...' if missing > 10 else ''}")

    # Regenerate manifests with actual file extensions
    _regenerate_manifests(class_dir)
    return report


def _regenerate_manifests(class_dir: Path):
    """Rewrite manifests so video_path reflects the actual file on disk."""
    # Extract class_id and class_name from dir name: class_<id>_<name>
    dirname = class_dir.name
    parts = dirname.split("_", 2)
    class_id = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
    class_name = parts[2] if len(parts) >= 3 else dirname

    manifest_dir = class_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        id_file = class_dir / "video_ids" / f"{split}_video_ids.txt"
        if not id_file.exists():
            continue
        ids = _read_id_file(id_file)
        vid_dir = class_dir / "videos" / split
        manifest_path = manifest_dir / f"{split}_manifest.csv"

        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write("video_id,video_path,label_id,label_name\n")
            for vid in ids:
                actual = _video_exists(vid_dir, vid)
                if actual:
                    vpath = str(actual)
                else:
                    vpath = str(vid_dir / f"{vid}.mkv")  # placeholder
                f.write(f"{vid},{vpath},{class_id},{class_name}\n")

    logger.info(f"  Manifests regenerated in {manifest_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _read_id_file(path: Path) -> List[str]:
    ids = []
    with open(path, "r") as f:
        for line in f:
            v = line.strip()
            if v:
                ids.append(v)
    return ids


def parse_args():
    p = argparse.ArgumentParser(description="Download AVA videos (S3-mirror-first)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video-ids", type=str,
                    help="Path to a text file with one video ID per line")
    g.add_argument("--class-dir", type=str,
                    help="Class output dir from filter_annotations.py "
                         "(reads video_ids/train_video_ids.txt + val_video_ids.txt)")
    g.add_argument("--verify-links", type=str, metavar="CLASS_DIR",
                    help="Verify and repair hard links in a class dir "
                         "(no downloads, just re-link from video-dir)")
    p.add_argument("--video-dir", type=str, default="data/raw_videos",
                    help="Shared directory for downloaded videos")
    p.add_argument("--source", type=str, default="auto",
                    choices=["mirror", "auto"],
                    help="'mirror' = S3 only; 'auto' = S3 + YouTube fallback")
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--delay", type=float, default=0.5,
                    help="Seconds between downloads (rate-limiting)")
    return p.parse_args()


def main():
    args = parse_args()
    video_dir = Path(args.video_dir)
    skip_existing = not args.no_skip_existing

    if args.verify_links:
        # Verify-only mode: repair links and regenerate manifests
        class_dir = Path(args.verify_links)
        logger.info(f"\nVerifying links for {class_dir.name}")
        logger.info(f"  Source: {video_dir}")
        report = verify_and_repair_links(class_dir, video_dir)
        for split, info in report.items():
            if info["missing_ids"]:
                logger.info(f"  {split} still missing {info['missing']} videos")
        return 0

    class_info = Path(args.class_dir).name if args.class_dir else ""

    if args.video_ids:
        # Single ID list mode
        ids = _read_id_file(Path(args.video_ids))
        logger.info(f"Loaded {len(ids)} video IDs from {args.video_ids}")
        download_videos(ids, video_dir, source=args.source,
                        skip_existing=skip_existing,
                        max_videos=args.max_videos,
                        delay=args.delay,
                        class_info=class_info)
    else:
        # Class-dir mode: process train + val splits
        class_dir = Path(args.class_dir)
        class_info = class_dir.name
        for split in ("train", "val"):
            id_file = class_dir / "video_ids" / f"{split}_video_ids.txt"
            if not id_file.exists():
                logger.warning(f"No {split} video-ID file at {id_file}")
                continue
            ids = _read_id_file(id_file)
            results = download_videos(ids, video_dir, source=args.source,
                                       skip_existing=skip_existing,
                                       max_videos=args.max_videos,
                                       delay=args.delay,
                                       split=split,
                                       class_info=class_info)
            _link_videos_to_class_dir(results, class_dir, split)

        # Post-download: verify all links and regenerate manifests
        logger.info("\nPost-download verification...")
        verify_and_repair_links(class_dir, video_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
