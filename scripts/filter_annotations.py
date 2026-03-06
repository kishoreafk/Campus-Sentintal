#!/usr/bin/env python3
"""
AVA Annotation Filter

Reads the full AVA v2.2 train/val CSVs, keeps only rows matching a given
action class, and writes:

  training_data/class_<id>_<name>/
      manifests/
          train_manifest.csv   (video_id,video_path,label_id,label_name)
          val_manifest.csv
      filtered_annotations/
          ava_train_v2.2_filtered.csv
          ava_val_v2.2_filtered.csv
      video_ids/
          train_video_ids.txt
          val_video_ids.txt

This is the *only* place that decides which videos belong to a class run.

Usage:
    python scripts/filter_annotations.py --class-name kiss
    python scripts/filter_annotations.py --class-id 30
    python scripts/filter_annotations.py --class-name kiss --annotation-dir data/annotations
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.ava_classes import AVA_CLASSES, COMMON_CLASSES, get_class_ids_from_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AVA CSV layout (no header row)
#   0: video_id
#   1: timestamp (middle_frame_timestamp)
#   2: x1  (person bbox, normalised)
#   3: y1
#   4: x2
#   5: y2
#   6: action_id
#   7: person_id  (may be absent in some rows)
# ---------------------------------------------------------------------------
COL_VIDEO_ID = 0
COL_TIMESTAMP = 1
COL_ACTION_ID = 6


def _safe_class_dir_name(class_id: int, class_name: str) -> str:
    """Turn 'hug (a person)' → 'hug_a_person' for a safe directory name."""
    safe = re.sub(r"[^a-z0-9]+", "_", class_name.lower()).strip("_")
    return f"class_{class_id}_{safe}"


def _parse_rows(csv_path: Path) -> List[str]:
    """Return all non-empty lines from a CSV file."""
    lines: List[str] = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


def filter_annotations(
    annotation_dir: Path,
    class_id: int,
    class_name: str,
    output_root: Path,
    video_dir: Optional[Path] = None,
) -> Tuple[Path, Dict[str, object]]:
    """Filter AVA annotations for *class_id* and write training-ready outputs.

    Returns (class_output_dir, stats_dict).
    """
    dir_name = _safe_class_dir_name(class_id, class_name)
    class_dir = output_root / dir_name
    manifest_dir = class_dir / "manifests"
    filtered_dir = class_dir / "filtered_annotations"
    vidid_dir = class_dir / "video_ids"
    train_vid_dir = class_dir / "videos" / "train"
    val_vid_dir = class_dir / "videos" / "val"

    for d in (manifest_dir, filtered_dir, vidid_dir, train_vid_dir, val_vid_dir):
        d.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, object] = {"class_id": class_id, "class_name": class_name}

    for split in ("train", "val"):
        csv_name = f"ava_{split}_v2.2.csv"
        csv_path = annotation_dir / csv_name
        if not csv_path.exists():
            logger.error(f"Missing {csv_path} — run download_annotations.py first.")
            stats[f"{split}_rows"] = 0
            stats[f"{split}_videos"] = 0
            continue

        rows = _parse_rows(csv_path)
        matched_rows: List[str] = []
        video_ids: Set[str] = set()

        for row in rows:
            parts = row.split(",")
            if len(parts) < 7:
                continue
            try:
                action_id = int(parts[COL_ACTION_ID])
            except ValueError:
                continue
            if action_id != class_id:
                continue

            matched_rows.append(row)
            vid = parts[COL_VIDEO_ID].strip()
            video_ids.add(vid)

        # Write filtered CSV
        filtered_csv = filtered_dir / f"ava_{split}_v2.2_filtered.csv"
        with open(filtered_csv, "w", encoding="utf-8") as f:
            f.write("\n".join(matched_rows) + "\n" if matched_rows else "")

        # Write video-ID list
        sorted_ids = sorted(video_ids)
        vidid_file = vidid_dir / f"{split}_video_ids.txt"
        with open(vidid_file, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted_ids) + "\n" if sorted_ids else "")

        # Write manifest
        manifest_path = manifest_dir / f"{split}_manifest.csv"
        _write_manifest(manifest_path, sorted_ids, class_id, class_name,
                        split, class_dir, video_dir)

        stats[f"{split}_rows"] = len(matched_rows)
        stats[f"{split}_videos"] = len(sorted_ids)
        logger.info(
            f"  {split}: {len(matched_rows)} rows, "
            f"{len(sorted_ids)} unique videos → {filtered_csv.name}"
        )

    return class_dir, stats


def _write_manifest(
    manifest_path: Path,
    video_ids: List[str],
    class_id: int,
    class_name: str,
    split: str,
    class_dir: Path,
    video_dir: Optional[Path],
):
    """Write a manifest CSV with columns: video_id,video_path,label_id,label_name"""
    known_exts = (".mkv", ".mp4", ".webm", ".avi")

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("video_id,video_path,label_id,label_name\n")
        for vid in video_ids:
            # Resolve actual video path if video_dir is given
            video_path = ""
            if video_dir:
                for ext in known_exts:
                    candidate = video_dir / f"{vid}{ext}"
                    if candidate.exists():
                        video_path = str(candidate)
                        break
            # Fallback: expected location inside the class folder
            if not video_path:
                video_path = str(class_dir / "videos" / split / f"{vid}.mkv")

            f.write(f"{vid},{video_path},{class_id},{class_name}\n")


# ---------------------------------------------------------------------------
# Resolve class from CLI args
# ---------------------------------------------------------------------------
def _resolve_class(args) -> Tuple[int, str]:
    """Return (class_id, class_name) from --class-id or --class-name."""
    if args.class_id:
        cid = int(args.class_id)
        if cid not in AVA_CLASSES:
            logger.error(f"Unknown class id {cid}")
            sys.exit(1)
        return cid, AVA_CLASSES[cid]

    if args.class_name:
        name = args.class_name.strip().lower()
        if name in COMMON_CLASSES:
            cid = COMMON_CLASSES[name]
            return cid, AVA_CLASSES[cid]
        ids = get_class_ids_from_names([name])
        if ids:
            cid = ids[0]
            return cid, AVA_CLASSES[cid]
        logger.error(f"Cannot resolve class name '{args.class_name}'")
        sys.exit(1)

    logger.error("Provide --class-id or --class-name")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Filter AVA annotations for a single class")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--class-id", type=int, help="AVA action class ID (1-80)")
    g.add_argument("--class-name", type=str, help="Action name (e.g. kiss, hug, walk)")
    p.add_argument("--annotation-dir", type=str, default="data/annotations",
                    help="Directory containing ava_train_v2.2.csv / ava_val_v2.2.csv")
    p.add_argument("--output-root", type=str, default="training_data",
                    help="Root directory for training-ready outputs")
    p.add_argument("--video-dir", type=str, default="data/raw_videos",
                    help="Directory containing downloaded videos (for manifest paths)")
    return p.parse_args()


def main():
    args = parse_args()
    class_id, class_name = _resolve_class(args)
    logger.info(f"Filtering for class {class_id}: {class_name}")

    class_dir, stats = filter_annotations(
        annotation_dir=Path(args.annotation_dir),
        class_id=class_id,
        class_name=class_name,
        output_root=Path(args.output_root),
        video_dir=Path(args.video_dir),
    )

    logger.info(f"Output directory: {class_dir}")
    logger.info(f"Stats: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
