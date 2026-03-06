#!/usr/bin/env python3
"""
YOLO-ACT Training Pipeline

Complete pipeline to train YOLO-ACT model with AVA dataset:
1. Download annotations (S3 zip)
2. Filter annotations for selected class
3. Download videos (S3 mirror-first, YouTube fallback)
4. Extract frames from videos
5. Train the model

Usage:
    python pipeline.py --class-name kiss
    python pipeline.py --class-id 30
    python pipeline.py --class-name kiss --max-videos 10
    python pipeline.py --class-name kiss --skip-download --skip-extract
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.ava_classes import AVA_CLASSES, COMMON_CLASSES, get_class_ids_from_names

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline orchestrator"""

    def __init__(
        self,
        class_id: int,
        class_name: str,
        annotation_dir: str = "data/annotations",
        video_dir: str = "data/raw_videos",
        training_root: str = "training_data",
        frame_dir: Optional[str] = None,
        checkpoint_dir: str = "models/checkpoints",
        log_dir: str = "logs",
        source: str = "auto",
    ):
        self.class_id = class_id
        self.class_name = class_name
        self.annotation_dir = Path(annotation_dir)
        self.video_dir = Path(video_dir)
        self.training_root = Path(training_root)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.source = source

        # Derive class output directory (matching filter_annotations.py)
        import re
        safe = re.sub(r"[^a-z0-9]+", "_", class_name.lower()).strip("_")
        self.class_dir = self.training_root / f"class_{class_id}_{safe}"

        # Frame dir defaults to inside the class directory
        self.frame_dir = Path(frame_dir) if frame_dir else self.class_dir / "frames"

        # Ensure directories exist
        for d in (self.annotation_dir, self.video_dir, self.checkpoint_dir,
                  self.log_dir, self.class_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Download annotations
    # ------------------------------------------------------------------
    def download_annotations(self, force: bool = False) -> bool:
        logger.info("=" * 60)
        logger.info("STEP 1: Download AVA annotations")
        logger.info("=" * 60)
        from scripts.download_annotations import download_annotations
        return download_annotations(self.annotation_dir, force=force)

    # ------------------------------------------------------------------
    # Step 2: Filter annotations for selected class
    # ------------------------------------------------------------------
    def filter_annotations(self) -> bool:
        logger.info("=" * 60)
        logger.info(f"STEP 2: Filter annotations for class {self.class_id} ({self.class_name})")
        logger.info("=" * 60)
        from scripts.filter_annotations import filter_annotations
        class_dir, stats = filter_annotations(
            annotation_dir=self.annotation_dir,
            class_id=self.class_id,
            class_name=self.class_name,
            output_root=self.training_root,
            video_dir=self.video_dir,
        )
        logger.info(f"Filter stats: {stats}")
        return stats.get("train_videos", 0) > 0 or stats.get("val_videos", 0) > 0

    # ------------------------------------------------------------------
    # Step 3: Download videos
    # ------------------------------------------------------------------
    def download_videos(self, max_videos: Optional[int] = None) -> bool:
        logger.info("=" * 60)
        logger.info("STEP 3: Download videos")
        logger.info("=" * 60)
        from scripts.download_videos_v2 import download_videos, _read_id_file, _link_videos_to_class_dir

        any_ok = False
        for split in ("train", "val"):
            id_file = self.class_dir / "video_ids" / f"{split}_video_ids.txt"
            if not id_file.exists():
                logger.warning(f"No {split} video ID list at {id_file}")
                continue
            ids = _read_id_file(id_file)
            if not ids:
                continue
            logger.info(f"[{split}] {len(ids)} videos to download")
            results = download_videos(
                ids, self.video_dir,
                source=self.source,
                max_videos=max_videos,
            )
            _link_videos_to_class_dir(results, self.class_dir, split)
            ok = sum(1 for v in results.values() if v is not None)
            if ok > 0:
                any_ok = True
        return any_ok

    # ------------------------------------------------------------------
    # Step 4: Extract frames
    # ------------------------------------------------------------------
    def extract_frames(self, fps: int = 30, img_size: int = 416,
                       max_videos: Optional[int] = None) -> bool:
        logger.info("=" * 60)
        logger.info("STEP 4: Extract frames")
        logger.info("=" * 60)
        cmd = [
            sys.executable, "scripts/extract_frames.py",
            "--video-dir", str(self.class_dir / "videos" / "train"),
            "--frame-dir", str(self.frame_dir),
            "--annotation-dir", str(self.annotation_dir),
            "--fps", str(fps),
            "--img-size", str(img_size),
            "--generate-annotations",
            "--classes", self.class_name,
        ]
        if max_videos:
            cmd.extend(["--max-videos", str(max_videos)])

        import subprocess
        r = subprocess.run(cmd, env=os.environ.copy())
        return r.returncode == 0

    # ------------------------------------------------------------------
    # Step 5: Train
    # ------------------------------------------------------------------
    def train(self, epochs: int = 50, batch_size: int = 4, lr: float = 0.001,
              img_size: int = 416, backbone: str = "yolov8m.pt",
              device: str = "cuda") -> bool:
        logger.info("=" * 60)
        logger.info("STEP 5: Train model")
        logger.info("=" * 60)
        cmd = [
            sys.executable, "scripts/train_yolo_act.py",
            "--frame-dir", str(self.frame_dir),
            "--checkpoint-dir", str(self.checkpoint_dir),
            "--log-dir", str(self.log_dir),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--img-size", str(img_size),
            "--backbone", backbone,
            "--device", device,
            "--classes", self.class_name,
        ]
        import subprocess
        r = subprocess.run(cmd, env=os.environ.copy())
        return r.returncode == 0

    # ------------------------------------------------------------------
    # Run all steps
    # ------------------------------------------------------------------
    def run(
        self,
        skip_download: bool = False,
        skip_filter: bool = False,
        skip_videos: bool = False,
        skip_extract: bool = False,
        skip_train: bool = False,
        max_videos: Optional[int] = None,
        epochs: int = 50,
        batch_size: int = 4,
        **train_kwargs,
    ):
        logger.info("=" * 60)
        logger.info(f"YOLO-ACT PIPELINE — class {self.class_id}: {self.class_name}")
        logger.info("=" * 60)

        # Step 1
        if not skip_download:
            if not self.download_annotations():
                logger.error("Annotation download failed.")
                return False
        else:
            logger.info("[Step 1] Skipped annotation download")

        # Step 2
        if not skip_filter:
            self.filter_annotations()
        else:
            logger.info("[Step 2] Skipped annotation filtering")

        # Step 3
        if not skip_videos:
            self.download_videos(max_videos=max_videos)
        else:
            logger.info("[Step 3] Skipped video download")

        # Step 4
        if not skip_extract:
            self.extract_frames(max_videos=max_videos)
        else:
            logger.info("[Step 4] Skipped frame extraction")

        # Step 5
        if not skip_train:
            self.train(epochs=epochs, batch_size=batch_size, **train_kwargs)
        else:
            logger.info("[Step 5] Skipped training")

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        return True


# ---------------------------------------------------------------------------
# Resolve class
# ---------------------------------------------------------------------------
def _resolve_class(args):
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


def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO-ACT Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --class-name kiss
  python pipeline.py --class-id 30 --max-videos 10
  python pipeline.py --class-name kiss --skip-download --skip-extract
        """,
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--class-id", type=int, help="AVA action class ID (1-80)")
    g.add_argument("--class-name", type=str, help="Action name (e.g. kiss, hug)")

    p.add_argument("--annotation-dir", default="data/annotations")
    p.add_argument("--video-dir", default="data/raw_videos")
    p.add_argument("--training-root", default="training_data")
    p.add_argument("--checkpoint-dir", default="models/checkpoints")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--source", default="auto", choices=["mirror", "auto"],
                    help="Video source: mirror=S3 only, auto=S3+YouTube")
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--img-size", type=int, default=416)
    p.add_argument("--backbone", default="yolov8m.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-filter", action="store_true")
    p.add_argument("--skip-videos", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    class_id, class_name = _resolve_class(args)
    logger.info(f"Target class: {class_id} — {class_name}")

    pipeline = Pipeline(
        class_id=class_id,
        class_name=class_name,
        annotation_dir=args.annotation_dir,
        video_dir=args.video_dir,
        training_root=args.training_root,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        source=args.source,
    )

    ok = pipeline.run(
        skip_download=args.skip_download,
        skip_filter=args.skip_filter,
        skip_videos=args.skip_videos,
        skip_extract=args.skip_extract,
        skip_train=args.skip_train,
        max_videos=args.max_videos,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        backbone=args.backbone,
        device=args.device,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
