#!/usr/bin/env python3
"""
Single-video download test for the AVA orchestrator.

Downloads annotations, picks one video ID (from the 'kiss' class), and
tries to download it through the full orchestrator pipeline.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.download_orchestrator import AVADownloadOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    orchestrator = AVADownloadOrchestrator(
        video_dir="data/raw_videos",
        annotation_dir="data/annotations",
        selected_classes=["kiss"],
        max_workers=1,
        skip_existing=False,   # Force re-download for testing
    )

    # Step 1: Download annotations
    logger.info("=" * 60)
    logger.info("STEP 1: Checking annotations")
    logger.info("=" * 60)
    orchestrator.download_annotations()

    # Step 2: Get video list
    logger.info("=" * 60)
    logger.info("STEP 2: Getting video list for 'kiss' class (action_id=30)")
    logger.info("=" * 60)
    video_ids = orchestrator.get_video_list()
    logger.info(f"Found {len(video_ids)} video IDs for 'kiss' class")

    if not video_ids:
        logger.error("No video IDs found! Cannot proceed.")
        return 1

    logger.info(f"First 5 IDs: {video_ids[:5]}")

    # Step 3: Try downloading the FIRST video only
    test_video = video_ids[0]
    logger.info("=" * 60)
    logger.info(f"STEP 3: Downloading single video: {test_video}")
    logger.info("=" * 60)

    # Show the CVDF candidate URLs that will be tried
    candidates = orchestrator._build_cvdf_candidates(test_video)
    logger.info(f"CVDF candidate URLs ({len(candidates)}):")
    for url in candidates[:6]:
        logger.info(f"  {url}")

    vid, status = orchestrator.download_single(test_video)
    logger.info(f"Result: {vid} -> {status}")

    # Step 4: Verify
    logger.info("=" * 60)
    logger.info("STEP 4: Verification")
    logger.info("=" * 60)
    if orchestrator.video_exists(test_video):
        from pathlib import Path
        for ext in [".mkv", ".mp4", ".webm", ".avi"]:
            p = Path("data/raw_videos") / f"{test_video}{ext}"
            if p.exists():
                size_mb = p.stat().st_size / (1024 * 1024)
                logger.info(f"SUCCESS: {p.name} ({size_mb:.1f} MB)")
                break
    else:
        logger.error(f"FAILED: Video {test_video} was not downloaded")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
