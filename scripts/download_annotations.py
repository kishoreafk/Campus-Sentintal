#!/usr/bin/env python3
"""
AVA Annotation Downloader

Downloads the official AVA v2.2 annotation archive from the S3 mirror and
extracts the core CSV files needed for training.

Source:
    https://s3.amazonaws.com/ava-dataset/annotations/ava_v2.2.zip

Outputs (into --output-dir):
    ava_train_v2.2.csv
    ava_val_v2.2.csv
    ava_train_excluded_timestamps_v2.2.csv
    ava_val_excluded_timestamps_v2.2.csv
    ava_action_list_v2.2_for_activitynet_2019.pbtxt

Usage:
    python scripts/download_annotations.py
    python scripts/download_annotations.py --output-dir data/annotations
    python scripts/download_annotations.py --force
"""

import argparse
import logging
import re
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANNOTATION_ZIP_URL = "https://s3.amazonaws.com/ava-dataset/annotations/ava_v2.2.zip"

# Files we require from the zip.
REQUIRED_FILES = [
    "ava_train_v2.2.csv",
    "ava_val_v2.2.csv",
    "ava_train_excluded_timestamps_v2.2.csv",
    "ava_val_excluded_timestamps_v2.2.csv",
    "ava_action_list_v2.2_for_activitynet_2019.pbtxt",
]

# Additional file we download separately (not in the zip).
FILE_NAMES_URL = (
    "https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _looks_like_html(path: Path) -> bool:
    """Return True if the first bytes look like an HTML error page."""
    try:
        header = path.read_bytes()[:4096].decode("utf-8", errors="ignore").lower()
    except Exception:
        return True
    return "<!doctype html" in header or "<html" in header or "<title>error" in header


def _validate_csv(path: Path) -> bool:
    """Quick sanity-check: the first data row has a YT-ID-shaped first column."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    if _looks_like_html(path):
        return False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            if len(parts) >= 7 and re.match(r"^[A-Za-z0-9_-]{8,20}$", parts[0]):
                return True
    return False


def _validate_pbtxt(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if _looks_like_html(path):
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "item {" in text and "name:" in text and "id:" in text


def _download(url: str, dest: Path, timeout: int = 300) -> bool:
    """Download *url* to *dest*. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        return dest.exists() and dest.stat().st_size > 0
    except Exception as exc:
        logger.warning(f"Download failed for {url}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
def download_annotations(output_dir: Path, force: bool = False) -> bool:
    """Download and extract AVA v2.2 annotations.

    Returns True when all required files are present and valid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- check whether we already have everything ----
    if not force:
        all_ok = True
        for name in REQUIRED_FILES:
            dest = output_dir / name
            if name.endswith(".pbtxt"):
                if not _validate_pbtxt(dest):
                    all_ok = False
                    break
            elif name.endswith(".csv"):
                # excluded-timestamp CSVs are small but valid
                if not dest.exists() or dest.stat().st_size == 0 or _looks_like_html(dest):
                    all_ok = False
                    break
        if all_ok:
            logger.info("All annotation files already present and valid — skipping download.")
            _ensure_file_names_list(output_dir)
            return True

    # ---- download the zip ----
    with tempfile.TemporaryDirectory(prefix="ava_ann_") as tmp:
        zip_path = Path(tmp) / "ava_v2.2.zip"
        logger.info(f"Downloading {ANNOTATION_ZIP_URL}")
        if not _download(ANNOTATION_ZIP_URL, zip_path):
            logger.error("Failed to download the annotation archive.")
            return False

        try:
            with zipfile.ZipFile(zip_path) as zf:
                members = set(zf.namelist())
                for name in REQUIRED_FILES:
                    if name not in members:
                        logger.warning(f"{name} is not in the zip — skipping.")
                        continue
                    zf.extract(name, path=output_dir)
                    logger.info(f"Extracted {name}")
        except zipfile.BadZipFile:
            logger.error("Downloaded zip is corrupt.")
            return False

    # ---- validate ----
    ok = True
    for name in REQUIRED_FILES:
        dest = output_dir / name
        if name.endswith(".pbtxt"):
            valid = _validate_pbtxt(dest)
        elif "excluded" in name:
            valid = dest.exists() and dest.stat().st_size > 0 and not _looks_like_html(dest)
        else:
            valid = _validate_csv(dest)

        if valid:
            logger.info(f"  ✓ {name}")
        else:
            logger.error(f"  ✗ {name} — invalid or missing")
            ok = False

    # ---- video-name list (separate file) ----
    _ensure_file_names_list(output_dir)

    return ok


def _ensure_file_names_list(output_dir: Path):
    """Download ava_file_names_trainval_v2.1.txt if not present."""
    dest = output_dir / "ava_file_names_trainval_v2.1.txt"
    if dest.exists() and dest.stat().st_size > 100 and not _looks_like_html(dest):
        return
    logger.info(f"Downloading {FILE_NAMES_URL}")
    if _download(FILE_NAMES_URL, dest):
        logger.info(f"  ✓ ava_file_names_trainval_v2.1.txt")
    else:
        logger.warning("  Could not download ava_file_names_trainval_v2.1.txt — "
                        "filter step will derive video IDs from CSVs instead.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Download AVA v2.2 annotations from S3")
    p.add_argument("--output-dir", type=str, default="data/annotations",
                    help="Directory to store annotation files")
    p.add_argument("--force", action="store_true",
                    help="Re-download even if files exist")
    return p.parse_args()


def main():
    args = parse_args()
    ok = download_annotations(Path(args.output_dir), force=args.force)
    if ok:
        logger.info("Annotation download complete.")
    else:
        logger.error("Annotation download had errors — check logs above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
