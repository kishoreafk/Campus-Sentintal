# YOLO-ACT — Real-time Action Detection on AVA Dataset

End-to-end pipeline: **download annotations → filter class → download videos → extract frames → train model**.

---

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| **Python 3.10+** | Runtime | <https://python.org> |
| **ffmpeg** | Video decoding / frame extraction | Windows: <https://ffmpeg.org/download.html> (add `bin/` to PATH) · Linux: `sudo apt install ffmpeg` |
| **aria2** *(optional)* | Parallel download acceleration | Windows: <https://github.com/aria2/aria2/releases> · Linux: `sudo apt install aria2` |
| **yt-dlp** *(optional)* | YouTube fallback source | `pip install yt-dlp` |
| **CUDA GPU** *(recommended)* | Training acceleration | <https://developer.nvidia.com/cuda-downloads> |

---

## 1 — Clone & Install

```bash
git clone https://github.com/kishoreafk/Campus-Sentintal.git
cd Campus-Sentintal

# Create virtual environment
python -m venv .venv

# Activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Verify CUDA (optional):

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## 2 — Run the Full Pipeline (one command)

The easiest way — runs all 5 steps automatically:

```bash
# Train on "kiss" action class (class ID 30)
python pipeline.py --class-name kiss

# Or use class ID directly
python pipeline.py --class-id 30

# Limit downloads for a quick test
python pipeline.py --class-name kiss --max-videos 5 --epochs 10
```

### Pipeline flags

| Flag | Default | Description |
|------|---------|-------------|
| `--class-name` | — | Action name (e.g. `kiss`, `hug`, `walk`) |
| `--class-id` | — | AVA class ID (1–80) |
| `--max-videos` | all | Limit number of videos per split |
| `--source` | `auto` | `mirror` = S3 only; `auto` = S3 + YouTube fallback |
| `--skip-download` | off | Skip annotation download (step 1) |
| `--skip-filter` | off | Skip annotation filtering (step 2) |
| `--skip-videos` | off | Skip video download (step 3) |
| `--skip-extract` | off | Skip frame extraction (step 4) |
| `--skip-train` | off | Skip model training (step 5) |
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 4 | Batch size |
| `--backbone` | `yolov8m.pt` | YOLOv8 backbone variant |
| `--device` | `cuda` | `cuda` or `cpu` |

---

## 3 — Step-by-Step Execution

If you prefer running each step manually:

### Step 1: Download AVA Annotations

Downloads `ava_v2.2.zip` from the S3 mirror and extracts the CSV files.

```bash
python scripts/download_annotations.py
# Output: data/annotations/ava_train_v2.2.csv, ava_val_v2.2.csv, etc.
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `data/annotations` | Where to save annotation files |
| `--force` | off | Re-download even if files exist |

### Step 2: Filter Annotations for a Class

Reads the full AVA CSVs, keeps only rows for your target class, and writes filtered outputs + video ID lists.

```bash
python scripts/filter_annotations.py --class-name kiss
# Or by ID:
python scripts/filter_annotations.py --class-id 30
```

**Output structure:**

```
training_data/class_30_kiss/
├── filtered_annotations/
│   ├── ava_train_v2.2_filtered.csv
│   └── ava_val_v2.2_filtered.csv
├── video_ids/
│   ├── train_video_ids.txt        # one video ID per line
│   └── val_video_ids.txt
└── manifests/
    ├── train_manifest.csv         # video_id, video_path, label_id, label_name
    └── val_manifest.csv
```

| Flag | Default | Description |
|------|---------|-------------|
| `--annotation-dir` | `data/annotations` | Location of AVA CSVs |
| `--output-root` | `training_data` | Base directory for class output |
| `--class-name` | — | Action name |
| `--class-id` | — | AVA class ID |

### Step 3: Download Videos

Downloads videos from the S3 AVA mirror (`.mkv` first), with optional YouTube/yt-dlp fallback. Hard-links downloaded files into the class directory to save disk space.

```bash
# Download all videos for a class (train + val splits)
python scripts/download_videos_v2.py --class-dir training_data/class_30_kiss

# Limit to 5 videos for testing
python scripts/download_videos_v2.py --class-dir training_data/class_30_kiss --max-videos 5

# S3 mirror only (no YouTube fallback)
python scripts/download_videos_v2.py --class-dir training_data/class_30_kiss --source mirror

# Download from a specific video-ID list
python scripts/download_videos_v2.py --video-ids training_data/class_30_kiss/video_ids/train_video_ids.txt
```

**Console output** — clean one-liner per video:

```
============================================================
  Download [train] — class_30_kiss  (50 videos)
============================================================
  [ 1/50]  OK    -ZFgsrolSxo  (S3 mirror, 366.9 MB)
  [ 2/50]  SKIP  -OyDO1g74vc  (exists, 212.9 MB)
  [ 3/50]  FAIL  0f39OWEqJ24  (all sources exhausted (S3+YouTube))
────────────────────────────────────────────────────────────
  Summary: 1 downloaded, 1 skipped, 1 failed  (total 3)
────────────────────────────────────────────────────────────
  Failed video IDs logged to: logs/failed_20260306_141523.json
  Full log: logs/download_20260306_141523.log
```

**Failure logs** are saved as JSON in `logs/` with timestamps:

```json
{
  "run_timestamp": "20260306_141523",
  "split": "train",
  "class_info": "class_30_kiss",
  "summary": { "total": 50, "downloaded": 35, "skipped": 10, "failed": 5 },
  "failed_videos": [
    { "video_id": "0f39OWEqJ24", "error": "all sources exhausted (S3+YouTube)", "timestamp": "2026-03-06T14:15:30" }
  ]
}
```

| Flag | Default | Description |
|------|---------|-------------|
| `--class-dir` | — | Class directory from step 2 |
| `--video-ids` | — | Path to a video-ID text file (alternative to `--class-dir`) |
| `--verify-links` | — | Verify/repair hard links without downloading |
| `--video-dir` | `data/raw_videos` | Shared directory for downloaded videos |
| `--source` | `auto` | `mirror` = S3 only; `auto` = S3 + YouTube |
| `--max-videos` | all | Limit number of videos |
| `--no-skip-existing` | off | Force re-download |
| `--delay` | 0.5 | Seconds between downloads (rate-limiting) |

### Step 3b: Verify & Repair Links

If you downloaded videos separately or want to re-link and regenerate manifests:

```bash
python scripts/download_videos_v2.py --verify-links training_data/class_30_kiss
```

This will:
- Scan `data/raw_videos/` for any videos matching the class's ID lists
- Hard-link them into `training_data/class_30_kiss/videos/{train,val}/`
- Regenerate manifests with correct file extensions

### Step 4: Extract Frames

Extracts frames from the AVA annotated temporal window (seconds 902–1798).

```bash
python scripts/extract_frames.py \
  --video-dir training_data/class_30_kiss/videos/train \
  --frame-dir training_data/class_30_kiss/frames \
  --annotation-dir data/annotations \
  --fps 30 --img-size 416 \
  --classes kiss
```

| Flag | Default | Description |
|------|---------|-------------|
| `--fps` | 30 | Frames per second to extract |
| `--img-size` | 416 | Output frame resolution |
| `--max-videos` | all | Limit for testing |
| `--no-decord` | off | Use ffmpeg instead of Decord |

### Step 5: Train the Model

```bash
python scripts/train_yolo_act.py \
  --frame-dir training_data/class_30_kiss/frames \
  --checkpoint-dir models/checkpoints \
  --epochs 50 --batch-size 4 \
  --classes kiss
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 100 | Number of epochs |
| `--batch-size` | 4 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--backbone` | `yolov8m.pt` | Backbone variant (`n`/`s`/`m`/`l`/`x`) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--no-amp` | off | Disable mixed-precision training |
| `--resume` | — | Path to checkpoint to resume from |
| `--auto-resume` | on | Auto-resume from `last_model.pth` |

---

## Project Structure

```
Campus-Sentintal/
├── pipeline.py                        # One-command pipeline orchestrator
├── requirements.txt
├── config/
│   ├── ava_classes.py                 # AVA 80 class IDs & mappings
│   └── training_config.py            # Hyperparameters & model config
├── scripts/
│   ├── download_annotations.py       # Step 1: download AVA annotation zip
│   ├── filter_annotations.py         # Step 2: filter by class, generate manifests
│   ├── download_videos_v2.py         # Step 3: S3 mirror-first video downloader
│   ├── extract_frames.py             # Step 4: video → frame extraction
│   ├── train_yolo_act.py             # Step 5: model training
│   ├── evaluate.py                   # AVA mAP evaluation
│   ├── download_orchestrator.py      # (legacy) multi-source downloader
│   ├── download_videos.py            # (legacy) class-filtered downloader
│   └── validate_dataset.py           # Dataset integrity checker
├── models/
│   ├── yolo_act.py                   # YOLO-ACT: YOLOv8 backbone + temporal head
│   └── checkpoints/                  # Saved model weights
├── data/
│   ├── annotations/                  # AVA v2.2 CSVs (auto-downloaded)
│   └── raw_videos/                   # Shared video store (gitignored)
├── training_data/
│   └── class_<id>_<name>/            # Per-class output from filter + download
│       ├── filtered_annotations/
│       ├── video_ids/
│       ├── manifests/
│       ├── videos/{train,val}/       # Hard-linked from raw_videos (gitignored)
│       └── frames/                   # Extracted frames
├── logs/                             # Download & training logs (gitignored)
└── tests/
    ├── test_yolo_act.py              # 22 unit tests
    └── test_download_orchestrator.py
```

---

## Download Sources

Videos are downloaded in this priority order:

| Priority | Source | Format | Notes |
|----------|--------|--------|-------|
| 1 | **S3 AVA mirror** | `.mkv` → `.mp4` → `.webm` | `s3.amazonaws.com/ava-dataset/trainval/<id>.mkv` |
| 2 | **YouTube** (yt-dlp) | `.mp4` | Only when `--source auto`; trimmed to 15:00–30:01 with ffmpeg |

Downloaded videos are stored once in `data/raw_videos/` and **hard-linked** into `training_data/class_*/videos/{train,val}/` to avoid duplication.

---

## Logging

| File | Location | Contents |
|------|----------|----------|
| Download log | `logs/download_<timestamp>.log` | Full debug-level download trace |
| Failure log | `logs/failed_<timestamp>.json` | Structured list of failed video IDs with class info |

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test class
python -m pytest tests/test_yolo_act.py::TestModelArchitecture -v
```

Expected: **22 passed** (~4 s).

---

## AVA Action Classes

80 classes total. Common ones:

| ID | Action | ID | Action |
|----|--------|----|--------|
| 9 | run/jog | 10 | sit |
| 11 | stand | 13 | walk |
| 20 | drink | 21 | eat |
| 27 | hug | 29 | kick |
| 30 | kiss | 42 | sit down |
| 43 | stand up | 45 | throw |
| 53 | fight | 3 | dance |

Full list: [config/ava_classes.py](config/ava_classes.py)

---

## Backbone Variants

| Variant | Params | Speed | Use case |
|---------|--------|-------|----------|
| `yolov8n.pt` | ~3 M | Fastest | Testing / low VRAM |
| `yolov8s.pt` | ~11 M | Fast | Prototyping |
| `yolov8m.pt` | ~25 M | Medium | **Default — balanced** |
| `yolov8l.pt` | ~43 M | Slow | High accuracy |
| `yolov8x.pt` | ~68 M | Slowest | Maximum accuracy |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: cv2` | `pip install opencv-python-headless` |
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| `ffmpeg not found` | Install ffmpeg and add to `PATH` |
| CUDA out of memory | `--batch-size 2 --backbone yolov8n.pt` |
| Training on CPU | `--device cpu --no-amp` |
| Empty frame directory | Run frame extraction first (step 4) |
| Videos not linked to class dir | `python scripts/download_videos_v2.py --verify-links training_data/class_30_kiss` |
| Manifest shows wrong extension | Re-run `--verify-links` to regenerate manifests |

---

## References

- [AVA Dataset](https://research.google.com/ava/) — Google Research
- [AVA S3 Mirror](https://s3.amazonaws.com/ava-dataset/) — Primary download source
- [YOLOv8 (Ultralytics)](https://docs.ultralytics.com/) — Backbone
- [ActivityNet](http://activity-net.org/) — Benchmark
