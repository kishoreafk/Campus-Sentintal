# YOLO-ACT Training Pipeline with AVA Dataset

A complete pipeline to train YOLO-ACT (Real-time Spatio-Temporal Action Detection) model using the AVA (Atomic Visual Actions) dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running Tests](#running-tests)
- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
- [Training Guide](#training-guide)
- [Configuration](#configuration)
- [AVA Dataset Classes](#ava-dataset-classes)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

This pipeline implements a complete end-to-end workflow for training action detection models:

1. **Download** AVA videos with multi-source fallback (CVDF → HuggingFace → YouTube)
2. **Validate** dataset integrity (completeness, corruption, duplicates)
3. **Extract** frames from videos (AVA annotated window: 902–1798 s)
4. **Train** YOLO-ACT model (YOLOv8 backbone + temporal encoder)

## ✨ Features

- 🎬 **Multi-Source Video Download**: Downloads from CVDF/Facebook mirror, HuggingFace, or YouTube
- 🔄 **Duplicate Avoidance**: Skip already downloaded videos
- ✅ **Dataset Validation**: Comprehensive integrity checking
- ⚡ **GPU Acceleration**: Full CUDA support for training
- 📊 **Mixed Precision**: Automatic Mixed Precision (AMP) for faster training
- 🧠 **Temporal Modeling**: 3D-Conv + Transformer temporal encoder
- 💾 **Checkpoint Management**: Save best / last / encrypted checkpoints

## 📁 Project Structure

```
d:/Campus/
├── config/
│   ├── ava_classes.py           # AVA 80 class mappings & helper functions
│   └── training_config.py       # Training hyperparameters & class weights
├── scripts/
│   ├── download_orchestrator.py  # Multi-source video downloader
│   ├── validate_dataset.py       # Dataset validation
│   ├── check_existing.py         # Check for existing videos
│   ├── extract_frames.py         # Video → frame extraction (Decord/ffmpeg)
│   ├── evaluate.py               # AVA mAP evaluation
│   └── train_yolo_act.py         # Full model training script
├── models/
│   ├── yolo_act.py               # YOLO-ACT model (YOLOv8 + temporal head)
│   └── checkpoints/              # Saved model weights
├── tests/
│   ├── __init__.py
│   └── test_yolo_act.py          # Unit tests (22 tests)
├── data/
│   ├── raw_videos/               # Downloaded videos
│   ├── frames/                   # Extracted frames (train/ & val/)
│   └── annotations/              # AVA CSV annotations
├── pipeline.py                   # Main orchestration script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 💾 Requirements

### System Dependencies

| Tool | Purpose | Install |
|------|---------|---------|
| Python 3.10+ | Runtime | https://python.org |
| ffmpeg | Video decoding / frame extraction | `apt install ffmpeg` or https://ffmpeg.org |
| aria2 | Parallel download acceleration | `apt install aria2` |

On **Windows**, install ffmpeg from https://ffmpeg.org/download.html and add to `PATH`.

### Python Packages

Core packages (installed via `requirements.txt`):

```
torch>=2.1.0
torchvision>=0.16.0
ultralytics>=8.0.0
opencv-python-headless>=4.8.0
decord>=0.6.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
pyyaml>=6.0
requests>=2.31.0
pytest>=7.4.0
```

## 🔧 Installation

### 1. Clone / open the project

```bash
cd d:/Campus
```

### 2. (Recommended) Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install all Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify CUDA availability (optional but recommended)

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## 🧪 Running Tests

All unit tests live in `tests/test_yolo_act.py` and cover:

| Test Class | What is tested |
|------------|---------------|
| `TestAVAClasses` | 80 class IDs, no duplicates, reverse mapping |
| `TestTrainingConfig` | Required config fields, class weights |
| `TestModelArchitecture` | Forward pass shape with ResNet50 fallback |
| `TestYOLOActLoss` | Loss output is dict, values are scalars, no NaN |
| `TestCheckpointManagement` | Save / load / best / last / cleanup |
| `TestDataLoading` | Dataloader creation (handles empty dataset) |
| `TestArgumentParsing` | CLI args: defaults, `--resume`, `--auto-resume` |
| `TestEvaluation` | IoU computation correctness |
| `TestFrameExtraction` | FrameExtractor init, AVA temporal constants |

### Run all tests

```bash
python -m pytest tests/ -v
```

### Run a specific test class

```bash
python -m pytest tests/test_yolo_act.py::TestModelArchitecture -v
```

### Run a single test

```bash
python -m pytest tests/test_yolo_act.py::TestYOLOActLoss::test_loss_computation -v
```

### Expected output

```
22 passed, 5 warnings in ~4s
```

> **Note:** The warnings about `torch.amp.GradScaler` on CPU-only machines are expected and do not affect functionality.

---

## 🚀 Quick Start

### Option 1: Full pipeline (download → extract → train)

```bash
python pipeline.py
```

### Option 2: Specific action classes only

```bash
python pipeline.py --classes kiss,hug,walk,run,sit,jump,fight
```

### Option 3: Step-by-step

```bash
# Step 1: Download videos
python scripts/download_orchestrator.py --classes kiss,hug,walk,run

# Step 2: Validate dataset
python scripts/validate_dataset.py

# Step 3: Extract frames (AVA window 902–1798 s, 30 FPS, 224×224)
python scripts/extract_frames.py --fps 30 --img-size 224

# Step 4: Train
python scripts/train_yolo_act.py --epochs 50 --batch-size 4
```

---

## 🔧 Pipeline Components

### 1. Video Download (`scripts/download_orchestrator.py`)

Downloads AVA videos with the following priority:
1. **CVDF / Facebook S3 Mirror** — pre-cut 15-min clips (fastest, most reliable)
2. **HuggingFace Mirrors** — community-maintained mirrors
3. **YouTube** — fallback, trimmed to the 15-minute annotation window

```bash
python scripts/download_orchestrator.py --classes kiss,hug,walk,run
```

### 2. Dataset Validation (`scripts/validate_dataset.py`)

- Completeness: expected vs downloaded count
- Integrity: verifies no corrupted videos
- Duplicates: filename + content-hash based
- Annotation alignment: CSV ↔ video file

```bash
python scripts/validate_dataset.py
```

### 3. Frame Extraction (`scripts/extract_frames.py`)

Extracts only the AVA annotated window (seconds 902–1798) to save disk space.
Prefers **Decord** (10× faster) and falls back to **ffmpeg**.

```bash
# Standard extraction
python scripts/extract_frames.py

# Fast: limit to 10 videos for testing
python scripts/extract_frames.py --max-videos 10

# Custom FPS / size
python scripts/extract_frames.py --fps 30 --img-size 416

# Use ffmpeg only (no Decord)
python scripts/extract_frames.py --no-decord
```

### 4. Model Training (`scripts/train_yolo_act.py`)

Trains the YOLO-ACT model.

---

## 🏋️ Training Guide

### Basic training

```bash
python scripts/train_yolo_act.py
```

This uses the defaults from `config/training_config.py`:
- 100 epochs, batch size 4, lr 1e-4, backbone `yolov8m.pt`
- Saves `best_model.pth` and `last_model.pth` to `models/checkpoints/`

### Custom training

```bash
python scripts/train_yolo_act.py \
  --epochs 50 \
  --batch-size 8 \
  --lr 0.0001 \
  --backbone yolov8n.pt \
  --num-frames 16 \
  --img-size 224 \
  --num-workers 4
```

### Train on specific classes

```bash
python scripts/train_yolo_act.py --classes walk,run,sit,jump,fight
```

### Resume training

```bash
# Auto-resume from last checkpoint (default behaviour)
python scripts/train_yolo_act.py

# Resume from a specific checkpoint
python scripts/train_yolo_act.py --resume models/checkpoints/checkpoint_epoch_30.pth

# Resume from the best checkpoint
python scripts/train_yolo_act.py --resume-from-best
```

### CPU training (no GPU)

```bash
python scripts/train_yolo_act.py --device cpu --no-amp
```

### All training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--frame-dir` | `data/frames` | Directory of extracted frames |
| `--checkpoint-dir` | `models/checkpoints` | Where to save checkpoints |
| `--log-dir` | `logs` | Directory for logs |
| `--epochs` / `-e` | `100` | Number of training epochs |
| `--batch-size` / `-b` | `4` | Batch size |
| `--lr` | `0.0001` | Learning rate |
| `--weight-decay` | `0.0001` | Weight decay (AdamW) |
| `--img-size` | `416` | Input image resolution |
| `--num-frames` | `16` | Frames per video clip |
| `--num-workers` | `4` | DataLoader worker threads |
| `--save-interval` | `5` | Save checkpoint every N epochs |
| `--backbone` | `yolov8m.pt` | YOLOv8 backbone variant |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--no-amp` | off | Disable mixed-precision (AMP) |
| `--classes` | all 80 | Comma-separated class names to train |
| `--resume` | – | Path to checkpoint to resume from |
| `--auto-resume` | on | Auto-resume from `last_model.pth` |
| `--resume-from-best` | off | Resume from `best_model.pth` |
| `--keep-best-only` | on | Only save best model |
| `--keep-last-only` | off | Only keep the latest checkpoint |
| `--max-checkpoints` | `5` | Max regular checkpoints to retain |
| `--encrypt-checkpoints` | off | AES-encrypt checkpoints |
| `--encryption-key` | – | Key for encrypted checkpoints |

---

## ⚙️ Configuration

### Training hyperparameters (`config/training_config.py`)

```python
TRAINING_CONFIG = {
    "batch_size": 4,
    "num_epochs": 50,
    "accumulation_steps": 4,   # effective batch = 4 * 4 = 16
    "optimizer": "adamw",
    "backbone_lr": 1e-5,
    "head_lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "use_amp": True,
    "freeze_backbone_epochs": 5,
}
```

### Model (`config/training_config.py`)

```python
MODEL_CONFIG = {
    "backbone": "yolov8m",     # no .pt extension here
    "clip_len": 32,
    "temporal_stride": 2,
}
```

### Backbone variants

| Variant | Params (approx.) | Speed | Recommended for |
|---------|-----------------|-------|----------------|
| `yolov8n.pt` | ~3 M | Fastest | Testing / low-VRAM |
| `yolov8s.pt` | ~11 M | Fast | Prototyping |
| `yolov8m.pt` | ~25 M | Medium | **Default / balanced** |
| `yolov8l.pt` | ~43 M | Slow | High-accuracy |
| `yolov8x.pt` | ~68 M | Slowest | Maximum accuracy |

---

## 🏷️ AVA Dataset Classes

The model trains on 80 AVA action categories (IDs 1–80). A selection of common classes and their IDs:

| ID | Class | ID | Class |
|----|-------|----|-------|
| 9  | run/jog | 10 | sit |
| 11 | stand | 13 | walk |
| 27 | hug (a person) | 30 | kiss |
| 29 | kick (a person) | 53 | fight (with person) |
| 25 | hit (a person) | 45 | throw |
| 43 | stand up | 42 | sit down |
| 21 | eat | 20 | drink |

Use `--classes` to filter to a subset:

```bash
python scripts/train_yolo_act.py --classes "walk,run,sit,jump,hug,kiss,fight"
```

---

## 🔍 Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python-headless
```

### `ModuleNotFoundError: No module named 'ultralytics'`
```bash
pip install ultralytics
```

### `ModuleNotFoundError: No module named 'decord'`
```bash
pip install decord
# or use ffmpeg fallback:
python scripts/extract_frames.py --no-decord
```

### CUDA out of memory
Reduce batch size or use a smaller backbone:
```bash
python scripts/train_yolo_act.py --batch-size 2 --backbone yolov8n.pt
```

### Training on CPU
```bash
python scripts/train_yolo_act.py --device cpu --no-amp
```

### `ValueError: num_samples should be a positive integer`
The frame directory is empty. Run frame extraction first:
```bash
python scripts/extract_frames.py
```

### ffmpeg not found
- **Windows**: Download from https://ffmpeg.org/download.html, extract, and add the `bin/` folder to your system `PATH`.
- **Linux**: `sudo apt install ffmpeg`


## 📁 Project Structure

```
d:/Campus/
├── config/
│   ├── ava_classes.py           # AVA 80 class mappings
│   └── training_config.py       # Training hyperparameters
├── scripts/
│   ├── download_orchestrator.py  # Multi-source video downloader
│   ├── validate_dataset.py       # Dataset validation
│   ├── check_existing.py        # Check for existing videos
│   ├── extract_frames.py        # Video to frame extraction
│   └── train_yolo_act.py        # Model training script
├── models/
│   ├── yolo_act.py             # YOLO-ACT model (YOLOv8 + temporal)
│   └── checkpoints/             # Saved model weights
├── data/
│   ├── raw_videos/             # Downloaded videos
│   ├── frames/                  # Extracted frames
│   └── annotations/             # AVA annotations
├── pipeline.py                  # Main orchestration script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 💾 Installation

1. **Install system dependencies** (required):
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg aria2
   
   # Windows (install ffmpeg manually and add to PATH)
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify CUDA availability**:
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline

```bash
python pipeline.py
```

### Option 2: Custom Classes

```bash
python pipeline.py --classes kiss,hug,walk,run,sit,jump,fight
```

### Option 3: Step by Step

```bash
# Step 1: Download videos (multi-source with fallback)
python scripts/download_orchestrator.py --classes kiss,hug,walk,run

# Step 2: Validate dataset
python scripts/validate_dataset.py

# Step 3: Extract frames
python scripts/extract_frames.py

# Step 4: Train
python scripts/train_yolo_act.py --epochs 50 --batch-size 4
```

## 🔧 Pipeline Components

### 1. Video Download Orchestrator (`scripts/download_orchestrator.py`)

Multi-source video download with priority:
1. **CVDF / Facebook S3 Mirror** (Primary - pre-cut 15min clips)
2. **HuggingFace Mirrors** (Community mirrors)
3. **YouTube** (Fallback - with trimming to AVA's 15-minute segments)

```bash
python scripts/download_orchestrator.py --classes kiss,hug,walk,run
```

### 2. Dataset Validator (`scripts/validate_dataset.py`)

Comprehensive validation:
- Completeness check (expected vs downloaded)
- Video integrity (not corrupted)
- Duplicate detection (filename + content hash)
- Annotation-video alignment

```bash
python scripts/validate_dataset.py
```

### 3. Frame Extractor (`scripts/extract_frames.py`)

Extract frames from videos:

```bash
python scripts/extract_frames.py --fps 30 --img-size 416
```

### 4. Model Training (`scripts/train_yolo_act.py`)

Train YOLO-ACT model:

```bash
python scripts/train_yolo_act.py --epochs 50 --batch-size 4
```

## 📖 Usage Examples

### Training with Limited Data (Testing)

```bash
python pipeline.py --max-videos 10 --epochs 10 --batch-size 2
```

### Resume Training

```bash
python scripts/train_yolo_act.py --resume models/checkpoints/checkpoint_epoch_50.pth
```

### Custom Training Configuration

```bash
python pipeline.py \
  --classes walk,run,sit,jump \
  --epochs 50 \
  --batch-size 4 \
  --lr 0.001 \
  --backbone yolov8m.pt
```

### CPU Training (No GPU)

```bash
python pipeline.py --device cpu
```

## ⚙️ Configuration

### Training Parameters

Edit `config/training_config.py`:

```python
TRAINING_CONFIG = {
    "batch_size": 4,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "use_amp": True,
}
```

### Model Configuration

```python
MODEL_CONFIG = {
    "backbone": "yolov8m.pt",  # or yolov8s, yolov8l, yolov8x
    "num_classes": 80,
    "clip_len": 32,
}
```

## 📚 AVA Dataset Classes

The AVA dataset contains 80 action classes. Default selected classes include:

| Class ID | Action |
|----------|--------|
| 30 | kiss |
| 27 | hug |
| 13 | walk |
| 9 | run/jog |
| 10 | sit |
| 6 | jump/leap |
| 53 | fight |
| 3 | dance |
| 21 | eat |
| 20 | drink |
| 11 | stand |
| 7 | lie down |
| 29 | kick |
| 45 | throw |

Full class list available in `config/ava_classes.py`

## 🐛 Troubleshooting

### Video Download Issues

1. **CVDF S3 unavailable**: Script automatically falls back to HuggingFace
2. **YouTube videos unavailable**: Some videos may have been removed
3. **Use aria2 for faster downloads**: Already configured

### CUDA Out of Memory

Reduce batch size or image size:
```bash
python scripts/train_yolo_act.py --batch-size 2 --img-size 320
```

### Training Takes Too Long

1. Enable mixed precision: Already enabled by default
2. Reduce epochs for testing
3. Use smaller backbone: `--backbone yolov8n.pt`

### Dataset Validation Fails

- Ensure videos downloaded completely
- Run: `python scripts/validate_dataset.py --resolve-duplicates`

## 📝 Notes

- Videos are downloaded from **official mirrors** (CVDF S3, HuggingFace), NOT directly from YouTube (except as fallback)
- AVA videos are 15-minute clips extracted from movies/TV shows
- The validation script checks for corruption and duplicates
- Multi-GPU training is supported with `--multi-gpu` flag

## 📎 References

- [YOLO-ACT Paper](link-to-paper)
- [AVA Dataset](https://research.google.com/ava/)
- [CVDF Mirror](https://s3.amazonaws.com/ava-dataset/)
- [ActivityNet](http://activity-net.org/)
