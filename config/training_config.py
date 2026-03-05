"""
Training Configuration for YOLO-ACT with AVA Dataset

Includes class weights for handling AVA's extreme class imbalance.
"""

import os

# =====================
# Dataset Configuration
# =====================
DATASET_CONFIG = {
    "name": "AVA Dataset",
    "num_classes": 80,
    "selected_classes": None,
    
    # AVA temporal window
    "annotation_start": 902,
    "annotation_end": 1798,
    "clip_duration": 901,  # seconds
    
    # Paths
    "data_root": "data",
    "video_dir": "data/raw_videos",
    "frame_dir": "data/frames",
    "annotation_dir": "data/annotations",
    "train_csv": "data/annotations/ava_train_v2.2.csv",
    "val_csv": "data/annotations/ava_val_v2.2.csv",
    "proposals_train": "data/ava/proposals/ava_dense_proposals_train.FAIR.recall_93.9.pkl",
    "proposals_val": "data/ava/proposals/ava_dense_proposals_val.FAIR.recall_93.9.pkl",
    
    # Frame extraction
    "target_fps": 30,
    "frame_interval": 1,
    "img_size": 224,
}

# =====================
# Class Weights for Imbalance
# AVA has extreme class imbalance (e.g., "stand" appears 100x more than "kiss")
# =====================
CLASS_WEIGHTS = {
    # Common actions (lower weight)
    1: 0.1,    # bend/bow
    2: 0.1,    # crouch/kneel
    9: 0.1,    # run/jog
    10: 0.1,   # sit
    11: 0.1,   # stand
    13: 0.1,   # walk
    40: 0.1,   # ride
    42: 0.1,   # sit down
    43: 0.1,   # stand up
    
    # Moderate actions
    3: 0.5,    # dance
    6: 0.5,    # jump/leap
    12: 0.5,   # swim
    20: 0.5,   # drink
    21: 0.5,   # eat
    39: 0.5,   # read
    52: 0.5,   # texting
    
    # Rare actions (higher weight to compensate)
    27: 2.0,    # hug (a person)
    28: 2.0,    # kick (an object)
    29: 2.0,    # kick (a person)
    30: 5.0,   # kiss
    53: 2.0,   # fight (with person)
    54: 2.0,   # hand shake
    55: 2.0,   # high five
    59: 2.0,   # wave
    70: 5.0,   # smell
    71: 5.0,   # taste
    72: 5.0,   # throat cut
}

# Default weight for classes not in the dictionary
DEFAULT_CLASS_WEIGHT = 1.0


def get_class_weights(num_classes: int = 80) -> dict:
    """Get class weights for all 80 classes"""
    weights = {}
    for i in range(1, num_classes + 1):
        weights[i] = CLASS_WEIGHTS.get(i, DEFAULT_CLASS_WEIGHT)
    return weights


# =====================
# Model Configuration
# =====================
MODEL_CONFIG = {
    "name": "YOLO-ACT",
    "backbone": "yolov8m",
    "pretrained": True,
    "num_classes": 80,
    
    # YOLO-ACT specific
    "clip_len": 32,  # Number of frames per clip
    "temporal_stride": 2,  # Temporal stride between frames
    
    # Architecture
    "use_temporal_module": True,
    "use_spatial_module": True,
}

# =====================
# Training Hyperparameters
# =====================
TRAINING_CONFIG = {
    # Batch and epochs
    "batch_size": 4,
    "num_epochs": 50,
    "accumulation_steps": 4,
    
    # Optimizer
    "optimizer": "adamw",
    "backbone_lr": 1e-5,    # Lower LR for pretrained backbone
    "head_lr": 1e-3,        # Higher LR for new head
    "weight_decay": 1e-4,
    "momentum": 0.9,
    
    # Learning rate scheduler
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    
    # Loss weights
    "cls_loss_weight": 1.0,
    "bbox_loss_weight": 5.0,
    "action_loss_weight": 1.0,
    
    # Data loading
    "num_workers": 8,
    "pin_memory": True,
    
    # Mixed precision training
    "use_amp": True,
    
    # Early stopping
    "patience": 15,
    "min_delta": 0.001,
    
    # Checkpointing
    "save_interval": 5,
    "keep_best": True,
    
    # Transfer learning
    "freeze_backbone_epochs": 5,
}

# =====================
# Paths
# =====================
PATHS = {
    "project_root": ".",
    "config_dir": "config",
    "scripts_dir": "scripts",
    "data_dir": "data",
    "models_dir": "models",
    "checkpoint_dir": "models/checkpoints",
    "log_dir": "logs",
    "tensorboard_dir": "logs/tensorboard",
}

# =====================
# Hardware Configuration
# =====================
HARDWARE_CONFIG = {
    "device": "cuda",
    "cuda_device": 0,
    "use_multi_gpu": False,
    "distributed": False,
}


def get_config():
    """Get complete configuration"""
    return {
        "dataset": DATASET_CONFIG,
        "model": MODEL_CONFIG,
        "training": TRAINING_CONFIG,
        "paths": PATHS,
        "hardware": HARDWARE_CONFIG,
        "class_weights": get_class_weights(),
    }


def save_config(config, path):
    """Save configuration to file"""
    import json
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(path):
    """Load configuration from file"""
    import json
    with open(path, 'r') as f:
        return json.load(f)


# Default selected classes for AVA actions (user request: kiss, hug, walk, run, sit, jump, fight)
DEFAULT_SELECTED_ACTIONS = [
    "kiss", "hug", "walk", "run", "sit", "jump", "fight",
    "dance", "eat", "drink", "stand", "lie down", "kick",
    "throw", "catch", "wave", "hand shake", "high five"
]


if __name__ == "__main__":
    print("YOLO-ACT Training Configuration")
    print("=" * 50)
    print(f"Model: {MODEL_CONFIG['name']}")
    print(f"Backbone: {MODEL_CONFIG['backbone']}")
    print(f"Clip Length: {MODEL_CONFIG['clip_len']}")
    print(f"Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"Learning Rate: {TRAINING_CONFIG['head_lr']}")
    print(f"Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"Device: {HARDWARE_CONFIG['device']}")
    
    print("\nClass Weights (sample):")
    for cls in [30, 27, 13, 9, 10, 6, 53]:  # kiss, hug, walk, run, sit, jump, fight
        print(f"  Class {cls}: {CLASS_WEIGHTS.get(cls, DEFAULT_CLASS_WEIGHT)}")
