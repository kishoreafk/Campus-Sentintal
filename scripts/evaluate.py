#!/usr/bin/env python3
"""
AVA Evaluation Script

Official AVA evaluation using ActivityNet metrics.
Computes frame-level mAP at IoU threshold 0.5.

Format required:
- predictions: [video_id, timestamp, x1, y1, x2, y2, action_id, score]
- groundtruths: [video_id, timestamp, x1, y1, x2, y2, action_id]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / max(union, 1e-6)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC-style AP computation"""
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Compute AP
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def compute_ava_map(
    predictions: List[Tuple],
    groundtruths: List[Tuple],
    iou_threshold: float = 0.5
) -> Tuple[float, Dict]:
    """
    Compute AVA mean Average Precision
    
    Args:
        predictions: List of (video_id, timestamp, x1, y1, x2, y2, action_id, score)
        groundtruths: List of (video_id, timestamp, x1, y1, x2, y2, action_id)
        iou_threshold: IoU threshold for matching
    
    Returns:
        (mAP, per-class AP)
    """
    
    # Group by action class
    ap_per_class = {}
    classes = set(g[6] for g in groundtruths)
    
    for cls in classes:
        cls_preds = [p for p in predictions if p[6] == cls]
        cls_gts = [g for g in groundtruths if g[6] == cls]
        
        if not cls_preds or not cls_gts:
            ap_per_class[cls] = 0.0
            continue
        
        # Sort predictions by score descending
        cls_preds.sort(key=lambda x: -x[7])
        
        # Group GT by (video_id, timestamp)
        gt_map = {}
        for g in cls_gts:
            key = (g[0], g[1])
            if key not in gt_map:
                gt_map[key] = []
            gt_map[key].append(g[2:6])
        
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        matched = {key: set() for key in gt_map.keys()}
        
        for i, pred in enumerate(cls_preds):
            key = (pred[0], pred[1])
            pred_box = pred[2:6]
            
            if key not in gt_map:
                fp[i] = 1
                continue
            
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_map[key]):
                if j in matched[key]:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[i] = 1
                matched[key].add(best_gt_idx)
            else:
                fp[i] = 1
        
        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / max(len(cls_gts), 1)
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-6)
        
        ap = compute_ap(recall, precision)
        ap_per_class[cls] = ap
    
    mAP = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
    
    return mAP, ap_per_class


def load_predictions(pred_file: str) -> List[Tuple]:
    """Load predictions from CSV file"""
    predictions = []
    
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                predictions.append((
                    parts[0],  # video_id
                    int(parts[1]),  # timestamp
                    float(parts[2]),  # x1
                    float(parts[3]),  # y1
                    float(parts[4]),  # x2
                    float(parts[5]),  # y2
                    int(parts[6]),  # action_id
                    float(parts[7])  # score
                ))
    
    return predictions


def load_groundtruths(gt_file: str) -> List[Tuple]:
    """Load ground truths from CSV file"""
    groundtruths = []
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                groundtruths.append((
                    parts[0],  # video_id
                    int(parts[1]),  # timestamp
                    float(parts[2]),  # x1
                    float(parts[3]),  # y1
                    float(parts[4]),  # x2
                    float(parts[5]),  # y2
                    int(parts[6])  # action_id
                ))
    
    return groundtruths


def evaluate_model(
    predictions_file: str,
    groundtruth_file: str,
    iou_threshold: float = 0.5
) -> Dict:
    """Run evaluation"""
    
    logger.info(f"Loading predictions from: {predictions_file}")
    predictions = load_predictions(predictions_file)
    logger.info(f"Loaded {len(predictions)} predictions")
    
    logger.info(f"Loading groundtruths from: {groundtruth_file}")
    groundtruths = load_groundtruths(groundtruth_file)
    logger.info(f"Loaded {len(groundtruths)} groundtruths")
    
    # Compute mAP
    mAP, ap_per_class = compute_ava_map(predictions, groundtruths, iou_threshold)
    
    return {
        "mAP": mAP,
        "per_class_ap": ap_per_class,
        "num_predictions": len(predictions),
        "num_groundtruths": len(groundtruths)
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AVA predictions")
    
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions CSV'
    )
    
    parser.add_argument(
        '--groundtruth',
        type=str,
        required=True,
        help='Path to groundtruth CSV'
    )
    
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold (default: 0.5)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    results = evaluate_model(
        args.predictions,
        args.groundtruth,
        args.iou_threshold
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"mAP@{args.iou_threshold}: {results['mAP']:.4f}")
    print(f"Predictions: {results['num_predictions']}")
    print(f"Groundtruths: {results['num_groundtruths']}")
    
    print("\nPer-class AP:")
    for cls, ap in sorted(results['per_class_ap'].items()):
        print(f"  Class {cls}: {ap:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
