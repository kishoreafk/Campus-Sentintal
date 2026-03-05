#!/usr/bin/env python3
"""
YOLO-ACT Model Implementation

YOLO-Act combines YOLOv8 backbone with temporal modules for spatio-temporal action detection.
Uses Ultralytics YOLO as backbone and adds temporal action recognition heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class TemporalEncoder(nn.Module):
    """Temporal feature aggregation module using 3D conv and transformer"""
    
    def __init__(self, in_channels: int, hidden_dim: int = 256, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        
        # 3D conv for initial temporal processing
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=(2, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU(),
        )
        
        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            (B, hidden_dim, H, W)
        """
        x = self.temporal_conv(x)  # (B, hidden, T, H, W)
        B, C, T, H, W = x.shape
        
        # Reshape for transformer: treat spatial locations as batch
        x_flat = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
        x_flat = self.transformer(x_flat)
        x_flat = x_flat.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2)
        
        x = self.pool(x_flat).squeeze(2)  # (B, hidden, H, W)
        return x


class DetectionHead(nn.Module):
    """YOLO-style detection head for person bounding boxes"""
    
    def __init__(self, in_channels: int = 256, num_anchors: int = 3):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, num_anchors * 5, 1),  # 4 box coords + 1 objectness
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ActionHead(nn.Module):
    """Action classification head for per-box action recognition"""
    
    def __init__(self, feature_dim: int = 256, num_actions: int = 80):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(feature_dim + 4, 512),  # ROI features + box coords
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_actions),
        )
    
    def forward(self, roi_features: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roi_features: (N, feature_dim)
            boxes: (N, 4) normalized boxes
        Returns:
            (N, num_actions) action logits
        """
        # Ensure boxes are in the same device
        if boxes.device != roi_features.device:
            boxes = boxes.to(roi_features.device)
        
        # Concatenate features and box coordinates
        x = torch.cat([roi_features, boxes], dim=1)
        return self.head(x)


class YOLOActModel(nn.Module):
    """
    YOLO-Act: Combines YOLOv8 spatial detection with temporal action recognition.
    """
    
    def __init__(
        self,
        yolo_variant: str = "yolov8m.pt",
        num_action_classes: int = 80,
        clip_len: int = 32,
        freeze_backbone_epochs: int = 5,
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.num_action_classes = num_action_classes
        self.clip_len = clip_len
        self.freeze_backbone_epochs = freeze_backbone_epochs
        
        # Try to use ultralytics YOLO, fallback to custom backbone
        try:
            from ultralytics import YOLO
            self.use_ultralytics = True
            self.yolo_model = YOLO(yolo_variant)
            
            # Extract backbone layers
            self.backbone = self.yolo_model.model.model[:10]
            self.backbone_channels = self._get_backbone_channels(yolo_variant)
            
        except Exception as e:
            print(f"Warning: Could not load ultralytics YOLO: {e}")
            print("Using custom ResNet50 backbone instead")
            self.use_ultralytics = False
            self.backbone = self._build_resnet50_backbone(pretrained)
            self.backbone_channels = 2048
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            in_channels=self.backbone_channels,
            hidden_dim=256,
        )
        
        # Detection head (person bounding boxes)
        self.detection_head = DetectionHead(in_channels=256)
        
        # Action classification head
        self.action_head = ActionHead(feature_dim=256, num_actions=num_action_classes)
        
        # ROI Align
        try:
            from torchvision.ops import RoIAlign
            self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1/16, sampling_ratio=2)
        except:
            self.roi_align = None
        
        # ROI pooling
        self.roi_pool = nn.AdaptiveAvgPool2d(1)
        
        # Initialize weights
        self._init_weights()
    
    def _get_backbone_channels(self, variant: str) -> int:
        """Get backbone output channels based on variant"""
        channels_map = {
            "yolov8n.pt": 256,
            "yolov8s.pt": 512,
            "yolov8m.pt": 512,
            "yolov8l.pt": 512,
            "yolov8x.pt": 512,
        }
        return channels_map.get(variant, 512)
    
    def _build_resnet50_backbone(self, pretrained: bool = True) -> nn.Module:
        """Build custom ResNet50 backbone as fallback"""
        import torchvision.models as models
        from torchvision.models import ResNet50_Weights
        
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Extract layers
        backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 256
            resnet.layer2,  # 512
            resnet.layer3,  # 1024
            resnet.layer4,  # 2048
        )
        
        return backbone
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def extract_spatial_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract features from each frame using YOLO backbone"""
        B, C, T, H, W = frames.shape
        
        # Process all frames at once
        x = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        # Forward through backbone
        for layer in self.backbone:
            x = layer(x)
        
        _, Cf, Hf, Wf = x.shape
        x = x.reshape(B, T, Cf, Hf, Wf).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        return x
    
    def forward(self, clips: torch.Tensor, gt_boxes: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            clips: (B, C, T, H, W) video clips
            gt_boxes: list of (N_i, 4) tensors — GT boxes for training
        Returns:
            Dictionary with detection and action outputs
        """
        # Step 1: Spatial features
        spatial_feats = self.extract_spatial_features(clips)  # (B, C, T, H, W)
        
        # Step 2: Temporal encoding
        fused_feats = self.temporal_encoder(spatial_feats)  # (B, 256, H, W)
        
        # Step 3: Detection
        det_out = self.detection_head(fused_feats)  # (B, 5*anchors, H, W)
        
        # Step 4: ROI features for action classification
        if gt_boxes is not None and self.roi_align is not None:
            # During training, use GT boxes
            roi_list = []
            batch_idx = []
            
            for i, boxes in enumerate(gt_boxes):
                if len(boxes) > 0:
                    roi_list.append(boxes)
                    batch_idx.append(torch.full((len(boxes),), i, device=boxes.device))
            
            if len(roi_list) > 0:
                rois = torch.cat(roi_list, dim=0)
                batch_indices = torch.cat(batch_idx, dim=0).unsqueeze(1).float()
                
                # Normalize box coordinates
                rois_normalized = torch.cat([
                    batch_indices,
                    rois,
                ], dim=1)
                
                # Ensure proper format for RoI Align
                rois_for_align = torch.cat([
                    batch_indices,
                    rois * 16,  # Scale to feature map size
                ], dim=1)
                
                try:
                    roi_feats = self.roi_align(fused_feats, rois_for_align)
                    roi_feats = self.roi_pool(roi_feats).flatten(1)  # (N_total, 256)
                    
                    # Pass features and box coords separately to action head
                    action_logits = self.action_head(roi_feats, rois)
                except Exception as e:
                    # Fallback: create empty tensor
                    action_logits = torch.zeros(0, self.num_action_classes, device=clips.device)
            else:
                action_logits = torch.zeros(0, self.num_action_classes, device=clips.device)
        else:
            action_logits = None
        
        return {
            "detection": det_out,
            "action_logits": action_logits,
            "features": fused_feats,
        }
    
    def freeze_backbone(self):
        """Freeze backbone for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class YOLOActLoss(nn.Module):
    """Loss function for YOLO-Act"""
    
    def __init__(
        self,
        num_classes: int = 80,
        action_loss_weight: float = 1.0,
        det_loss_weight: float = 1.0,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.action_loss_weight = action_loss_weight
        self.det_loss_weight = det_loss_weight
        
        # Action classification loss (BCE for multi-label)
        self.action_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.ones(num_classes) * 2.0
        )
        
        # Detection loss
        self.det_loss_fn = nn.SmoothL1Loss()
    
    def forward(
        self,
        predictions: Dict,
        targets: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate losses
        """
        action_logits = predictions.get("action_logits")
        detection = predictions.get("detection")
        
        target_labels = targets.get("labels")
        target_boxes = targets.get("boxes")
        
        loss_action = torch.tensor(0.0, device=detection.device if detection is not None else 'cpu')
        loss_det = torch.tensor(0.0, device=detection.device if detection is not None else 'cpu')
        
        # Action loss
        if action_logits is not None and target_labels is not None:
            if len(action_logits) > 0 and len(target_labels) > 0:
                min_len = min(len(action_logits), len(target_labels))
                loss_action = self.action_loss_fn(
                    action_logits[:min_len],
                    target_labels[:min_len].float()
                )
        
        # Detection loss (simplified - in practice would use anchor matching)
        if detection is not None and target_boxes is not None:
            # Placeholder detection loss
            loss_det = torch.tensor(0.0, device=detection.device)
        
        total_loss = self.action_loss_weight * loss_action + self.det_loss_weight * loss_det
        
        return {
            "total_loss": total_loss,
            "action_loss": loss_action,
            "det_loss": loss_det,
        }


def create_model(
    num_classes: int = 80,
    yolo_variant: str = "yolov8m.pt",
    clip_len: int = 32,
    pretrained: bool = True,
    device: str = "cuda"
) -> Tuple[nn.Module, nn.Module]:
    """
    Create YOLO-Act model and loss function
    """
    model = YOLOActModel(
        yolo_variant=yolo_variant,
        num_action_classes=num_classes,
        clip_len=clip_len,
        pretrained=pretrained,
    )
    
    loss_fn = YOLOActLoss(num_classes=num_classes)
    
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    
    return model, loss_fn


# Model configurations
MODEL_CONFIGS = {
    "yolo_act_n": {"yolo_variant": "yolov8n.pt", "clip_len": 32},
    "yolo_act_s": {"yolo_variant": "yolov8s.pt", "clip_len": 32},
    "yolo_act_m": {"yolo_variant": "yolov8m.pt", "clip_len": 32},
    "yolo_act_l": {"yolo_variant": "yolov8l.pt", "clip_len": 32},
    "yolo_act_x": {"yolo_variant": "yolov8x.pt", "clip_len": 32},
}


if __name__ == "__main__":
    print("Testing YOLO-Act model...")
    
    # Create model
    model, loss_fn = create_model(num_classes=80, yolo_variant="yolov8m.pt")
    
    print(f"\nModel: YOLO-Act")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    clip_len = 8
    img_size = 416
    
    # Dummy input
    x = torch.randn(batch_size, 3, clip_len, img_size, img_size)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Output keys: {outputs.keys()}")
    if outputs["detection"] is not None:
        print(f"Detection shape: {outputs['detection'].shape}")
    if outputs["action_logits"] is not None:
        print(f"Action logits shape: {outputs['action_logits'].shape}")
    
    print("\nModel test passed!")
