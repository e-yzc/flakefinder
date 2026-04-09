from __future__ import annotations

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNN


def build_maskrcnn(size: str = "small", num_classes: int = 2) -> MaskRCNN:
    if size not in {"small", "large"}:
        raise ValueError(f"Unsupported model size: {size}")

    if size == "small":
        model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            trainable_backbone_layers=3,
            min_size=512,
            max_size=768,
            box_detections_per_img=100,
        )
    else:
        model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            trainable_backbone_layers=5,
            min_size=800,
            max_size=1333,
            box_detections_per_img=200,
        )

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels

    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, num_classes)

    return model


def build_model(family: str = "maskrcnn", size: str = "small", num_classes: int = 2) -> torch.nn.Module:
    if family != "maskrcnn":
        raise ValueError(f"Only maskrcnn is supported, got: {family}")
    return build_maskrcnn(size=size, num_classes=num_classes)
