from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import load_json, mask_to_box


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Manifest is not a list: {path}")
    return payload


def _read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {path}")
    return (mask > 0).astype(np.uint8)


def _image_augment(image: np.ndarray, masks: list[np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = image.shape[:2]

    if rng.random() < 0.5:
        image = image[:, ::-1].copy()
        masks = [m[:, ::-1].copy() for m in masks]

    if rng.random() < 0.2:
        image = image[::-1, :].copy()
        masks = [m[::-1, :].copy() for m in masks]

    if rng.random() < 0.45:
        gain = float(rng.uniform(0.85, 1.2))
        bias = float(rng.uniform(-12.0, 12.0))
        image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

    if rng.random() < 0.2:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    if rng.random() < 0.2:
        sigma = float(rng.uniform(2.0, 8.0))
        image = np.clip(image.astype(np.float32) + rng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)

    return image.reshape(h, w, 3), masks


def _build_target(
    instances: list[dict[str, Any]],
    masks: list[np.ndarray],
    index: int,
    image_shape: tuple[int, int],
) -> dict[str, torch.Tensor]:
    boxes: list[list[int]] = []
    filtered_masks: list[np.ndarray] = []

    for inst, mask in zip(instances, masks):
        bbox = mask_to_box(mask)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        filtered_masks.append(mask)

    if boxes:
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        masks_tensor = torch.as_tensor(np.stack(filtered_masks), dtype=torch.uint8)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
    else:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        h, w = image_shape
        masks_tensor = torch.zeros((0, h, w), dtype=torch.uint8)
        labels = torch.zeros((0,), dtype=torch.int64)
        area = torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((0,), dtype=torch.int64)

    target = {
        "boxes": boxes_tensor,
        "labels": labels,
        "masks": masks_tensor,
        "image_id": torch.tensor([index], dtype=torch.int64),
        "area": area,
        "iscrowd": iscrowd,
    }
    return target


class HybridInstanceDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | list[dict[str, Any]],
        train: bool = False,
        seed: int = 42,
    ) -> None:
        if isinstance(manifest, (str, Path)):
            self.items = load_manifest(manifest)
        else:
            self.items = manifest
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        entry = self.items[index]
        image = _read_rgb(entry["image_path"])

        instances = entry.get("instances", [])
        masks = [_read_mask(inst["mask_path"]) for inst in instances]

        if self.train:
            rng = np.random.default_rng(self.seed + index)
            image, masks = _image_augment(image, masks, rng)

        image_tensor = torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        target = _build_target(instances, masks, index, image.shape[:2])

        return image_tensor, target
