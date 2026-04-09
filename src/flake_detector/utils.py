from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import json
import random

import cv2
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(override: str | None = None) -> torch.device:
    if override:
        lowered = override.lower()
        if lowered == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if lowered == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if lowered == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_json(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, target)


def mask_to_box(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return [x1, y1, x2, y2]


def collate_fn(batch: list[tuple[Any, Any]]) -> tuple[list[Any], list[Any]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().clamp(0, 1).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255.0).astype(np.uint8)
    return arr


def draw_overlay(
    image: np.ndarray,
    masks: Iterable[np.ndarray],
    boxes: Iterable[Iterable[int]],
    scores: Iterable[float] | None = None,
) -> np.ndarray:
    canvas = image.copy()
    score_list = list(scores) if scores is not None else None

    for idx, (mask, box) in enumerate(zip(masks, boxes)):
        color = (
            int(40 + (idx * 47) % 180),
            int(80 + (idx * 59) % 160),
            int(120 + (idx * 73) % 120),
        )
        binary = (mask > 0).astype(np.uint8)
        if binary.any():
            colored = np.zeros_like(canvas)
            colored[:, :, 0] = color[0]
            colored[:, :, 1] = color[1]
            colored[:, :, 2] = color[2]
            canvas = np.where(binary[:, :, None] > 0, (0.65 * canvas + 0.35 * colored).astype(np.uint8), canvas)

        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if score_list is not None and idx < len(score_list):
            text = f"{score_list[idx]:.2f}"
            cv2.putText(canvas, text, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return canvas
