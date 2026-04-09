from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18

from .utils import ensure_dirs, save_json


class _SSLHead(torch.nn.Module):
    def __init__(self, in_dim: int = 512, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _random_view(image: np.ndarray, crop_size: int, rng: np.random.Generator) -> torch.Tensor:
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        image = cv2.resize(image, (max(w, crop_size), max(h, crop_size)), interpolation=cv2.INTER_LINEAR)
        h, w = image.shape[:2]

    y = int(rng.integers(0, h - crop_size + 1))
    x = int(rng.integers(0, w - crop_size + 1))
    crop = image[y : y + crop_size, x : x + crop_size].copy()

    if rng.random() < 0.5:
        crop = crop[:, ::-1]
    if rng.random() < 0.4:
        gain = float(rng.uniform(0.85, 1.15))
        bias = float(rng.uniform(-8, 8))
        crop = np.clip(crop.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

    tensor = torch.as_tensor(crop.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    return tensor


def run_optional_ssl(
    ssl_cfg: Any,
    image_index: list[dict[str, Any]],
    device: torch.device,
    output_dir: str | Path,
    seed: int,
) -> dict[str, Any]:
    if not bool(getattr(ssl_cfg, "enabled", False)):
        return {"enabled": False, "status": "disabled"}

    if not image_index:
        return {"enabled": True, "status": "skipped_no_real_images"}

    ensure_dirs(output_dir)
    rng = np.random.default_rng(seed)
    print(
        f"[SSL] Starting optional SSL pretraining with {len(image_index)} source images on device {device}",
        flush=True,
    )

    encoder = resnet18(weights=None)
    encoder.fc = torch.nn.Identity()
    head = _SSLHead()

    encoder.to(device)
    head.to(device)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)

    crop_size = int(getattr(ssl_cfg, "crop_size", 224))
    steps = max(1, 20 * int(getattr(ssl_cfg, "epochs", 1)))
    losses: list[float] = []
    progress_interval = max(1, steps // 5)
    print(f"[SSL] Running {steps} optimization steps", flush=True)

    encoder.train()
    head.train()

    for step in range(steps):
        selected = image_index[int(rng.integers(0, len(image_index)))]
        image = cv2.imread(selected["path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        v1 = _random_view(image, crop_size, rng).unsqueeze(0).to(device)
        v2 = _random_view(image, crop_size, rng).unsqueeze(0).to(device)

        z1 = F.normalize(head(encoder(v1)), dim=1)
        z2 = F.normalize(head(encoder(v2)), dim=1)

        # Negative cosine similarity for two-view alignment.
        loss = 1.0 - (z1 * z2).sum(dim=1).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

        if (step + 1) % progress_interval == 0 or (step + 1) == steps:
            print(f"[SSL] Progress: {step + 1}/{steps}", flush=True)

    encoder_path = Path(output_dir) / "ssl_encoder.pt"
    torch.save({"encoder": encoder.state_dict(), "head": head.state_dict()}, encoder_path)

    report = {
        "enabled": True,
        "status": "completed",
        "steps": len(losses),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "encoder_path": str(encoder_path),
    }
    save_json(report, Path(output_dir) / "ssl_report.json")
    print(f"[SSL] Completed optional SSL pretraining: {report}", flush=True)
    return report
