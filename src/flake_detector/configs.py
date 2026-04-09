from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
import json


@dataclass
class PathsConfig:
    real_dir: str = "data/real"
    generated_dir: str = "data/generated"
    splits_dir: str = "data/splits"
    outputs_dir: str = "outputs"
    checkpoints_dir: str = "outputs/checkpoints"
    metrics_dir: str = "outputs/metrics"
    previews_dir: str = "outputs/previews"
    reports_dir: str = "outputs/reports"
    pseudo_labels_dir: str = "outputs/pseudo_labels"


@dataclass
class MiningConfig:
    max_images: int | None = None
    min_area_px: int = 120
    max_area_ratio: float = 0.12
    min_solidity: float = 0.5
    min_aspect_ratio: float = 0.15
    max_aspect_ratio: float = 6.5
    residual_blur_ksize: int = 41
    morph_ksize: int = 3
    max_patches: int = 1800
    preview_count: int = 24


@dataclass
class SynthConfig:
    crop_size: tuple[int, int] = (640, 640)
    train_images: int = 400
    val_images: int = 80
    min_instances: int = 0
    max_instances: int = 5
    real_patch_prob: float = 0.75
    procedural_prob: float = 0.25
    hard_negative_prob: float = 0.35
    hard_negative_max_per_image: int = 3
    alpha_min: float = 0.45
    alpha_max: float = 0.9
    blur_prob: float = 0.25
    noise_prob: float = 0.3
    vignetting_prob: float = 0.2
    intensity_drift_prob: float = 0.25


@dataclass
class ModelConfig:
    family: str = "maskrcnn"
    size: str = "small"
    num_classes: int = 2
    score_thresh: float = 0.5


@dataclass
class TrainConfig:
    batch_size: int = 2
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 6
    max_train_steps_per_epoch: int | None = None
    max_val_steps: int | None = None
    preview_count: int = 8
    mixed_precision: bool = False
    device_override: str | None = None


@dataclass
class SSLConfig:
    enabled: bool = False
    method: str = "simclr"
    epochs: int = 1
    batch_size: int = 32
    crop_size: int = 224


@dataclass
class SelfTrainingConfig:
    enabled: bool = False
    confidence_threshold: float = 0.9
    max_predictions: int = 200


@dataclass
class ExperimentConfig:
    seed: int = 42
    experiment_name: str = "flake_pretrain"
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    paths: PathsConfig = field(default_factory=PathsConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    self_training: SelfTrainingConfig = field(default_factory=SelfTrainingConfig)


def default_config() -> ExperimentConfig:
    return ExperimentConfig()


def _update_dataclass(instance: Any, update_data: dict[str, Any]) -> Any:
    if not is_dataclass(instance):
        return update_data

    field_map = {f.name: f for f in fields(instance)}
    for key, value in update_data.items():
        if key not in field_map:
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    cfg = default_config()
    _update_dataclass(cfg, data)
    return cfg


def load_config(config_path: str | Path | None = None) -> ExperimentConfig:
    if config_path is None:
        return default_config()

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return config_from_dict(payload)


def merge_overrides(cfg: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    return _update_dataclass(cfg, overrides)


def to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return asdict(cfg)


def save_config_snapshot(cfg: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(to_dict(cfg), f, indent=2)
