from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .utils import ensure_dirs, mask_to_box, save_json


def _read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def _random_background(
    image_index: list[dict[str, Any]],
    crop_h: int,
    crop_w: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int | None, str | None]:
    if not image_index:
        base = rng.normal(128, 18, size=(crop_h, crop_w, 3)).astype(np.float32)
        base += rng.normal(0, 6, size=(crop_h, crop_w, 3)).astype(np.float32)
        return np.clip(base, 0, 255).astype(np.uint8), None, None

    entry = image_index[int(rng.integers(0, len(image_index)))]
    image = _read_image(entry["path"])
    h, w = image.shape[:2]

    if h < crop_h or w < crop_w:
        image = cv2.resize(image, (max(crop_w, w), max(crop_h, h)), interpolation=cv2.INTER_LINEAR)
        h, w = image.shape[:2]

    y0 = int(rng.integers(0, h - crop_h + 1))
    x0 = int(rng.integers(0, w - crop_w + 1))
    crop = image[y0 : y0 + crop_h, x0 : x0 + crop_w].copy()
    return crop, entry.get("image_id"), entry.get("path")


def _procedural_flake(rng: np.random.Generator, target_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    h = int(rng.integers(max(32, target_size // 3), target_size + 1))
    w = int(rng.integers(max(32, target_size // 3), target_size + 1))

    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    points = []
    n = int(rng.integers(6, 12))
    base_r = min(h, w) * 0.38

    for i in range(n):
        angle = 2.0 * np.pi * i / n + rng.normal(0, 0.12)
        radius = base_r * float(rng.uniform(0.65, 1.35))
        x = int(np.clip(cx + radius * np.cos(angle), 1, w - 2))
        y = int(np.clip(cy + radius * np.sin(angle), 1, h - 2))
        points.append([x, y])

    polygon = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)

    texture = rng.normal(140, 25, size=(h, w, 3)).astype(np.float32)
    grad_x = np.linspace(0.75, 1.15, w, dtype=np.float32)[None, :, None]
    grad_y = np.linspace(1.15, 0.8, h, dtype=np.float32)[:, None, None]
    texture = texture * (0.45 * grad_x + 0.55 * grad_y)
    texture += rng.normal(0, 10, size=(h, w, 3)).astype(np.float32)
    texture = np.clip(texture, 0, 255).astype(np.uint8)

    return texture, mask


def _transform_patch(
    patch: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    scale = float(rng.uniform(0.55, 1.65))
    angle = float(rng.uniform(-35.0, 35.0))

    h, w = patch.shape[:2]
    nw = max(8, int(round(w * scale)))
    nh = max(8, int(round(h * scale)))

    patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    center = (nw / 2.0, nh / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    patch = cv2.warpAffine(
        patch,
        matrix,
        (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    mask = cv2.warpAffine(
        mask,
        matrix,
        (nw, nh),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return patch, (mask > 0).astype(np.uint8) * 255, {"scale": scale, "angle": angle}


def _blend_patch(
    canvas: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    alpha_min: float,
    alpha_max: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]] | None:
    h, w = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    if ph >= h or pw >= w:
        return None

    y = int(rng.integers(0, h - ph + 1))
    x = int(rng.integers(0, w - pw + 1))

    mask_bin = (mask > 0).astype(np.uint8)
    if mask_bin.sum() < 40:
        return None

    roi = canvas[y : y + ph, x : x + pw].astype(np.float32)
    patch_f = patch.astype(np.float32)

    m = mask_bin.astype(bool)
    if m.any():
        bg_mean = roi[m].reshape(-1, 3).mean(axis=0)
        fg_mean = patch_f[m].reshape(-1, 3).mean(axis=0)
        patch_f = np.clip(patch_f + 0.6 * (bg_mean - fg_mean), 0, 255)

    alpha = float(rng.uniform(alpha_min, alpha_max))
    alpha_mask = (mask_bin.astype(np.float32) * alpha)[:, :, None]

    blended = roi * (1.0 - alpha_mask) + patch_f * alpha_mask
    canvas[y : y + ph, x : x + pw] = np.clip(blended, 0, 255).astype(np.uint8)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y : y + ph, x : x + pw] = (mask_bin * 255).astype(np.uint8)

    return canvas, full_mask, {"alpha": alpha, "x": float(x), "y": float(y)}


def _add_hard_negative(canvas: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    h, w = canvas.shape[:2]
    box_w = int(rng.integers(max(12, w // 14), max(18, w // 5)))
    box_h = int(rng.integers(max(12, h // 14), max(18, h // 5)))

    x0 = int(rng.integers(0, max(1, w - box_w)))
    y0 = int(rng.integers(0, max(1, h - box_h)))

    patch = canvas[y0 : y0 + box_h, x0 : x0 + box_w].copy()
    patch = cv2.GaussianBlur(patch, (3, 3), 0)

    edges = cv2.Canny(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 30, 80)
    edges_col = np.stack([edges, edges, edges], axis=-1)
    patch = np.clip(0.7 * patch + 0.6 * edges_col, 0, 255).astype(np.uint8)

    x1 = int(rng.integers(0, max(1, w - box_w)))
    y1 = int(rng.integers(0, max(1, h - box_h)))

    alpha = float(rng.uniform(0.2, 0.55))
    roi = canvas[y1 : y1 + box_h, x1 : x1 + box_w].astype(np.float32)
    mixed = roi * (1.0 - alpha) + patch.astype(np.float32) * alpha
    canvas[y1 : y1 + box_h, x1 : x1 + box_w] = np.clip(mixed, 0, 255).astype(np.uint8)

    return {
        "source_type": "hard_negative",
        "bbox": [x1, y1, x1 + box_w - 1, y1 + box_h - 1],
        "alpha": alpha,
    }


def _apply_global_nuisance(canvas: np.ndarray, cfg: Any, rng: np.random.Generator) -> np.ndarray:
    out = canvas.astype(np.float32)

    if rng.random() < float(cfg.intensity_drift_prob):
        gain = float(rng.uniform(0.9, 1.12))
        bias = float(rng.uniform(-8, 8))
        out = out * gain + bias

    if rng.random() < float(cfg.vignetting_prob):
        h, w = out.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx = 0.5 * w
        cy = 0.5 * h
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r_norm = r / max(np.sqrt(cx**2 + cy**2), 1.0)
        vignette = 1.0 - 0.35 * (r_norm ** 1.8)
        out = out * vignette[:, :, None]

    if rng.random() < float(cfg.blur_prob):
        out = cv2.GaussianBlur(out, (3, 3), 0)

    if rng.random() < float(cfg.noise_prob):
        sigma = float(rng.uniform(3.0, 10.0))
        out += rng.normal(0.0, sigma, size=out.shape)

    return np.clip(out, 0, 255).astype(np.uint8)


def _load_patch_from_manifest(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    patch = _read_image(entry["patch_image_path"])
    mask = cv2.imread(entry["patch_mask_path"], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read patch mask: {entry['patch_mask_path']}")
    return patch, (mask > 0).astype(np.uint8) * 255


def _select_instance_source(
    patch_bank: list[dict[str, Any]],
    cfg: Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str, int | None, str | None]:
    use_real_patch = bool(patch_bank) and rng.random() < float(cfg.real_patch_prob)

    if use_real_patch:
        source = patch_bank[int(rng.integers(0, len(patch_bank)))]
        patch, mask = _load_patch_from_manifest(source)
        return patch, mask, "real_patch", source.get("source_image_id"), source.get("source_image_path")

    patch, mask = _procedural_flake(rng)
    return patch, mask, "procedural", None, None


def generate_split(
    split_name: str,
    num_images: int,
    output_root: str | Path,
    image_index: list[dict[str, Any]],
    patch_bank: list[dict[str, Any]],
    cfg: Any,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    print(f"[Synthesis:{split_name}] Generating {num_images} images", flush=True)

    crop_h, crop_w = int(cfg.crop_size[0]), int(cfg.crop_size[1])

    split_root = Path(output_root) / split_name
    image_dir = split_root / "images"
    mask_dir = split_root / "masks"
    ensure_dirs(image_dir, mask_dir)

    manifest: list[dict[str, Any]] = []
    instance_total = 0
    progress_interval = max(1, num_images // 10) if num_images > 0 else 1

    for idx in range(num_images):
        canvas, bg_image_id, bg_image_path = _random_background(image_index, crop_h, crop_w, rng)

        instance_count = int(rng.integers(int(cfg.min_instances), int(cfg.max_instances) + 1))
        instances: list[dict[str, Any]] = []
        hard_negatives: list[dict[str, Any]] = []

        for inst_i in range(instance_count):
            patch, mask, source_type, source_image_id, source_image_path = _select_instance_source(patch_bank, cfg, rng)
            patch, mask, geom = _transform_patch(patch, mask, rng)
            blended = _blend_patch(canvas, patch, mask, rng, float(cfg.alpha_min), float(cfg.alpha_max))
            if blended is None:
                continue

            canvas, full_mask, photo = blended
            bbox = mask_to_box(full_mask)
            if bbox is None:
                continue

            mask_name = f"{split_name}_{idx:06d}_inst_{inst_i:03d}.png"
            mask_path = mask_dir / mask_name
            cv2.imwrite(str(mask_path), full_mask)

            instances.append(
                {
                    "instance_id": inst_i,
                    "category_id": 1,
                    "source_type": source_type,
                    "source_image_id": source_image_id,
                    "source_image_path": source_image_path,
                    "bbox": bbox,
                    "mask_path": str(mask_path),
                    "overlap_order": inst_i,
                    "generation_seed": int(seed),
                    "local_photometric": {
                        "alpha": photo["alpha"],
                        "x": photo["x"],
                        "y": photo["y"],
                        "scale": geom["scale"],
                        "angle": geom["angle"],
                    },
                }
            )

        if rng.random() < float(cfg.hard_negative_prob):
            n_hn = int(rng.integers(1, int(cfg.hard_negative_max_per_image) + 1))
            for _ in range(n_hn):
                hard_negatives.append(_add_hard_negative(canvas, rng))

        canvas = _apply_global_nuisance(canvas, cfg, rng)

        image_name = f"{split_name}_{idx:06d}.png"
        image_path = image_dir / image_name
        cv2.imwrite(str(image_path), canvas)

        instance_total += len(instances)

        manifest.append(
            {
                "image_id": f"{split_name}_{idx:06d}",
                "image_path": str(image_path),
                "width": crop_w,
                "height": crop_h,
                "instances": instances,
                "hard_negatives": hard_negatives,
                "metadata": {
                    "split": split_name,
                    "seed": int(seed),
                    "background_source_image_id": bg_image_id,
                    "background_source_image_path": bg_image_path,
                },
            }
        )

        if (idx + 1) % progress_interval == 0 or (idx + 1) == num_images:
            print(
                f"[Synthesis:{split_name}] Generated {idx + 1}/{num_images} images (instances={instance_total})",
                flush=True,
            )

    save_json(manifest, split_root / "manifest.json")
    print(
        f"[Synthesis:{split_name}] Completed with {len(manifest)} images and {instance_total} instances",
        flush=True,
    )
    return manifest


def generate_hybrid_dataset(
    output_root: str | Path,
    image_index: list[dict[str, Any]],
    patch_bank: list[dict[str, Any]],
    cfg: Any,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    print("[Synthesis] Starting hybrid dataset generation", flush=True)
    train_manifest = generate_split(
        split_name="train",
        num_images=int(cfg.train_images),
        output_root=output_root,
        image_index=image_index,
        patch_bank=patch_bank,
        cfg=cfg,
        seed=seed,
    )
    val_manifest = generate_split(
        split_name="val",
        num_images=int(cfg.val_images),
        output_root=output_root,
        image_index=image_index,
        patch_bank=patch_bank,
        cfg=cfg,
        seed=seed + 1000,
    )

    summary = {
        "train_images": len(train_manifest),
        "val_images": len(val_manifest),
        "train_instances": int(sum(len(x["instances"]) for x in train_manifest)),
        "val_instances": int(sum(len(x["instances"]) for x in val_manifest)),
        "patch_bank_size": len(patch_bank),
    }

    save_json(summary, Path(output_root) / "dataset_summary.json")
    print(f"[Synthesis] Hybrid dataset generation complete: {summary}", flush=True)

    return {
        "train": train_manifest,
        "val": val_manifest,
    }
