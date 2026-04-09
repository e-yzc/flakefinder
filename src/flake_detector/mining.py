from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .utils import ensure_dirs, mask_to_box, save_json


@dataclass
class CandidateRegion:
    bbox: list[int]
    area: float
    solidity: float
    perimeter: float
    score: float


def _read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def list_real_images(real_dir: str | Path, extensions: Iterable[str]) -> list[Path]:
    real_root = Path(real_dir)
    ext_set = {e.lower() for e in extensions}
    image_paths: list[Path] = []
    for path in real_root.rglob("*"):
        if path.suffix.lower() in ext_set:
            image_paths.append(path)
    return sorted(image_paths)


def _image_stats(image_bgr: np.ndarray) -> dict[str, Any]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    mean = rgb.mean(axis=(0, 1)).tolist()
    std = rgb.std(axis=(0, 1)).tolist()
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    illum = float(gray.mean())

    return {
        "mean_rgb": [float(x) for x in mean],
        "std_rgb": [float(x) for x in std],
        "sharpness": sharpness,
        "illumination": illum,
    }


def build_image_index(
    real_dir: str | Path,
    extensions: Iterable[str],
    max_images: int | None = None,
) -> list[dict[str, Any]]:
    paths = list_real_images(real_dir, extensions)
    if max_images is not None:
        paths = paths[:max_images]

    total = len(paths)
    print(f"[Index] Processing {total} real images from {real_dir}", flush=True)

    entries: list[dict[str, Any]] = []
    for idx, path in enumerate(paths):
        image = _read_image(path)
        h, w = image.shape[:2]
        stats = _image_stats(image)
        entries.append(
            {
                "image_id": idx,
                "path": str(path),
                "width": int(w),
                "height": int(h),
                **stats,
            }
        )

        if (idx + 1) % 25 == 0 or (idx + 1) == total:
            print(f"[Index] Processed {idx + 1}/{total} images", flush=True)

    return entries


def save_index_preview_grid(index: list[dict[str, Any]], output_path: str | Path, max_images: int = 16) -> None:
    if not index:
        return

    count = min(len(index), max_images)
    tile_size = 192
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))

    canvas = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)

    for i in range(count):
        entry = index[i]
        img = _read_image(entry["path"])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        r = i // cols
        c = i % cols
        y1 = r * tile_size
        y2 = y1 + tile_size
        x1 = c * tile_size
        x2 = x1 + tile_size
        canvas[y1:y2, x1:x2] = resized

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def mine_candidate_regions(image_bgr: np.ndarray, cfg: Any) -> tuple[list[CandidateRegion], list[np.ndarray]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_ksize = max(3, int(cfg.residual_blur_ksize))
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    background = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    residual = cv2.absdiff(gray, background)
    residual = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)

    _, thresh = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mk = max(1, int(cfg.morph_ksize))
    kernel = np.ones((mk, mk), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = float(gray.shape[0] * gray.shape[1])
    max_area = float(cfg.max_area_ratio) * image_area

    candidates: list[CandidateRegion] = []
    masks: list[np.ndarray] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(cfg.min_area_px):
            continue
        if area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0 or w <= 0:
            continue
        aspect = w / h
        if aspect < float(cfg.min_aspect_ratio) or aspect > float(cfg.max_aspect_ratio):
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < float(cfg.min_solidity):
            continue

        perimeter = float(cv2.arcLength(contour, True))

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)

        score = float(residual[mask > 0].mean()) if mask.any() else 0.0
        bbox = [x, y, x + w - 1, y + h - 1]

        candidates.append(
            CandidateRegion(
                bbox=bbox,
                area=area,
                solidity=float(solidity),
                perimeter=perimeter,
                score=score,
            )
        )
        masks.append(mask)

    return candidates, masks


def _crop_with_margin(image: np.ndarray, mask: np.ndarray, bbox: list[int], margin: int = 4) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w - 1, x2 + margin)
    y2 = min(h - 1, y2 + margin)

    patch = image[y1 : y2 + 1, x1 : x2 + 1]
    patch_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
    return patch, patch_mask


def mine_patch_bank(
    image_index: list[dict[str, Any]],
    patch_bank_dir: str | Path,
    cfg: Any,
    preview_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    patch_root = Path(patch_bank_dir)
    images_dir = patch_root / "images"
    masks_dir = patch_root / "masks"
    ensure_dirs(images_dir, masks_dir)

    preview_canvas = None
    preview_slots = min(int(cfg.preview_count), len(image_index))
    tile = 256
    if preview_slots > 0:
        cols = int(np.ceil(np.sqrt(preview_slots)))
        rows = int(np.ceil(preview_slots / cols))
        preview_canvas = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)

    patch_manifest: list[dict[str, Any]] = []
    patch_id = 0
    total_images = len(image_index)
    print(
        f"[Mining] Scanning {total_images} indexed images for candidate patches (max_patches={int(cfg.max_patches)})",
        flush=True,
    )

    for image_i, entry in enumerate(image_index):
        if patch_id >= int(cfg.max_patches):
            break

        image = _read_image(entry["path"])
        candidates, masks = mine_candidate_regions(image, cfg)

        if preview_canvas is not None and image_i < preview_slots:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            overlay = rgb.copy()
            for cand in candidates:
                x1, y1, x2, y2 = cand.bbox
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (240, 90, 60), 2)
            resized = cv2.resize(overlay, (tile, tile), interpolation=cv2.INTER_AREA)
            cols = preview_canvas.shape[1] // tile
            r = image_i // cols
            c = image_i % cols
            preview_canvas[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = resized

        for cand, mask in zip(candidates, masks):
            if patch_id >= int(cfg.max_patches):
                break

            patch_img, patch_mask = _crop_with_margin(image, mask, cand.bbox)
            box = mask_to_box(patch_mask)
            if box is None:
                continue

            image_name = f"patch_{patch_id:06d}.png"
            mask_name = f"patch_{patch_id:06d}_mask.png"
            image_path = images_dir / image_name
            mask_path = masks_dir / mask_name

            cv2.imwrite(str(image_path), patch_img)
            cv2.imwrite(str(mask_path), patch_mask)

            patch_manifest.append(
                {
                    "patch_id": patch_id,
                    "patch_image_path": str(image_path),
                    "patch_mask_path": str(mask_path),
                    "source_image_id": entry["image_id"],
                    "source_image_path": entry["path"],
                    "source_bbox": cand.bbox,
                    "patch_bbox": box,
                    "region": asdict(cand),
                }
            )
            patch_id += 1

            if patch_id % 200 == 0:
                print(f"[Mining] Collected {patch_id} patches so far", flush=True)

        if (image_i + 1) % 20 == 0 or (image_i + 1) == total_images:
            print(
                f"[Mining] Scanned {image_i + 1}/{total_images} images, collected {patch_id} patches",
                flush=True,
            )

    save_json(patch_manifest, patch_root / "patch_bank_manifest.json")
    print(f"[Mining] Patch bank complete with {len(patch_manifest)} patches", flush=True)

    if preview_canvas is not None and preview_path:
        preview_target = Path(preview_path)
        preview_target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_target), cv2.cvtColor(preview_canvas, cv2.COLOR_RGB2BGR))

    return patch_manifest


def load_labeled_patch_bank(labeled_dir: str | Path) -> list[dict[str, Any]]:
    labeled_root = Path(labeled_dir)
    images_dir = labeled_root / "images"
    masks_dir = labeled_root / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        return []

    mask_lookup = {path.stem: path for path in masks_dir.glob("*.png")}
    entries: list[dict[str, Any]] = []

    image_paths = sorted(path for path in images_dir.iterdir() if path.is_file())
    for patch_id, image_path in enumerate(image_paths):
        if "_annotated" not in image_path.stem:
            continue
        source_stem = image_path.stem.removesuffix("_annotated")
        mask_key = f"{source_stem}_mask"
        mask_path = mask_lookup.get(mask_key)
        if mask_path is None:
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read labeled mask: {mask_path}")
        box = mask_to_box(mask)
        if box is None:
            continue

        entries.append(
            {
                "patch_id": patch_id,
                "patch_image_path": str(image_path),
                "patch_mask_path": str(mask_path),
                "source_image_id": source_stem,
                "source_image_path": str(image_path),
                "source_bbox": box,
                "patch_bbox": box,
                "region": {
                    "bbox": box,
                    "area": float((mask > 0).sum()),
                    "solidity": 1.0,
                    "perimeter": 0.0,
                    "score": 1.0,
                },
                "source_type": "manual_label",
            }
        )

    return entries
