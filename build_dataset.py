from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flake_detector.configs import load_config, save_config_snapshot  # noqa: E402
from flake_detector.mining import (  # noqa: E402
    build_image_index,
    load_labeled_patch_bank,
    mine_patch_bank,
    save_index_preview_grid,
)
from flake_detector.synth_data import generate_hybrid_dataset  # noqa: E402
from flake_detector.utils import ensure_dirs, save_json, set_seed  # noqa: E402


def _save_generated_preview(manifest: list[dict], output_path: Path, max_images: int = 16) -> None:
    if not manifest:
        return

    tile = 192
    count = min(len(manifest), max_images)
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    canvas = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)

    for i in range(count):
        img = cv2.imread(manifest[i]["image_path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (tile, tile), interpolation=cv2.INTER_AREA)
        r = i // cols
        c = i % cols
        canvas[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = resized

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build image index, patch bank, and hybrid training dataset")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config override file")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument("--max-images", type=int, default=None, help="Optional max number of real images to index")
    parser.add_argument("--train-images", type=int, default=None, help="Synthetic train image count override")
    parser.add_argument("--val-images", type=int, default=None, help="Synthetic val image count override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_images is not None:
        cfg.mining.max_images = args.max_images
    if args.train_images is not None:
        cfg.synth.train_images = args.train_images
    if args.val_images is not None:
        cfg.synth.val_images = args.val_images

    print("[Setup] Starting dataset build pipeline", flush=True)
    print(
        f"[Setup] seed={cfg.seed} real_dir={cfg.paths.real_dir} generated_dir={cfg.paths.generated_dir}",
        flush=True,
    )

    set_seed(cfg.seed)

    ensure_dirs(
        cfg.paths.generated_dir,
        cfg.paths.splits_dir,
        cfg.paths.previews_dir,
        cfg.paths.reports_dir,
    )

    print("[Stage A] Building real image index...", flush=True)
    image_index = build_image_index(
        real_dir=cfg.paths.real_dir,
        extensions=cfg.image_extensions,
        max_images=cfg.mining.max_images,
    )
    print(f"[Stage A] Indexed {len(image_index)} real images", flush=True)

    print("[Stage A] Saving image index artifacts...", flush=True)
    save_json(image_index, Path(cfg.paths.splits_dir) / "image_index.json")
    save_index_preview_grid(
        image_index,
        Path(cfg.paths.previews_dir) / "index_preview.png",
        max_images=cfg.mining.preview_count,
    )

    print("[Stage B] Mining candidate regions and building patch bank...", flush=True)
    mined_patch_bank = mine_patch_bank(
        image_index=image_index,
        patch_bank_dir=Path(cfg.paths.splits_dir) / "patch_bank",
        cfg=cfg.mining,
        preview_path=Path(cfg.paths.previews_dir) / "mining_preview.png",
    )
    labeled_patch_bank = load_labeled_patch_bank(cfg.paths.labeled_dir)
    patch_bank = [*labeled_patch_bank, *mined_patch_bank]
    print(
        f"[Stage B] Patch bank size: total={len(patch_bank)} labeled={len(labeled_patch_bank)} mined={len(mined_patch_bank)}",
        flush=True,
    )

    print(
        f"[Stage C] Generating hybrid dataset (train={cfg.synth.train_images}, val={cfg.synth.val_images})...",
        flush=True,
    )
    split_manifests = generate_hybrid_dataset(
        output_root=cfg.paths.generated_dir,
        image_index=image_index,
        patch_bank=patch_bank,
        cfg=cfg.synth,
        seed=cfg.seed,
    )

    train_manifest = split_manifests["train"]
    val_manifest = split_manifests["val"]
    print(
        f"[Stage C] Generated split manifests: train={len(train_manifest)} val={len(val_manifest)}",
        flush=True,
    )

    print("[Finalize] Saving manifests, summary, config snapshot, and previews...", flush=True)
    save_json(train_manifest, Path(cfg.paths.splits_dir) / "train_manifest.json")
    save_json(val_manifest, Path(cfg.paths.splits_dir) / "val_manifest.json")

    dataset_summary = {
        "seed": cfg.seed,
        "indexed_real_images": len(image_index),
        "patch_bank_size": len(patch_bank),
        "train_images": len(train_manifest),
        "val_images": len(val_manifest),
        "train_instances": int(sum(len(x["instances"]) for x in train_manifest)),
        "val_instances": int(sum(len(x["instances"]) for x in val_manifest)),
    }

    save_json(dataset_summary, Path(cfg.paths.splits_dir) / "dataset_summary.json")
    save_config_snapshot(cfg, Path(cfg.paths.splits_dir) / "build_config_snapshot.json")

    _save_generated_preview(train_manifest, Path(cfg.paths.previews_dir) / "generated_train_preview.png")
    _save_generated_preview(val_manifest, Path(cfg.paths.previews_dir) / "generated_val_preview.png")

    print("[Finalize] Dataset build complete", flush=True)
    print(dataset_summary, flush=True)


if __name__ == "__main__":
    main()
