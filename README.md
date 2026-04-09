# Flake Detector Pretraining Pipeline

Compact PyTorch pipeline for **class-agnostic instance segmentation** of flake-like objects in microscope images.

This repository implements only the pretraining stage (universal detector) using unlabeled real microscope images plus hybrid pseudo-synthetic supervision.

## Purpose

The detector answers one question:

> Where are the flake-like instances in this image?

It does **not** predict material class, thickness, or downstream utility labels.

## Scope

Implemented:
- Real-image indexing and statistics.
- Candidate region mining to create a real patch bank.
- Hybrid synthetic generation (real backgrounds, real copy-paste patches, procedural minority source, hard negatives).
- Class-agnostic Mask R-CNN training (`small`, `large`).
- Quantitative and qualitative evaluation outputs.
- Real-image prediction dumps for sanity checking.
- Lightweight smoke tests.

Out of scope:
- Material/thickness classification.
- Active learning platform.
- GUI/web/backend/deployment tooling.
- Framework-heavy abstractions.

## Setup (uv)

```bash
uv sync --dev
```

If needed, add your real microscope images under:

```text
data/real/
```

Supported extensions: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`.

## Dataset Build

Runs Stage A/B/C:
1. Index real images + stats.
2. Mine candidate regions + patch bank.
3. Generate hybrid train/val synthetic dataset + manifests.

```bash
uv run python build_dataset.py
```

Useful overrides:

```bash
uv run python build_dataset.py --max-images 80 --train-images 300 --val-images 60 --seed 7
```

Primary outputs:
- `data/splits/image_index.json`
- `data/splits/patch_bank/patch_bank_manifest.json`
- `data/splits/train_manifest.json`
- `data/splits/val_manifest.json`
- `data/splits/dataset_summary.json`
- `outputs/previews/index_preview.png`
- `outputs/previews/mining_preview.png`
- `outputs/previews/generated_train_preview.png`

If `labeled/images` and `labeled/masks` exist, Stage B automatically merges those manual labels into the patch bank before synthetic generation.


## Manual Labeling App

Use the lightweight OpenCV annotator to manually label flakes on real images.

```bash
uv run python annotate_dataset.py --source-dir data/real --labeled-dir labeled
```

Controls:
- Left click: add polygon vertex.
- Middle click: close + commit polygon (saved immediately).
- Right click: remove last vertex.
- `n`: move to next random unlabeled image.
- `r`: reset current mask.
- `q`: quit.

Outputs:
- `labeled/images/<name>_annotated.<ext>`
- `labeled/masks/<name>_mask.png`

A filename check is used to avoid duplicate work (e.g., `img.png` maps to `img_annotated.png`).
These labeled image/mask pairs are consumed automatically in Stage B by `build_dataset.py`.

## Training

Trains class-agnostic Mask R-CNN with one model family and two sizes:
- `small`
- `large`

```bash
uv run python train.py --model-size small
uv run python train.py --model-size large
```

Optional overrides:

```bash
uv run python train.py --epochs 4 --batch-size 2 --device mps
```

Training outputs are saved to:
- `outputs/checkpoints/`
- `outputs/metrics/`
- `outputs/previews/<run_id>/`
- `outputs/reports/<run_id>/`

Each run stores:
- checkpoint(s)
- config snapshot
- seed/device metadata
- metrics log (`jsonl`)
- validation overlays
- real unlabeled image prediction dumps (`real_sanity/`)
- run summary (includes latency sanity check)

## Hybrid Synthetic Strategy

Each generated sample uses a configurable mix of:
- real background crops,
- mined real flake patches (`real_patch`),
- procedural flakes (`procedural`, minority source),
- unlabeled hard negatives (`hard_negative` metadata).

Per-instance metadata includes:
- `instance_id`
- `source_type`
- `source_image_id`
- `bbox`
- `mask_path`
- `overlap_order`
- `generation_seed`
- local photometric/geometric parameters.

## Tests

Run smoke tests:

```bash
uv run pytest -q
```

Covers:
- config loading,
- dataset loading,
- synthetic generation,
- one-batch forward pass,
- one-batch training step,
- device selection.

## Repository Layout

```text
flakefinder/
├─ pyproject.toml
├─ README.md
├─ build_dataset.py
├─ train.py
├─ inspect.ipynb
├─ src/flake_detector/
│  ├─ __init__.py
│  ├─ configs.py
│  ├─ utils.py
│  ├─ models.py
│  ├─ data.py
│  ├─ mining.py
│  ├─ synth_data.py
│  ├─ eval.py
│  └─ ssl.py
├─ data/
│  ├─ real/
│  ├─ generated/
│  └─ splits/
├─ outputs/
│  ├─ checkpoints/
│  ├─ metrics/
│  ├─ previews/
│  ├─ reports/
│  └─ pseudo_labels/
└─ tests/
   ├─ test_config.py
   ├─ test_data_smoke.py
   ├─ test_synth_smoke.py
   ├─ test_model_smoke.py
   └─ test_train_step.py
```
