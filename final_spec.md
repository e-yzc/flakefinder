# Final Spec: Compact Pretraining Pipeline for Class-Agnostic Flake Instance Segmentation

## 1. Purpose

Build a compact research-grade pretraining pipeline for microscope images of exfoliated 2D materials.

The goal of this stage is to train a **class-agnostic instance segmentation model** that detects and segments flake-like objects in microscope images.

This project must stay small enough to inspect comfortably in an IDE and simple enough that one person can understand the full execution path quickly. It is **not** a general-purpose ML framework, **not** a production application, and **not** a large reusable platform.

This spec is for **pretraining only**.

---

## 2. Scope boundary

### In scope

Implement only what is needed to:

- index and inspect real unlabeled microscope images,
- derive useful image statistics,
- optionally mine approximate flake-like regions from real images,
- generate a hybrid synthetic or pseudo-synthetic training set,
- optionally run self-supervised pretraining on real unlabeled crops,
- train a class-agnostic instance segmentation model,
- evaluate and visualize detector outputs,
- save checkpoints, metrics, previews, and reproducibility metadata.

### Explicitly out of scope

Do **not** implement:

- downstream material classification,
- thickness prediction,
- few-shot adaptation tooling,
- active learning infrastructure,
- microscope control software,
- GUI or web frontend,
- database or backend integration,
- deployment packaging,
- large plugin or registry systems,
- a heavy CLI platform,
- a large multi-file configuration hierarchy,
- a multi-agent orchestration system.

This must remain a focused pretraining project.

---

## 3. Full-project context

The eventual full system will likely have two stages:

1. **Universal detector**
   - receives a microscope image,
   - outputs candidate flake instances with masks and confidence scores.

2. **Material-specific interpreter**
   - receives detected flake crops, masks, and derived features,
   - predicts later-stage labels such as material, thickness, or utility score.

This project builds **only stage 1** while preserving enough metadata and artifact structure for stage 2 to be added later.

That means:

- the detector must remain **class-agnostic**,
- synthetic generation should preserve useful instance-level metadata,
- future classifier needs must **not** drive the architecture into unnecessary complexity today.

---

## 4. Core design principles

### 4.1 Compact codebase
Keep the number of files low. Prefer a few explicit modules over many abstract ones.

### 4.2 Easy IDE inspection
A user should be able to open the repo and understand the main execution path without navigating dozens of files.

### 4.3 Real data first
The main asset is the collection of unlabeled microscope images. The pipeline should exploit them directly.

### 4.4 Hybrid supervision
Training supervision should come from a mixture of:
- real-image-derived candidate regions,
- copy-paste compositing,
- hard negatives from real artifacts,
- and a limited procedural synthetic component.

### 4.5 Boring and maintainable
Prefer explicit PyTorch code over abstraction-heavy patterns.

### 4.6 One task only
This project trains a class-agnostic detector. It does not solve the downstream classifier problem.

---

## 5. Data assumptions

### 5.1 Real data

Available:
- curated unlabeled RGB microscope images,
- nominal raw size around **1920 × 1200**.

Not assumed available:
- real instance masks,
- material labels,
- thickness labels.

### 5.2 Detector objective

The model should answer only:

> Where are the flake-like instances in this image?

It must **not** predict material type or thickness in this stage.

### 5.3 Training supervision sources

The training set should be built from a configurable mixture of:

- real background crops,
- mined approximate flake-like regions from real images,
- copy-paste compositing across real images,
- hard negatives derived from real backgrounds and artifacts,
- optional procedural flakes.

**Important:** real-derived pseudo-real supervision is preferred, but it must **not** be treated as guaranteed ground truth. The system should support a mixture of sources so that poor candidate mining does not dominate the training signal.

---

## 6. Recommended learning strategy

Organize the work into the following stages.

### 6.1 Stage A — real-image indexing and statistics

Scan the real image collection and build a lightweight image index containing:

- file path,
- image size,
- basic color statistics,
- optional focus/blur measures,
- optional illumination summaries,
- optional crop availability.

This stage should also produce preview grids for manual inspection.

### 6.2 Stage B — candidate region mining on real images

Attempt to extract approximate flake-like regions directly from real images using unsupervised or weakly supervised classical methods.

Possible methods include:

- illumination correction,
- local contrast residuals,
- color-space thresholding,
- gradient-based segmentation,
- superpixels plus merging,
- watershed or contour extraction,
- morphological filtering,
- shape filtering using size, solidity, and perimeter features.

These regions are **not** assumed to be perfect labels. They are intended to create a **candidate patch bank** of plausible real flake crops and masks.

### 6.3 Stage C — hybrid synthetic generation

The main training set should be generated by composing scenes from real data plus optional procedural components.

Each training sample may contain:

- a real background crop,
- zero or more mined real flake patches pasted from a patch bank,
- optional procedurally generated flakes,
- nuisance effects such as blur, noise, vignetting, intensity drift, and focus variation,
- hard-negative structures that resemble flakes but should not be labeled as flakes.

This stage must output:

- RGB image,
- per-instance masks,
- bounding boxes,
- image-level metadata,
- per-instance metadata.

### 6.4 Stage D — optional SSL on real crops

If implemented cleanly and without much extra complexity, include optional self-supervised pretraining on real unlabeled crops.

Acceptable examples:
- MAE,
- BYOL,
- SimCLR,
- DINO-style methods.

This stage is optional and must not complicate the main supervised path.

### 6.5 Stage E — supervised detector training

Train a class-agnostic instance segmentation model on the hybrid dataset.

Start from a **single model family** and vary only its scale.

### 6.6 Stage F — optional self-training

Optional only after the baseline exists and only if the synthetic-to-real gap remains significant.

Possible workflow:
- run detector on real unlabeled images,
- keep only high-confidence predictions,
- filter aggressively,
- reuse them as pseudo-labels in a later training round.

This is **not** part of the core baseline path and must not complicate the initial implementation.

---

## 7. Synthetic generation requirements

Synthetic generation is important, but it must be grounded in real microscope structure.

### 7.1 Avoid relying primarily on

- random polygon overlays,
- flat color fills,
- simplistic contrast shifts,
- unrealistic masks detached from real morphology.

### 7.2 Prefer

#### Real-flake copy-paste
Use candidate real flake regions mined from unlabeled images and paste them into other real backgrounds.

#### Texture-aware synthesis
Use real flake textures or regions and warp them mildly to increase diversity.

#### Illumination-aware compositing
Blend patches into real backgrounds using local color and illumination matching.

#### Hard negatives
Generate distracting non-flake structures from real microscope artifacts such as:
- dust,
- bubbles,
- scratches,
- blur patches,
- substrate structures,
- contamination,
- illumination gradients.

These are critical for controlling false positives.

### 7.3 Procedural component
Procedural generation is allowed, but should remain:
- a minority component,
- a morphology widener,
- or an augmentation source.

It must not be the sole or dominant supervision source unless real-derived mining proves unusable.

### 7.4 Required outputs
For each generated training sample, the pipeline must be able to produce:

- final RGB image,
- per-instance binary masks,
- bounding boxes,
- image-level metadata,
- per-instance metadata.

### 7.5 Required per-instance metadata
Keep at least:

- instance id,
- source type (`real_patch`, `procedural`, `hard_negative` when applicable),
- source image id if derived from a real image,
- bounding box,
- mask reference,
- overlap/order information,
- generation seed,
- local photometric parameters.

Optional future-looking fields such as approximate thickness or material tags may be stored if convenient, but they are not required for training in this stage.

---

## 8. Data representation

Use a simple internal representation centered on raster masks and a stable manifest format.

### Requirement
The project must define **one canonical annotation schema** for instance data. It should be simple, inspectable, and easy to serialize.

A COCO-like structure is recommended for manifests, but the code does **not** need to revolve around heavy COCO tooling.

### Acceptable approach
- internal Python dicts or dataclasses,
- JSON manifests,
- PNG or NumPy mask storage,
- optional COCO export for compatibility and visualization.

The priority is simplicity **without** losing annotation consistency.

---

## 9. Model choice

### 9.1 One model family only

Use a single practical instance segmentation family.

Recommended default:
- **Mask R-CNN**

Reason:
- mature,
- class-agnostic segmentation is straightforward,
- well supported in PyTorch,
- easy to debug,
- realistic for Apple Silicon and modest GPUs.

Avoid a model zoo or architecture registry.

### 9.2 Scale variants

Support exactly two variants:

- `small`
- `large`

These must be selected by a single config field or argument, not separate codepaths.

Scale differences may come from:
- backbone size,
- input crop size,
- training schedule,
- FPN width or ROI settings.

No forked architecture hierarchy is needed.

---

## 10. Platform and framework requirements

### Framework
Use **PyTorch**.

### Package manager
Use **uv**.

### Device support
Support:
- CUDA,
- MPS,
- CPU.

Use one small helper for device selection. Do not scatter backend-specific logic throughout the code.

### Precision policy
Keep mixed precision conservative and backend-safe. Do not assume CUDA-only AMP behavior.

---

## 11. Repository layout

Use a compact repository with a small `src` package.

```text
flake_detector/
├─ pyproject.toml
├─ README.md
├─ train.py
├─ build_dataset.py
├─ inspect.ipynb
├─ src/
│  └─ flake_detector/
│     ├─ __init__.py
│     ├─ configs.py
│     ├─ utils.py
│     ├─ models.py
│     ├─ data.py
│     ├─ mining.py
│     ├─ synth_data.py
│     ├─ eval.py
│     └─ ssl.py
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

### File responsibilities

#### `train.py`
Main script for:
- loading config,
- creating model,
- training,
- validation,
- saving checkpoints,
- saving preview images,
- optional real-image prediction dumps.

#### `build_dataset.py`
Builds:
- image index,
- train/val splits,
- dataset stats,
- candidate region proposals,
- optional patch bank.

#### `src/flake_detector/configs.py`
Contains a small number of Python configs or dataclasses.

#### `src/flake_detector/utils.py`
Contains:
- seed setup,
- device selection,
- checkpoint helpers,
- simple logging,
- plotting helpers.

#### `src/flake_detector/models.py`
Contains:
- model creation,
- `small` / `large` selection,
- any backbone-specific code.

#### `src/flake_detector/data.py`
Contains:
- dataset classes,
- transforms,
- manifest loading,
- collate function.

#### `src/flake_detector/mining.py`
Contains:
- candidate-region mining from real micrographs,
- shape filtering,
- patch-bank extraction utilities.

#### `src/flake_detector/synth_data.py`
Builds hybrid synthetic samples from:
- real backgrounds,
- real mined regions,
- optional procedural flakes,
- hard negatives.

May either generate data on the fly or precompute and save it.

#### `src/flake_detector/eval.py`
Contains:
- validation metrics,
- overlay generation,
- preview dumps,
- real-image sanity-check prediction exports.

#### `src/flake_detector/ssl.py`
Optional SSL utilities, kept isolated.

#### `inspect.ipynb`
Used for:
- visual checking,
- mask inspection,
- synthetic sample review,
- failure analysis.

This notebook is part of the intended workflow.

---

## 12. Configuration requirements

Keep configuration simple and local.

A single Python config file or dataclass setup is preferred.

At minimum support configuration for:

- data paths,
- model size,
- batch size,
- learning rate,
- epoch count,
- input crop size,
- augmentation settings,
- synthetic/pseudo-real mix ratio,
- whether SSL is enabled,
- whether self-training is enabled,
- random seed,
- device override.

No hidden defaults beyond reasonable script-level defaults.

---

## 13. Training requirements

### 13.1 Main path

The core baseline deliverable is:

1. prepare real image index,
2. mine candidate regions,
3. build hybrid training data,
4. train instance detector,
5. evaluate and inspect outputs.

### 13.2 Losses

Use the losses native to the chosen instance segmentation family unless there is a strong reason not to.

Do not invent an elaborate custom loss stack at the start.

### 13.3 Augmentation

Include only useful microscope-relevant augmentations:

- flips if physically acceptable,
- small rotations if appropriate,
- color/intensity jitter,
- blur,
- noise,
- local contrast shifts,
- crop and resize.

Avoid generic augmentation clutter.

### 13.4 Hard negatives

Hard negatives must be part of training, not an afterthought.

The dataset should include some regions or images with:
- strong structure,
- artifacts,
- contrast,
- non-flake regions that could confuse the model.

---

## 14. Evaluation requirements

Evaluation must stay simple and useful, but it must be explicit.

### 14.1 Quantitative outputs

Support at least:

- validation loss,
- mask IoU or AP-style metrics on held-out synthetic or hybrid validation data,
- proposal counts,
- precision/recall summaries if practical,
- a simple latency sanity check for `small` and `large`.

### 14.2 Qualitative outputs

Always save:

- prediction overlays,
- mask visualizations,
- representative failures,
- predictions on held-out real unlabeled micrographs.

### 14.3 Real-image sanity check

Even without labels, the pipeline must save detector predictions on real unlabeled micrographs for manual inspection.

This is mandatory because synthetic validation numbers alone may be misleading.

---

## 15. Saved artifacts

For each run, save:

- model checkpoint,
- training config snapshot,
- random seed,
- metrics log,
- representative prediction previews,
- dataset summary,
- optional pseudo-label cache information.

A small JSON or CSV-based logging approach is enough. Do not build a large experiment-tracking subsystem.

---

## 16. Testing requirements

Keep tests lightweight.

At minimum, include smoke tests for:

- config loading,
- dataset loading,
- one synthetic sample generation,
- one-batch forward pass,
- one-batch training step,
- device selection.

These tests should be fast and practical.

---

## 17. Documentation requirements

Keep documentation minimal.

`README.md` must explain:
- project purpose,
- setup with `uv`,
- dataset preparation,
- how training works,
- how hybrid synthetic generation works,
- how to run `small` and `large`,
- what is explicitly out of scope.

Short notes may be added later if needed, but do not split documentation into many files without a strong reason.

---

## 18. Success criteria

This stage is successful if:

- the repo is compact and easy to inspect,
- real unlabeled images can be indexed and previewed,
- a candidate-region bank can be derived from real images,
- hybrid training samples can be created reproducibly,
- a `small` and a `large` class-agnostic detector can train end-to-end,
- predictions on real unlabeled micrographs are visually plausible,
- checkpoints and previews are saved cleanly,
- the project remains tightly scoped to pretraining only.

---

## 19. Explicit implementation guidance

- Keep the project compact.
- Use one main training script.
- Use real unlabeled data aggressively.
- Prefer pseudo-real copy-paste supervision over naive polygon synthesis.
- Treat procedural synthetic flakes as secondary.
- Keep the detector class-agnostic.
- Do not introduce framework-style abstractions.
- Do not optimize for hypothetical future team scaling.
- Optimize for getting a working microscope-domain detector with minimum conceptual clutter.

---

## 20. Suggested milestones

### Milestone 1
Set up repo, config, training skeleton, and image indexing.

### Milestone 2
Implement candidate-region mining on real micrographs and generate preview outputs.

### Milestone 3
Build patch bank and hybrid synthetic generation with real backgrounds and copy-paste compositing.

### Milestone 4
Train `small` detector end-to-end and inspect predictions.

### Milestone 5
Improve generation with hard negatives and better compositing.

### Milestone 6
Train `large` variant.

### Milestone 7
Optionally add SSL pretraining or self-training only if real-image transfer is still weak.

---

## 21. Final note on the synthetic-data strategy

The central design choice in this spec is that synthetic supervision should be treated as a **domain-bridging problem grounded in real unlabeled data**, not primarily as procedural rendering.

That means the preferred strategy is:

- reuse and remix real microscope structure wherever possible,
- use candidate mining and copy-paste to capture real morphology and texture,
- add procedural generation only to widen support and prevent overfitting,
- keep the codebase simple enough to iterate quickly when the first version fails.

This is more likely to produce a useful pretrained detector than a large but abstract synthetic-rendering framework.
