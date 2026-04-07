# Spec: Pretraining Pipeline for Universal 2D-Material Flake Instance Segmentation

## 1. Purpose

Build the **pretraining stage** of a modular machine-learning pipeline for microscope images of exfoliated 2D materials.

The long-term system will eventually support:

- flake detection in microscope images,
- per-flake downstream classification and ranking,
- adaptation to new materials and substrates with very few labeled examples,
- deployment on modest lab hardware.

**This stage is only for pretraining the universal detector.**

The pretrained model must learn **class-agnostic instance segmentation of flake-like objects** in microscope images. It should detect and segment candidate flakes without requiring material or thickness labels at training time.

This stage must also include a **synthetic data generation pipeline** that can produce instance masks and auxiliary per-instance metadata for future downstream classifier work, even though classifier training itself is out of scope here.

---

## 2. Hard scope boundary

### In scope
Implement only the components needed to:

1. define the training data model,
2. ingest curated real unlabeled microscope images,
3. generate synthetic training samples using those real images as background/style source,
4. optionally run self-supervised or reconstruction-style pretraining on real unlabeled images if useful,
5. train a **class-agnostic instance segmentation model**,
6. evaluate the pretrained detector on synthetic and held-out validation data,
7. export trained checkpoints, logs, reproducibility artifacts, and feature embeddings for later use.

### Explicitly out of scope
Do **not** implement any of the following in this stage:

- downstream material classifier,
- thickness classifier,
- few-shot adaptation pipeline for new materials,
- active learning loop,
- microscope hardware control,
- production GUI,
- scan scheduler,
- deployment packaging,
- online tracking or revisit logic,
- database/backend integration,
- end-user inference application.

Agents must not quietly expand scope.

---

## 3. Context for the full project

The eventual full system will likely have two stages:

1. **Universal detector**
   - receives a microscope image,
   - outputs instance masks and confidence scores for flake-like objects.

2. **Material-specific interpreter**
   - receives per-flake crops, masks, and derived features,
   - predicts later-stage labels such as material, thickness, or utility score,
   - can be adapted with very few labeled examples.

This pretraining project builds only stage 1, while preserving interfaces and metadata needed for stage 2 later.

That means:

- the detector must remain **class-agnostic**,
- the synthetic generator should already emit future-useful per-instance metadata,
- the codebase must be modular enough that the downstream classifier can be added later without restructuring the repo.

---

## 4. Target data characteristics

### Input image size
Base microscope images are **1920 × 1200** RGB.

The training pipeline may use:
- full-resolution training,
- random crops,
- tiled training,
- or controlled resize policies,

but the code must treat **1920×1200** as the canonical raw input format.

### Available real data at this stage
Available initially:
- curated **unlabeled** microscope images.

Not assumed available initially:
- real flake masks,
- real material labels,
- real thickness labels.

### Synthetic data requirement
Synthetic data generation is a core deliverable of this stage.

The synthetic generator must:
- consume curated unlabeled real microscope images as background/style source,
- overlay procedurally generated synthetic flakes,
- output exact instance masks,
- output COCO-format annotations,
- emit auxiliary metadata for each synthetic flake for future classifier work.

---

## 5. Primary learning objective

Train a **class-agnostic instance segmentation model** that answers:

> “Where are the flake-like instances in this microscope image?”

The model must **not** predict material class or thickness in this stage.

### Desired properties of the pretrained model
- robust to substrate/background variability,
- robust to nuisance changes in illumination and color,
- reusable across multiple material systems,
- efficient enough to support future inference around 1 image/s on modest hardware,
- scalable with two model sizes:
  - **small**
  - **large**

---

## 6. Platform and framework requirements

### Package management
Use **uv** for package and environment management.

### Framework
Use **PyTorch**.

### Hardware support
The code must be modular and portable across:
- Apple Silicon via **MPS**,
- NVIDIA via **CUDA**,
- CPU fallback,
- future supportable backends where PyTorch permits.

Do not hard-code Apple-only or CUDA-only logic into the training stack.

### Design principle
Apple Silicon is the likely first training device, but the code must be written as a portable PyTorch project with backend selection abstracted cleanly.

---

## 7. Architecture direction

## 7.1 Model family
Implement a **class-agnostic instance segmentation** training pipeline with two variants:

- `small`
- `large`

The exact detector family may be chosen during implementation, but it must satisfy:

- strong support in PyTorch,
- practical trainability on Apple Silicon,
- exportability and maintainability,
- compatibility with COCO-style datasets,
- support for mask prediction per instance.

### Acceptable examples
- Mask R-CNN style models,
- other modern instance segmentation architectures if they remain practical and maintainable.

Avoid exotic research-only implementations that are brittle or difficult to reproduce.

## 7.2 Scaling requirement
The repo must define a single architecture family with two model configs:

- `small`: speed-leaning default
- `large`: accuracy-leaning default

The scaling knobs should be config-driven, not forked code.

---

## 8. Synthetic data generation requirements

This is a first-class subsystem, not a placeholder.

## 8.1 Purpose
Synthetic generation provides the supervision needed to pretrain the detector before manually labeled real data exist.

## 8.2 Inputs
- curated real unlabeled microscope images,
- config-defined flake generation priors,
- augmentation and rendering parameters,
- optional substrate/background statistics estimated from real images.

## 8.3 Outputs
For each synthetic sample, generate:

- rendered RGB image,
- per-instance masks,
- bounding boxes,
- COCO annotation manifest,
- per-instance metadata for future downstream classifier training.

## 8.4 Required per-instance synthetic metadata
Even though this stage does not use it for training, synthetic generator outputs must include fields such as:

- synthetic instance id,
- polygon or raster mask reference,
- bounding box,
- z-order / overlap metadata,
- approximate synthetic thickness label,
- synthetic material tag,
- shape family / generator seed,
- contrast/render parameters,
- local background statistics used for rendering.

These fields are for future use and reproducibility.

## 8.5 Rendering requirements
The synthetic generator should model, at minimum:

- irregular flake boundaries,
- polygonal and fractured shapes,
- interior intensity/color variation,
- mild texture or nonuniformity,
- edge sharpness variation,
- overlap or near-contact between instances,
- local illumination variability,
- blur / focus variation,
- sensor noise,
- vignetting or field shading,
- scale variation.

The generator does not need to be physically perfect, but it must be configurable and systematic rather than ad hoc.

## 8.6 Real-image usage
Curated real unlabeled images should be used for:
- background sampling,
- style reference,
- global color/illumination statistics,
- optional normalization prior estimation.

Agents must not require hand-labeled real masks for the initial synthetic pretraining pipeline.

---

## 9. Data format

Standardize on **COCO-style instance segmentation**.

### Requirement
All supervised detector datasets used in this stage must be representable through a COCO-style manifest.

### Internal storage
The implementation may use raster masks internally, but the dataset layer must expose COCO-compatible instance structure.

### Why
This keeps the project interoperable and future-proof for:
- instance segmentation training,
- evaluation,
- visual inspection,
- later integration with labeling tools.

---

## 10. Recommended training strategy

This section defines the intended strategy, not necessarily a rigid algorithm.

## 10.1 Base path
Pretraining should rely primarily on:

- synthetic instance-segmentation supervision,
- optionally self-supervised representation learning or auxiliary unsupervised objectives on real unlabeled images.

## 10.2 Optional real-image pretraining
If useful and cleanly modular, the repo may include an optional pretraining stage on real unlabeled microscope images, such as:
- self-supervised encoder pretraining,
- masked image modeling,
- contrastive learning,
- reconstruction-based pretext tasks.

This must remain optional and must not block the main synthetic supervised path.

## 10.3 Mandatory supervised detector pretraining
The core deliverable is supervised pretraining of the instance detector on synthetic data.

## 10.4 Validation
Validation must support:
- synthetic held-out validation,
- future real validation hooks,
- visual debugging of predicted masks,
- artifact logging.

---

## 11. Deliverables of this stage

At minimum, this stage must produce:

1. **reproducible training code**
2. **synthetic data generation pipeline**
3. **trained checkpoints**
4. **training configs for small and large variants**
5. **evaluation report artifacts**
6. **exported embeddings/features for analysis**
7. **visualization outputs**
8. **experiment metadata sufficient for reruns**

### Required exported artifacts
For each trained run, save:

- model checkpoint,
- optimizer and scheduler state if desired,
- config snapshot,
- git commit hash if available,
- environment summary,
- dataset manifest/version summary,
- training metrics,
- validation metrics,
- prediction visualizations,
- optional extracted backbone embeddings for a selected validation subset.

---

## 12. Non-functional requirements

### Reproducibility
Every run must be reproducible enough to recover:
- exact config,
- exact code revision,
- random seed,
- dataset generation settings.

### Modularity
Synthetic generation, dataset IO, model definition, training loop, evaluation, and export must be separate modules.

### Maintainability
Prefer boring, explicit code over clever abstractions.

### Observability
Training must expose:
- losses,
- validation metrics,
- sample visualizations,
- configuration snapshots,
- failure logs.

### Portability
No backend-specific hacks should leak across the codebase.

---

# 13. Repository layout

Use the following structure.

```text
flake-pretrain/
├─ pyproject.toml
├─ uv.lock
├─ README.md
├─ .python-version
├─ .gitignore
├─ Makefile
├─ configs/
│  ├─ base/
│  │  ├─ runtime.yaml
│  │  ├─ paths.yaml
│  │  ├─ logging.yaml
│  │  └─ hardware.yaml
│  ├─ data/
│  │  ├─ real_unlabeled.yaml
│  │  ├─ synthetic_generator.yaml
│  │  ├─ synthetic_dataset.yaml
│  │  └─ coco_schema.yaml
│  ├─ model/
│  │  ├─ detector_small.yaml
│  │  ├─ detector_large.yaml
│  │  └─ export.yaml
│  ├─ train/
│  │  ├─ pretrain_supervised_small.yaml
│  │  ├─ pretrain_supervised_large.yaml
│  │  ├─ pretrain_ssl_small.yaml
│  │  └─ pretrain_ssl_large.yaml
│  ├─ eval/
│  │  ├─ synthetic_val.yaml
│  │  └─ embedding_export.yaml
│  └─ experiment/
│     ├─ small_default.yaml
│     └─ large_default.yaml
├─ data/
│  ├─ raw/
│  │  ├─ real_unlabeled/
│  │  │  ├─ images/
│  │  │  └─ manifests/
│  │  └─ external/
│  ├─ interim/
│  │  ├─ real_index/
│  │  ├─ stats/
│  │  └─ previews/
│  ├─ synthetic/
│  │  ├─ images/
│  │  ├─ masks/
│  │  ├─ annotations/
│  │  ├─ metadata/
│  │  └─ previews/
│  ├─ processed/
│  │  ├─ coco_train/
│  │  ├─ coco_val/
│  │  └─ subsets/
│  └─ artifacts/
│     ├─ reports/
│     └─ audits/
├─ scripts/
│  ├─ bootstrap.sh
│  ├─ prepare_real_index.py
│  ├─ compute_real_stats.py
│  ├─ generate_synthetic_dataset.py
│  ├─ validate_coco_dataset.py
│  ├─ train_detector.py
│  ├─ train_ssl.py
│  ├─ evaluate_detector.py
│  ├─ export_embeddings.py
│  ├─ export_checkpoint.py
│  ├─ visualize_samples.py
│  └─ smoke_test.py
├─ src/
│  └─ flake_pretrain/
│     ├─ __init__.py
│     ├─ cli/
│     │  ├─ __init__.py
│     │  └─ main.py
│     ├─ config/
│     │  ├─ __init__.py
│     │  ├─ loader.py
│     │  └─ schema.py
│     ├─ utils/
│     │  ├─ __init__.py
│     │  ├─ seed.py
│     │  ├─ paths.py
│     │  ├─ io.py
│     │  ├─ image.py
│     │  ├─ logging.py
│     │  ├─ device.py
│     │  └─ profiling.py
│     ├─ data/
│     │  ├─ __init__.py
│     │  ├─ manifests.py
│     │  ├─ coco.py
│     │  ├─ dataset_real.py
│     │  ├─ dataset_synthetic.py
│     │  ├─ transforms.py
│     │  ├─ samplers.py
│     │  └─ stats.py
│     ├─ synthetic/
│     │  ├─ __init__.py
│     │  ├─ generator.py
│     │  ├─ compositor.py
│     │  ├─ geometry.py
│     │  ├─ renderer.py
│     │  ├─ materials.py
│     │  ├─ backgrounds.py
│     │  ├─ noise.py
│     │  ├─ metadata.py
│     │  └─ validate.py
│     ├─ model/
│     │  ├─ __init__.py
│     │  ├─ registry.py
│     │  ├─ builder.py
│     │  ├─ detector.py
│     │  ├─ backbones.py
│     │  ├─ heads.py
│     │  ├─ losses.py
│     │  ├─ postprocess.py
│     │  └─ export.py
│     ├─ train/
│     │  ├─ __init__.py
│     │  ├─ engine.py
│     │  ├─ loops.py
│     │  ├─ optim.py
│     │  ├─ schedulers.py
│     │  ├─ ema.py
│     │  ├─ amp.py
│     │  ├─ checkpoints.py
│     │  └─ hooks.py
│     ├─ ssl/
│     │  ├─ __init__.py
│     │  ├─ objectives.py
│     │  ├─ datasets.py
│     │  └─ pretrain.py
│     ├─ eval/
│     │  ├─ __init__.py
│     │  ├─ metrics.py
│     │  ├─ evaluator.py
│     │  ├─ visualizer.py
│     │  ├─ embedding.py
│     │  └─ reports.py
│     └─ experiments/
│        ├─ __init__.py
│        ├─ run_metadata.py
│        └─ tracking.py
├─ tests/
│  ├─ test_config.py
│  ├─ test_coco_io.py
│  ├─ test_device.py
│  ├─ test_synthetic_generator.py
│  ├─ test_dataset_shapes.py
│  ├─ test_model_build.py
│  ├─ test_train_smoke.py
│  └─ test_eval_smoke.py
├─ docs/
│  ├─ architecture.md
│  ├─ synthetic_data.md
│  ├─ training.md
│  ├─ evaluation.md
│  ├─ reproducibility.md
│  └─ agent_tasks.md
└─ outputs/
   ├─ runs/
   ├─ checkpoints/
   ├─ embeddings/
   ├─ visualizations/
   └─ reports/
```

---

# 14. File and module responsibilities

## Root files

### `pyproject.toml`
Must define:
- project metadata,
- `uv` dependencies,
- optional dependency groups,
- console entry points,
- formatting/lint/test tool config where useful.

### `README.md`
Must explain:
- project purpose,
- scope boundary,
- setup with `uv`,
- how to generate synthetic data,
- how to train small/large models,
- how to evaluate,
- what is intentionally not implemented yet.

### `Makefile`
Simple developer shortcuts only. Examples:
- install
- lint
- test
- synth-generate
- train-small
- train-large
- eval-small

---

## Configs

Config must drive all major choices:
- data paths,
- generator priors,
- model scale,
- training hyperparameters,
- device selection,
- logging,
- export behavior.

No hidden hard-coded experiment logic.

---

## `src/flake_pretrain/synthetic/`
This package is responsible for synthetic sample creation.

### Required behavior
- sample background regions from real unlabeled data,
- generate 1..N synthetic flakes,
- render them onto background,
- produce masks and metadata,
- serialize COCO annotations,
- support deterministic regeneration via seed.

### Important rule
Keep generation logic decomposed into:
- geometry,
- rendering,
- compositing,
- metadata generation.

Do not bury everything in one script.

---

## `src/flake_pretrain/data/`
This package manages:
- raw real image indexing,
- synthetic dataset loading,
- COCO dataset parsing,
- augmentation/transforms,
- train/val split logic,
- data statistics.

---

## `src/flake_pretrain/model/`
This package defines:
- detector registry,
- small and large model variants,
- backbone/head construction,
- postprocessing,
- export helpers.

Keep model-family-specific logic isolated behind the builder/registry pattern.

---

## `src/flake_pretrain/train/`
This package defines:
- training loop,
- optimizer/scheduler setup,
- mixed precision policy,
- checkpointing,
- hooks and logging,
- resumption logic.

---

## `src/flake_pretrain/ssl/`
Optional module for self-supervised or unsupervised real-image pretraining.
Must be isolated and optional.

This must not complicate the core synthetic supervised path.

---

## `src/flake_pretrain/eval/`
This package defines:
- instance segmentation metrics,
- qualitative visualizations,
- evaluation reports,
- embedding export.

---

## `docs/agent_tasks.md`
This file must decompose the project into parallelizable agent tasks.

A suggested decomposition is given below.

---

# 15. Parallelizable AI-agent work breakdown

The project should be organized so different agents can work independently with minimal collision.

## Agent A: repository scaffolding and environment
Owns:
- `pyproject.toml`
- uv setup
- CLI skeleton
- config loading
- Makefile
- test scaffolding

## Agent B: real-data indexing and statistics
Owns:
- `prepare_real_index.py`
- `compute_real_stats.py`
- `src/.../data/manifests.py`
- `src/.../data/stats.py`

Deliverables:
- manifest format for unlabeled real images,
- image statistics report,
- background sampling support.

## Agent C: synthetic geometry and rendering
Owns:
- `synthetic/geometry.py`
- `synthetic/renderer.py`
- `synthetic/noise.py`
- unit tests for shape/rendering integrity.

Deliverables:
- synthetic flakes with masks and seeds,
- configurable appearance variability.

## Agent D: synthetic compositing and metadata
Owns:
- `synthetic/compositor.py`
- `synthetic/metadata.py`
- COCO annotation writer integration.

Deliverables:
- final synthetic scenes,
- per-instance metadata,
- COCO export.

## Agent E: dataset and transform pipeline
Owns:
- `data/coco.py`
- `data/dataset_synthetic.py`
- `data/transforms.py`
- dataset validation.

Deliverables:
- correct instance-segmentation dataloaders,
- shape and schema tests.

## Agent F: model implementation
Owns:
- `model/builder.py`
- `model/detector.py`
- `model/backbones.py`
- `model/heads.py`
- config-driven small/large variants.

Deliverables:
- model family build path,
- forward pass smoke tests.

## Agent G: training engine
Owns:
- `train/engine.py`
- `train/loops.py`
- `train/optim.py`
- `train/checkpoints.py`
- resume and logging hooks.

Deliverables:
- stable train loop,
- reproducible checkpoints,
- support for small/large configs.

## Agent H: evaluation and reporting
Owns:
- `eval/metrics.py`
- `eval/evaluator.py`
- `eval/visualizer.py`
- `eval/reports.py`
- `export_embeddings.py`

Deliverables:
- validation metrics,
- qualitative output grids,
- embedding export and report bundle.

## Agent I: optional SSL path
Owns:
- `ssl/`
- `scripts/train_ssl.py`
- integration docs.

Deliverables:
- optional real-image encoder pretraining path,
- no disruption to core supervised path.

---

# 16. Command-line interface requirements

Provide a single CLI entry point, for example:

```bash
uv run flake-pretrain <command> [options]
```

Required commands:

```bash
uv run flake-pretrain prepare-real-index
uv run flake-pretrain compute-real-stats
uv run flake-pretrain generate-synthetic
uv run flake-pretrain validate-coco
uv run flake-pretrain train-detector
uv run flake-pretrain train-ssl
uv run flake-pretrain evaluate
uv run flake-pretrain export-embeddings
uv run flake-pretrain visualize
uv run flake-pretrain smoke-test
```

Each command must accept config paths rather than relying on hidden defaults.

---

# 17. Configuration requirements

All important behavior must be config-driven.

## Required configurable groups
- filesystem paths,
- model size (`small`, `large`),
- device/backend selection,
- synthetic generator priors,
- augmentation policy,
- image resize/crop policy,
- batch size,
- optimizer,
- LR schedule,
- AMP / mixed precision behavior,
- checkpoint cadence,
- eval cadence,
- export settings,
- random seed.

---

# 18. Evaluation requirements

This stage is pretraining only, so evaluation should focus on detector quality and training usefulness.

## Quantitative requirements
Support at least:
- box AP-like metrics if model family supports them,
- mask AP-like metrics,
- precision/recall summaries,
- size-bucket analysis if practical,
- latency benchmark script for small and large variants.

## Qualitative requirements
Save:
- overlay visualizations,
- predicted masks,
- false positive / false negative galleries where possible,
- synthetic-vs-prediction comparison grids.

## Embedding export
Export a selected set of backbone or pooled instance embeddings for later analysis.
These are not the final product, but they are useful for diagnosing transfer quality.

---

# 19. Hardware/backend abstraction requirements

Implement a clear device utility layer.

## Required behavior
- detect available backend,
- support explicit device override,
- run on MPS if requested and available,
- run on CUDA if available,
- otherwise CPU.

### Do not
- scatter backend checks throughout the code,
- write CUDA-only training assumptions,
- hard-code precision policies that break on MPS.

---

# 20. Testing requirements

At minimum, implement:

- config loading tests,
- COCO read/write tests,
- synthetic generator determinism test,
- synthetic mask validity test,
- model construction smoke tests,
- one-batch train smoke test,
- one-batch eval smoke test,
- device selection test.

Tests should be small and fast.

---

# 21. Documentation requirements

## `docs/architecture.md`
Describe:
- overall pretraining-only architecture,
- how this stage connects to later project stages,
- module boundaries.

## `docs/synthetic_data.md`
Describe:
- generation assumptions,
- rendering model,
- metadata schema,
- limitations.

## `docs/training.md`
Describe:
- training flow,
- configs,
- checkpoints,
- backend guidance.

## `docs/evaluation.md`
Describe:
- metrics,
- visual diagnostics,
- latency measurement.

## `docs/reproducibility.md`
Describe:
- seeds,
- environment capture,
- versioning.

## `docs/agent_tasks.md`
Describe:
- parallel task decomposition,
- interfaces between agent-owned modules,
- merge order recommendations.

---

# 22. Success criteria for this stage

This stage is successful if all of the following are true:

1. a user can set up the repo with `uv`,
2. curated real unlabeled images can be indexed and summarized,
3. a synthetic instance-segmentation dataset can be generated reproducibly,
4. COCO-format annotations are valid,
5. a small and large detector can both train end-to-end,
6. trained checkpoints and run artifacts are exported,
7. detector evaluation produces usable quantitative and qualitative outputs,
8. the project remains cleanly scoped to pretraining only,
9. downstream classifier work can be added later without reorganizing the repo.

---

# 23. Explicit implementation notes for agents

- Keep the detector **class-agnostic**.
- Preserve future classifier metadata, but do not train on it now.
- Do not invent a downstream classifier in this stage.
- Do not over-engineer for production deployment.
- Prefer maintainable PyTorch code that runs on Apple Silicon MPS first, without breaking CUDA portability.
- Use COCO-style manifests everywhere supervised data appear.
- Treat synthetic generation as a serious subsystem, not a mock.
- Respect the `small` and `large` split from the beginning.

---

# 24. Suggested first milestones

## Milestone 1
Repo scaffolding, uv setup, configs, CLI, tests skeleton.

## Milestone 2
Real-image indexing and background statistics.

## Milestone 3
Synthetic generator producing valid images, masks, COCO manifests, and metadata.

## Milestone 4
Dataset loader and validation tooling.

## Milestone 5
Small detector training end-to-end.

## Milestone 6
Large detector training end-to-end.

## Milestone 7
Evaluation, visualization, embedding export, and report bundle.
