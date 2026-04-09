from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flake_detector.configs import load_config, save_config_snapshot, to_dict  # noqa: E402
from flake_detector.data import HybridInstanceDataset  # noqa: E402
from flake_detector.eval import (  # noqa: E402
    dump_real_image_predictions,
    evaluate_mask_quality,
    save_prediction_overlays,
    validation_loss,
)
from flake_detector.models import build_model  # noqa: E402
from flake_detector.ssl import run_optional_ssl  # noqa: E402
from flake_detector.utils import (  # noqa: E402
    append_jsonl,
    collate_fn,
    ensure_dirs,
    load_json,
    save_checkpoint,
    save_json,
    select_device,
    set_seed,
    utc_timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train class-agnostic flake instance segmentation model")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file")
    parser.add_argument("--model-size", type=str, default=None, choices=["small", "large"], help="Override model size")
    parser.add_argument("--device", type=str, default=None, help="Override device: cuda/mps/cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    return parser.parse_args()


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _measure_latency_ms(model: torch.nn.Module, dataset: HybridInstanceDataset, device: torch.device, n: int = 8) -> float:
    model.eval()
    if len(dataset) == 0:
        return 0.0

    limit = min(n, len(dataset))
    elapsed: list[float] = []
    print(f"[Eval] Running latency benchmark on {limit} samples", flush=True)

    with torch.no_grad():
        for i in range(limit):
            image, _ = dataset[i]
            _synchronize_if_needed(device)
            start = time.perf_counter()
            _ = model([image.to(device)])
            _synchronize_if_needed(device)
            elapsed.append((time.perf_counter() - start) * 1000.0)

            if (i + 1) % 4 == 0 or (i + 1) == limit:
                print(f"[Eval] Latency benchmark progress: {i + 1}/{limit}", flush=True)

    return float(sum(elapsed) / len(elapsed)) if elapsed else 0.0


def _load_manifests(cfg) -> tuple[Path, Path]:
    train_a = Path(cfg.paths.splits_dir) / "train_manifest.json"
    val_a = Path(cfg.paths.splits_dir) / "val_manifest.json"

    train_b = Path(cfg.paths.generated_dir) / "train" / "manifest.json"
    val_b = Path(cfg.paths.generated_dir) / "val" / "manifest.json"

    if train_a.exists() and val_a.exists():
        return train_a, val_a
    if train_b.exists() and val_b.exists():
        return train_b, val_b

    raise FileNotFoundError(
        "Manifest files not found. Run build_dataset.py first to create train/val manifests."
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.model_size is not None:
        cfg.model.size = args.model_size
    if args.device is not None:
        cfg.train.device_override = args.device
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size

    set_seed(cfg.seed)
    device = select_device(cfg.train.device_override)
    print("[Setup] Starting training run", flush=True)
    print(
        f"[Setup] model_size={cfg.model.size} device={device} epochs={cfg.train.epochs} batch_size={cfg.train.batch_size}",
        flush=True,
    )

    ensure_dirs(
        cfg.paths.checkpoints_dir,
        cfg.paths.metrics_dir,
        cfg.paths.previews_dir,
        cfg.paths.reports_dir,
        cfg.paths.pseudo_labels_dir,
    )

    run_id = f"{cfg.experiment_name}_{cfg.model.size}_{utc_timestamp()}"
    run_report_dir = Path(cfg.paths.reports_dir) / run_id
    run_preview_dir = Path(cfg.paths.previews_dir) / run_id
    ensure_dirs(run_report_dir, run_preview_dir)

    save_config_snapshot(cfg, run_report_dir / "train_config_snapshot.json")
    save_json({"seed": cfg.seed, "device": str(device)}, run_report_dir / "reproducibility.json")

    train_manifest_path, val_manifest_path = _load_manifests(cfg)
    train_dataset = HybridInstanceDataset(train_manifest_path, train=True, seed=cfg.seed)
    val_dataset = HybridInstanceDataset(val_manifest_path, train=False, seed=cfg.seed)
    print(
        f"[Data] Loaded manifests: train={train_manifest_path} val={val_manifest_path}",
        flush=True,
    )
    print(f"[Data] Dataset sizes: train={len(train_dataset)} val={len(val_dataset)}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
    )

    if cfg.ssl.enabled:
        print("[SSL] Running optional SSL pretraining...", flush=True)
        image_index_path = Path(cfg.paths.splits_dir) / "image_index.json"
        image_index = load_json(image_index_path) if image_index_path.exists() else []
        ssl_report = run_optional_ssl(cfg.ssl, image_index, device, run_report_dir / "ssl", seed=cfg.seed)
        save_json(ssl_report, run_report_dir / "ssl_summary.json")

    print("[Model] Building model and optimizer...", flush=True)
    model = build_model(family=cfg.model.family, size=cfg.model.size, num_classes=cfg.model.num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    use_amp = bool(cfg.train.mixed_precision) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    metrics_path = Path(cfg.paths.metrics_dir) / f"{run_id}.jsonl"

    best_f1 = -1.0
    best_epoch = -1
    print(f"[Train] Starting supervised training for {cfg.train.epochs} epochs", flush=True)

    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()
        epoch_losses: list[float] = []
        print(f"[Train] Epoch {epoch}/{cfg.train.epochs} started", flush=True)

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.train.epochs}", leave=False)
        for step, (images, targets) in enumerate(progress):
            if cfg.train.max_train_steps_per_epoch is not None and step >= cfg.train.max_train_steps_per_epoch:
                break

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", enabled=True):
                    loss_dict = model(images, targets)
                    loss = sum(loss_value for loss_value in loss_dict.values())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                loss = sum(loss_value for loss_value in loss_dict.values())
                loss.backward()
                optimizer.step()

            loss_value = float(loss.item())
            epoch_losses.append(loss_value)
            progress.set_postfix({"loss": f"{loss_value:.4f}"})

        train_loss = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else 0.0
        print(f"[Eval] Epoch {epoch}: computing validation loss...", flush=True)
        val_loss = validation_loss(model, val_loader, device, max_steps=cfg.train.max_val_steps)
        print(f"[Eval] Epoch {epoch}: computing mask quality metrics...", flush=True)
        val_metrics = evaluate_mask_quality(
            model,
            val_loader,
            device,
            score_thresh=cfg.model.score_thresh,
            max_steps=cfg.train.max_val_steps,
        )

        metrics_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        append_jsonl(metrics_row, metrics_path)

        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": to_dict(cfg),
            "metrics": metrics_row,
        }

        ckpt_path = Path(cfg.paths.checkpoints_dir) / f"{run_id}_epoch_{epoch:03d}.pt"
        save_checkpoint(checkpoint_payload, ckpt_path)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            save_checkpoint(checkpoint_payload, Path(cfg.paths.checkpoints_dir) / f"{run_id}_best.pt")
            print(f"[Checkpoint] New best model at epoch {epoch} with f1={best_f1:.4f}", flush=True)

        if epoch == int(cfg.train.epochs):
            print("[Eval] Saving validation overlay previews...", flush=True)
            save_prediction_overlays(
                model,
                val_dataset,
                device,
                run_preview_dir / "val_overlays",
                max_images=cfg.train.preview_count,
                score_thresh=cfg.model.score_thresh,
            )

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} f1={val_metrics['f1']:.4f}", flush=True)

    image_index_path = Path(cfg.paths.splits_dir) / "image_index.json"
    if image_index_path.exists():
        print("[Eval] Generating predictions on real unlabeled images...", flush=True)
        image_index = load_json(image_index_path)
        dump_real_image_predictions(
            model,
            image_index,
            device,
            run_preview_dir / "real_sanity",
            score_thresh=cfg.model.score_thresh,
            max_images=cfg.train.preview_count,
        )
    else:
        print("[Eval] Skipping real-image sanity predictions (image index not found)", flush=True)

    print("[Eval] Measuring final latency...", flush=True)
    latency_ms = _measure_latency_ms(model, val_dataset, device)

    report = {
        "run_id": run_id,
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "latency_ms": latency_ms,
        "device": str(device),
    }
    save_json(report, run_report_dir / "run_summary.json")

    print("Training complete", flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()
