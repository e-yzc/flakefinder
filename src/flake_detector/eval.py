from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import draw_overlay, ensure_dirs, tensor_image_to_numpy


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def _match_masks(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], iou_thr: float = 0.5) -> tuple[int, int, int, list[float]]:
    if not pred_masks and not gt_masks:
        return 0, 0, 0, []
    if not pred_masks:
        return 0, 0, len(gt_masks), []
    if not gt_masks:
        return 0, len(pred_masks), 0, []

    matches: list[tuple[float, int, int]] = []
    for pi, p in enumerate(pred_masks):
        for gi, g in enumerate(gt_masks):
            iou = _mask_iou(p, g)
            if iou >= iou_thr:
                matches.append((iou, pi, gi))

    matches.sort(key=lambda x: x[0], reverse=True)

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matched_ious: list[float] = []

    for iou, pi, gi in matches:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matched_ious.append(iou)

    tp = len(matched_ious)
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp
    return tp, fp, fn, matched_ious


def validation_loss(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_steps: int | None = None,
) -> float:
    model.train()
    losses: list[float] = []
    total_steps = len(data_loader)
    target_steps = min(total_steps, max_steps) if max_steps is not None else total_steps
    print(f"[Eval] validation_loss started ({target_steps} batches)", flush=True)

    with torch.no_grad():
        for step, (images, targets) in enumerate(data_loader):
            if max_steps is not None and step >= max_steps:
                break
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total = sum(v.item() for v in loss_dict.values())
            losses.append(total)

            current_step = step + 1
            if current_step % 10 == 0 or current_step == target_steps:
                print(f"[Eval] validation_loss progress: {current_step}/{target_steps}", flush=True)

    if not losses:
        print("[Eval] validation_loss finished with no batches", flush=True)
        return 0.0
    print("[Eval] validation_loss finished", flush=True)
    return float(np.mean(losses))


def evaluate_mask_quality(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    score_thresh: float = 0.5,
    iou_thr: float = 0.5,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_steps = len(data_loader)
    target_steps = min(total_steps, max_steps) if max_steps is not None else total_steps
    print(f"[Eval] evaluate_mask_quality started ({target_steps} batches)", flush=True)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious: list[float] = []
    proposal_counts: list[int] = []

    with torch.no_grad():
        for step, (images, targets) in enumerate(data_loader):
            if max_steps is not None and step >= max_steps:
                break

            outputs = model([img.to(device) for img in images])

            for output, target in zip(outputs, targets):
                scores = output["scores"].detach().cpu().numpy()
                keep = scores >= score_thresh

                masks = output["masks"].detach().cpu().numpy()
                masks = masks[keep, 0] > 0.5
                pred_masks = [m.astype(np.uint8) for m in masks]
                proposal_counts.append(len(pred_masks))

                gt_masks_t = target["masks"].detach().cpu()
                if gt_masks_t.ndim == 4:
                    gt_masks_t = gt_masks_t[:, 0]
                gt_masks = [m.numpy().astype(np.uint8) for m in gt_masks_t]

                tp, fp, fn, ious = _match_masks(pred_masks, gt_masks, iou_thr=iou_thr)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                all_ious.extend(ious)

            current_step = step + 1
            if current_step % 10 == 0 or current_step == target_steps:
                print(f"[Eval] evaluate_mask_quality progress: {current_step}/{target_steps}", flush=True)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mask_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "proposal_count": float(np.mean(proposal_counts)) if proposal_counts else 0.0,
        "tp": float(total_tp),
        "fp": float(total_fp),
        "fn": float(total_fn),
    }


def save_prediction_overlays(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    output_dir: str | Path,
    max_images: int = 8,
    score_thresh: float = 0.5,
) -> None:
    ensure_dirs(output_dir)
    model.eval()

    limit = min(max_images, len(dataset))
    print(f"[Eval] Saving {limit} validation overlay previews to {output_dir}", flush=True)
    with torch.no_grad():
        for idx in range(limit):
            image_tensor, target = dataset[idx]
            output = model([image_tensor.to(device)])[0]

            scores = output["scores"].detach().cpu().numpy()
            keep = scores >= score_thresh

            pred_masks = output["masks"].detach().cpu().numpy()[keep, 0] > 0.5
            pred_boxes = output["boxes"].detach().cpu().numpy()[keep]
            pred_scores = scores[keep]

            img_np = tensor_image_to_numpy(image_tensor)
            pred_overlay = draw_overlay(img_np, pred_masks.astype(np.uint8), pred_boxes.astype(int), pred_scores.tolist())

            gt_masks_t = target["masks"].detach().cpu()
            if gt_masks_t.ndim == 4:
                gt_masks_t = gt_masks_t[:, 0]
            gt_masks = [m.numpy().astype(np.uint8) for m in gt_masks_t]
            gt_boxes = target["boxes"].detach().cpu().numpy().astype(int)
            gt_overlay = draw_overlay(img_np, gt_masks, gt_boxes)

            stacked = np.concatenate([gt_overlay, pred_overlay], axis=1)
            out_path = Path(output_dir) / f"overlay_{idx:04d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))

            if (idx + 1) % 5 == 0 or (idx + 1) == limit:
                print(f"[Eval] Overlay progress: {idx + 1}/{limit}", flush=True)


def dump_real_image_predictions(
    model: torch.nn.Module,
    image_index: list[dict[str, Any]],
    device: torch.device,
    output_dir: str | Path,
    score_thresh: float = 0.5,
    max_images: int = 8,
) -> None:
    ensure_dirs(output_dir)
    model.eval()
    limit = min(max_images, len(image_index))
    print(f"[Eval] Dumping predictions for {limit} real images to {output_dir}", flush=True)

    with torch.no_grad():
        for idx, entry in enumerate(image_index[:max_images]):
            image_bgr = cv2.imread(entry["path"], cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_tensor = torch.as_tensor(image_rgb.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            output = model([image_tensor.to(device)])[0]

            scores = output["scores"].detach().cpu().numpy()
            keep = scores >= score_thresh

            pred_masks = output["masks"].detach().cpu().numpy()[keep, 0] > 0.5
            pred_boxes = output["boxes"].detach().cpu().numpy()[keep]
            pred_scores = scores[keep]

            overlay = draw_overlay(image_rgb, pred_masks.astype(np.uint8), pred_boxes.astype(int), pred_scores.tolist())
            out_path = Path(output_dir) / f"real_pred_{idx:04d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if (idx + 1) % 5 == 0 or (idx + 1) == limit:
                print(f"[Eval] Real-image prediction progress: {idx + 1}/{limit}", flush=True)
