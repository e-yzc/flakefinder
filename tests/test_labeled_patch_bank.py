from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from flake_detector.mining import load_labeled_patch_bank


def test_load_labeled_patch_bank_reads_pairs(tmp_path: Path) -> None:
    labeled_dir = tmp_path / "labeled"
    images_dir = labeled_dir / "images"
    masks_dir = labeled_dir / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    image_path = images_dir / "img_annotated.png"
    mask_path = masks_dir / "img_mask.png"

    cv2.imwrite(str(image_path), np.full((32, 32, 3), 120, dtype=np.uint8))
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 10:22] = 255
    cv2.imwrite(str(mask_path), mask)

    entries = load_labeled_patch_bank(labeled_dir)

    assert len(entries) == 1
    assert entries[0]["patch_image_path"] == str(image_path)
    assert entries[0]["patch_mask_path"] == str(mask_path)
    assert entries[0]["source_type"] == "manual_label"
    assert entries[0]["patch_bbox"] == [10, 8, 21, 23]


def test_load_labeled_patch_bank_skips_non_matching_names(tmp_path: Path) -> None:
    labeled_dir = tmp_path / "labeled"
    images_dir = labeled_dir / "images"
    masks_dir = labeled_dir / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    cv2.imwrite(str(images_dir / "img.png"), np.full((16, 16, 3), 120, dtype=np.uint8))
    cv2.imwrite(str(masks_dir / "img_mask.png"), np.full((16, 16), 255, dtype=np.uint8))

    entries = load_labeled_patch_bank(labeled_dir)

    assert entries == []
