from pathlib import Path
import json

import cv2
import numpy as np

from flake_detector.data import HybridInstanceDataset


def test_dataset_loading_smoke(tmp_path: Path) -> None:
    image = np.full((96, 96, 3), 120, dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    cv2.rectangle(mask, (20, 25), (60, 70), 255, -1)

    image_path = tmp_path / "sample.png"
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)

    manifest = [
        {
            "image_id": "train_000001",
            "image_path": str(image_path),
            "width": 96,
            "height": 96,
            "instances": [
                {
                    "instance_id": 0,
                    "category_id": 1,
                    "bbox": [20, 25, 60, 70],
                    "mask_path": str(mask_path),
                    "source_type": "procedural",
                }
            ],
        },
        {
            "image_id": "train_000002",
            "image_path": str(image_path),
            "width": 96,
            "height": 96,
            "instances": [],
        },
    ]

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dataset = HybridInstanceDataset(manifest_path, train=True, seed=7)
    assert len(dataset) == 2

    image_tensor, target = dataset[0]
    assert image_tensor.shape == (3, 96, 96)
    assert target["boxes"].shape[0] == 1
    assert target["masks"].shape[0] == 1

    image_tensor2, target2 = dataset[1]
    assert image_tensor2.shape == (3, 96, 96)
    assert target2["boxes"].shape[0] == 0
