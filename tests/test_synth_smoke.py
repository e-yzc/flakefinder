from pathlib import Path

import cv2
import numpy as np

from flake_detector.configs import default_config
from flake_detector.synth_data import generate_hybrid_dataset


def test_synth_generation_smoke(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir(parents=True, exist_ok=True)

    for i in range(2):
        image = np.clip(np.random.normal(128, 20, size=(200, 240, 3)), 0, 255).astype(np.uint8)
        cv2.imwrite(str(real_dir / f"real_{i:02d}.png"), image)

    image_index = [
        {
            "image_id": i,
            "path": str(real_dir / f"real_{i:02d}.png"),
            "width": 240,
            "height": 200,
        }
        for i in range(2)
    ]

    cfg = default_config().synth
    cfg.crop_size = (128, 128)
    cfg.train_images = 3
    cfg.val_images = 2
    cfg.max_instances = 2

    output_root = tmp_path / "generated"
    manifests = generate_hybrid_dataset(
        output_root=output_root,
        image_index=image_index,
        patch_bank=[],
        cfg=cfg,
        seed=11,
    )

    assert len(manifests["train"]) == 3
    assert len(manifests["val"]) == 2
    assert (output_root / "train" / "manifest.json").exists()
    assert (output_root / "val" / "manifest.json").exists()
