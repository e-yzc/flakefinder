import torch

from flake_detector.models import build_model


def test_single_train_step_smoke() -> None:
    model = build_model(family="maskrcnn", size="small", num_classes=2)
    model.train()

    image = torch.rand(3, 128, 128)
    mask = torch.zeros((1, 128, 128), dtype=torch.uint8)
    mask[:, 30:85, 40:92] = 1

    target = {
        "boxes": torch.tensor([[40.0, 30.0, 91.0, 84.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": mask,
        "image_id": torch.tensor([0], dtype=torch.int64),
        "area": torch.tensor([51.0 * 54.0], dtype=torch.float32),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    loss_dict = model([image], [target])
    loss = sum(v for v in loss_dict.values())
    assert torch.isfinite(loss)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
