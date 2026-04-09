import torch

from flake_detector.models import build_model


def test_model_forward_smoke() -> None:
    model = build_model(family="maskrcnn", size="small", num_classes=2)
    model.eval()

    with torch.no_grad():
        outputs = model([torch.rand(3, 128, 128)])

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert "boxes" in outputs[0]
    assert "masks" in outputs[0]
