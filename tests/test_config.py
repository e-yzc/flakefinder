from flake_detector.configs import config_from_dict, default_config
from flake_detector.utils import select_device


def test_default_config_smoke() -> None:
    cfg = default_config()
    assert cfg.model.family == "maskrcnn"
    assert cfg.model.size in {"small", "large"}
    assert cfg.synth.train_images > 0


def test_config_override_smoke() -> None:
    cfg = config_from_dict({"model": {"size": "large"}, "train": {"epochs": 2}})
    assert cfg.model.size == "large"
    assert cfg.train.epochs == 2


def test_device_selection_smoke() -> None:
    device = select_device("cpu")
    assert str(device) == "cpu"
