from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
WINDOW_NAME = "flake-annotator"


@dataclass(frozen=True)
class LabelPaths:
    image_path: Path
    mask_path: Path


@dataclass
class SessionState:
    source_path: Path
    image: np.ndarray
    mask: np.ndarray
    points: list[tuple[int, int]]
    label_paths: LabelPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive flake annotation app")
    parser.add_argument("--source-dir", type=str, default="data/real", help="Directory that contains source images")
    parser.add_argument("--labeled-dir", type=str, default="labeled", help="Directory to store labeled outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image sampling order")
    return parser.parse_args()


def _iter_images(source_dir: Path, extensions: Iterable[str]) -> list[Path]:
    ext_set = {ext.lower() for ext in extensions}
    images = [path for path in source_dir.rglob("*") if path.suffix.lower() in ext_set]
    return sorted(images)


def _label_paths_for_image(source_path: Path, labeled_dir: Path) -> LabelPaths:
    image_name = f"{source_path.stem}_annotated{source_path.suffix.lower()}"
    mask_name = f"{source_path.stem}_mask.png"
    return LabelPaths(
        image_path=labeled_dir / "images" / image_name,
        mask_path=labeled_dir / "masks" / mask_name,
    )


def _is_labeled(source_path: Path, labeled_dir: Path) -> bool:
    return _label_paths_for_image(source_path, labeled_dir).image_path.exists()


def _select_random_unlabeled(images: list[Path], labeled_dir: Path, rng: random.Random) -> Path | None:
    unlabeled = [path for path in images if not _is_labeled(path, labeled_dir)]
    if not unlabeled:
        return None
    return unlabeled[rng.randrange(len(unlabeled))]


def _load_session(source_path: Path, labeled_dir: Path) -> SessionState:
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read source image: {source_path}")

    label_paths = _label_paths_for_image(source_path, labeled_dir)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if label_paths.mask_path.exists():
        loaded_mask = cv2.imread(str(label_paths.mask_path), cv2.IMREAD_GRAYSCALE)
        if loaded_mask is not None and loaded_mask.shape == mask.shape:
            mask = loaded_mask

    return SessionState(source_path=source_path, image=image, mask=mask, points=[], label_paths=label_paths)


def _save_session(session: SessionState) -> None:
    session.label_paths.image_path.parent.mkdir(parents=True, exist_ok=True)
    session.label_paths.mask_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(session.label_paths.image_path), session.image)
    cv2.imwrite(str(session.label_paths.mask_path), session.mask)


def _render_view(session: SessionState) -> np.ndarray:
    overlay = session.image.copy()
    mask_bool = session.mask > 0
    if mask_bool.any():
        overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)

    for point in session.points:
        cv2.circle(overlay, point, radius=3, color=(0, 200, 255), thickness=-1)

    if len(session.points) >= 2:
        poly = np.array(session.points, dtype=np.int32)
        cv2.polylines(overlay, [poly], isClosed=False, color=(0, 200, 255), thickness=1)

    label_text = (
        f"{session.source_path.name} | polygons committed immediately | "
        "L-click add, M-click close, R-click undo, n next, r reset, q quit"
    )
    cv2.putText(overlay, label_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, label_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return overlay


def _commit_polygon(session: SessionState) -> bool:
    if len(session.points) < 3:
        return False
    poly = np.array(session.points, dtype=np.int32)
    cv2.fillPoly(session.mask, [poly], color=255)
    session.points = []
    _save_session(session)
    return True


def _run_annotation_loop(images: list[Path], labeled_dir: Path, seed: int) -> None:
    rng = random.Random(seed)

    source_path = _select_random_unlabeled(images, labeled_dir, rng)
    if source_path is None:
        print("[Annotator] No unlabeled images found.", flush=True)
        return

    session = _load_session(source_path, labeled_dir)
    _save_session(session)

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            session.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and session.points:
            session.points.pop()
        elif event == cv2.EVENT_MBUTTONDOWN:
            changed = _commit_polygon(session)
            if changed:
                print(f"[Annotator] Saved polygon for {session.source_path.name}", flush=True)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("[Annotator] Started. Press q to quit.", flush=True)
    while True:
        view = _render_view(session)
        cv2.imshow(WINDOW_NAME, view)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        if key == ord("r"):
            session.mask.fill(0)
            session.points = []
            _save_session(session)
            print(f"[Annotator] Reset mask for {session.source_path.name}", flush=True)
        if key == ord("n"):
            if session.points:
                _commit_polygon(session)
            next_path = _select_random_unlabeled(images, labeled_dir, rng)
            if next_path is None:
                print("[Annotator] Annotation complete. No unlabeled images left.", flush=True)
                break
            session = _load_session(next_path, labeled_dir)
            _save_session(session)
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)
            print(f"[Annotator] Loaded {session.source_path.name}", flush=True)

    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    labeled_dir = Path(args.labeled_dir)

    if not source_dir.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    images = _iter_images(source_dir, SUPPORTED_EXTENSIONS)
    print(f"[Annotator] Found {len(images)} candidate images in {source_dir}", flush=True)
    if not images:
        print("[Annotator] Nothing to annotate.", flush=True)
        return

    _run_annotation_loop(images, labeled_dir, seed=args.seed)


if __name__ == "__main__":
    main()
