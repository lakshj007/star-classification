#!/usr/bin/env python3
"""Train a small CNN to detect stars in FITS images."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from astropy.io import fits
import cv2
from scipy.ndimage import maximum_filter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN to detect stars in FITS files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Practice Image Sets"),
        help="Directory that contains FITS files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Where to store trained model and metrics.",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=32,
        help="Square patch edge length extracted around candidate stars.",
    )
    parser.add_argument(
        "--pos-per-image",
        type=int,
        default=256,
        help="Maximum number of positive patches per FITS frame.",
    )
    parser.add_argument(
        "--neg-per-image",
        type=int,
        default=256,
        help="Maximum number of background patches per FITS frame.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Sigma threshold for selecting bright peaks as positives.",
    )
    parser.add_argument(
        "--candidate-method",
        choices=["opencv", "localmax"],
        default="opencv",
        help="Method for mining star candidates: OpenCV pipeline or legacy local-max search.",
    )
    parser.add_argument(
        "--cv-blur-ksize",
        type=int,
        default=7,
        help="Odd kernel size for Gaussian blur when using the OpenCV candidate miner.",
    )
    parser.add_argument(
        "--cv-thresh-factor",
        type=float,
        default=1.4,
        help="Multiplier applied to the std-dev based threshold in the OpenCV miner.",
    )
    parser.add_argument(
        "--cv-thresh-offset",
        type=float,
        default=8.0,
        help="Offset subtracted inside the adaptive threshold (higher -> more detections).",
    )
    parser.add_argument(
        "--cv-min-area",
        type=float,
        default=2.0,
        help="Minimum contour area (in pixels) to accept as a star when using OpenCV.",
    )
    parser.add_argument(
        "--cv-max-area",
        type=float,
        default=160.0,
        help="Maximum contour area (in pixels) to accept as a star when using OpenCV.",
    )
    parser.add_argument(
        "--cv-min-distance",
        type=float,
        default=5.0,
        help="Minimum center-to-center distance (pixels) when deduplicating OpenCV detections.",
    )
    parser.add_argument(
        "--cv-tile-size",
        type=int,
        default=128,
        help="Tile edge (pixels) used to cap detections per region in the OpenCV miner.",
    )
    parser.add_argument(
        "--cv-max-per-tile",
        type=int,
        default=12,
        help="Maximum detections per tile to avoid single clusters dominating positives.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of patches reserved for validation.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    parser.add_argument(
        "--predict",
        type=Path,
        default=None,
        help="Optional FITS file to score after training. Saves probability map next to the weights.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=12,
        help="Stride (pixels) used when sliding the network during prediction.",
    )
    parser.add_argument(
        "--detect-threshold",
        type=float,
        default=0.85,
        help="Probability cut-off used to report detections during prediction.",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="When --predict is provided, also save a PNG overlay marking detected stars.",
    )
    parser.add_argument(
        "--show-overlay",
        action="store_true",
        help="Display the detection overlay in an interactive window when predicting.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda or cpu).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_fits_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    return sorted(data_dir.rglob("*.fits"))


def load_fits(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
    data = np.nan_to_num(data)
    return data


def robust_normalize(image: np.ndarray) -> np.ndarray:
    median = np.median(image)
    mad = np.median(np.abs(image - median)) + 1e-6
    normalized = (image - median) / mad
    return np.clip(normalized, -5.0, 25.0)


def local_maxima_mask(image: np.ndarray, threshold: float, footprint: int = 7) -> np.ndarray:
    footprint = max(3, footprint)
    if footprint % 2 == 0:
        footprint += 1
    maxed = maximum_filter(image, size=footprint)
    mask = (image == maxed) & (image > threshold)
    return mask


def valid_center(coord: Tuple[int, int], shape: Tuple[int, int], patch_size: int) -> bool:
    half = patch_size // 2
    y, x = coord
    return (y - half) >= 0 and (x - half) >= 0 and (y + half) < shape[0] and (x + half) < shape[1]


def crop_patch(image: np.ndarray, center: Tuple[int, int], patch_size: int) -> np.ndarray:
    half = patch_size // 2
    y, x = center
    patch = image[y - half : y + half, x - half : x + half]
    return patch.copy()


def coords_to_mask(shape: Tuple[int, int], coords: Sequence[Tuple[int, int]], radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for y, x in coords:
        y0 = max(0, y - radius)
        y1 = min(shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(shape[1], x + radius + 1)
        mask[y0:y1, x0:x1] = True
    return mask


def ensure_odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, value)
    return value if value % 2 == 1 else value + 1


def detect_stars_opencv(
    norm_image: np.ndarray,
    patch_size: int,
    params: CVCandidateConfig,
) -> List[Tuple[int, int, float]]:
    scaled = norm_image - np.min(norm_image)
    scaled /= np.max(scaled) + 1e-6
    scaled_uint8 = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    ksize = ensure_odd(params.blur_ksize)
    blurred = cv2.GaussianBlur(scaled_uint8, (ksize, ksize), 0)
    enhanced = cv2.subtract(scaled_uint8, blurred)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(enhanced)
    adaptive = cv2.adaptiveThreshold(
        clahe_img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        ensure_odd(15),
        -params.thresh_offset,
    )
    mu = float(np.mean(enhanced))
    sigma = float(np.std(enhanced) + 1e-6)
    global_thresh = int(np.clip(mu + params.thresh_factor * sigma, 0, 255))
    _, binary = cv2.threshold(enhanced, global_thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(adaptive, binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[int, int, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params.min_area or area > params.max_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if not valid_center((cy, cx), norm_image.shape, patch_size):
            continue
        score = float(enhanced[cy, cx]) / 255.0
        candidates.append((cy, cx, score))

    candidates.sort(key=lambda c: c[2], reverse=True)
    filtered: List[Tuple[int, int, float]] = []
    tile_counts: dict[Tuple[int, int], int] = defaultdict(int)
    tile_size = max(32, params.tile_size)
    min_dist_sq = params.min_distance ** 2
    for cy, cx, score in candidates:
        tile = (cy // tile_size, cx // tile_size)
        if tile_counts[tile] >= params.max_per_tile:
            continue
        too_close = False
        for py, px, _ in filtered:
            if (cy - py) ** 2 + (cx - px) ** 2 < min_dist_sq:
                too_close = True
                break
        if not too_close:
            filtered.append((cy, cx, score))
            tile_counts[tile] += 1
    return filtered


def sample_opencv_coords(
    norm_image: np.ndarray,
    patch_size: int,
    max_per_image: int,
    params: CVCandidateConfig,
) -> List[Tuple[int, int]]:
    detected = detect_stars_opencv(norm_image, patch_size, params)
    coords = [(cy, cx) for cy, cx, _ in detected[:max_per_image]]
    return coords


def sample_positive_coords(
    norm_image: np.ndarray,
    patch_size: int,
    threshold: float,
    max_per_image: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    maxima = local_maxima_mask(norm_image, threshold=threshold)
    coords = np.argwhere(maxima)
    rng.shuffle(coords)
    positives: List[Tuple[int, int]] = []
    for y, x in coords:
        if valid_center((y, x), norm_image.shape, patch_size):
            positives.append((y, x))
        if len(positives) >= max_per_image:
            break
    return positives


def sample_negative_coords(
    norm_image: np.ndarray,
    patch_size: int,
    positives: Sequence[Tuple[int, int]],
    max_per_image: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    forbidden = coords_to_mask(norm_image.shape, positives, radius=patch_size)
    h, w = norm_image.shape
    candidates: List[Tuple[int, int]] = []
    attempts = 0
    need = max_per_image
    while len(candidates) < need and attempts < need * 20:
        attempts += 1
        y = rng.integers(patch_size // 2, h - patch_size // 2)
        x = rng.integers(patch_size // 2, w - patch_size // 2)
        if forbidden[y, x]:
            continue
        if not (-1.5 <= norm_image[y, x] <= 3.0):
            continue
        candidates.append((int(y), int(x)))
    return candidates


@dataclass
class PatchDataset:
    patches: np.ndarray
    labels: np.ndarray


@dataclass
class Detection:
    y: int
    x: int
    probability: float


@dataclass
class CVCandidateConfig:
    blur_ksize: int
    thresh_factor: float
    thresh_offset: float
    min_area: float
    max_area: float
    min_distance: float
    tile_size: int
    max_per_tile: int


def build_patch_dataset(
    fits_paths: Sequence[Path],
    patch_size: int,
    pos_per_image: int,
    neg_per_image: int,
    threshold: float,
    candidate_method: str,
    cv_params: Optional[CVCandidateConfig],
    rng: np.random.Generator,
) -> PatchDataset:
    patch_bank: List[np.ndarray] = []
    label_bank: List[float] = []
    for path in fits_paths:
        image = load_fits(path)
        norm = robust_normalize(image)
        if candidate_method == "opencv":
            if cv_params is None:
                raise ValueError("OpenCV candidate method selected but parameters are missing.")
            positives = sample_opencv_coords(norm, patch_size, pos_per_image, cv_params)
        else:
            positives = sample_positive_coords(norm, patch_size, threshold, pos_per_image, rng)
        negatives = sample_negative_coords(norm, patch_size, positives, neg_per_image, rng)
        for coord in positives:
            patch_bank.append(crop_patch(norm, coord, patch_size))
            label_bank.append(1.0)
        for coord in negatives:
            patch_bank.append(crop_patch(norm, coord, patch_size))
            label_bank.append(0.0)
        print(
            f"{path.name}: {len(positives)} star patches, {len(negatives)} background patches",
            flush=True,
        )
    patches = np.stack(patch_bank, axis=0).astype(np.float32)
    labels = np.array(label_bank, dtype=np.float32)
    return PatchDataset(patches, labels)


def train_val_split(
    dataset: PatchDataset, val_fraction: float, rng: np.random.Generator
) -> Tuple[PatchDataset, PatchDataset]:
    total = len(dataset.labels)
    indices = np.arange(total)
    rng.shuffle(indices)
    split = int(math.floor(total * (1.0 - val_fraction)))
    train_idx = indices[:split]
    val_idx = indices[split:]
    train = PatchDataset(dataset.patches[train_idx], dataset.labels[train_idx])
    val = PatchDataset(dataset.patches[val_idx], dataset.labels[val_idx])
    return train, val


class StarPatchDataset(Dataset):
    def __init__(self, patches: np.ndarray, labels: np.ndarray, augment: bool, seed: int):
        self.patches = patches
        self.labels = labels
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[idx]
        if self.augment:
            patch = self._augment(patch)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return patch_tensor, label

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        aug = patch.copy()
        if self.rng.random() < 0.5:
            aug = np.flip(aug, axis=1)
        if self.rng.random() < 0.5:
            aug = np.flip(aug, axis=0)
        k = self.rng.integers(0, 4)
        if k:
            aug = np.rot90(aug, k)
        scale = self.rng.uniform(0.9, 1.1)
        shift = self.rng.uniform(-0.1, 0.1)
        aug = aug * scale + shift
        if self.rng.random() < 0.3:
            noise = self.rng.normal(0.0, 0.05, size=aug.shape)
            aug += noise
        return aug.astype(np.float32)


class StarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        return self.classifier(features)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for patches, labels in loader:
        patches = patches.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(patches).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * patches.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        total_correct += (preds == labels.long()).sum().item()
        total_examples += patches.size(0)
    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for patches, labels in loader:
        patches = patches.to(device)
        labels = labels.to(device)
        logits = model(patches).squeeze(1)
        loss = criterion(logits, labels)
        total_loss += loss.item() * patches.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        total_correct += (preds == labels.long()).sum().item()
        total_examples += patches.size(0)
    return total_loss / total_examples, total_correct / total_examples


def slide_and_predict(
    model: nn.Module,
    image: np.ndarray,
    patch_size: int,
    stride: int,
    device: torch.device,
    detect_threshold: float,
) -> Tuple[np.ndarray, List[Detection]]:
    norm = robust_normalize(image)
    h, w = norm.shape
    scores = np.zeros((h, w), dtype=np.float32)
    detections: List[Detection] = []
    half = patch_size // 2
    centers_y = range(half, h - half, stride)
    centers_x = range(half, w - half, stride)
    model.eval()
    for y in centers_y:
        for x in centers_x:
            patch = norm[y - half : y + half, x - half : x + half]
            if patch.shape != (patch_size, patch_size):
                continue
            tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
            logit = model(tensor)
            prob = torch.sigmoid(logit).item()
            scores[y, x] = prob
            if prob >= detect_threshold:
                detections.append(Detection(y=int(y), x=int(x), probability=float(prob)))
    return scores, detections


def save_probability_map(prob_map: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(prob_map.astype(np.float32))
    hdu.writeto(output_path, overwrite=True)
    print(f"Saved probability map to {output_path}")


def save_detection_catalog(detections: Sequence[Detection], output_path: Path) -> None:
    serialized = [asdict(det) for det in detections]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(serialized, fp, indent=2)
    print(f"Saved {len(detections)} detections to {output_path}")


def render_detection_overlay(
    image: np.ndarray, detections: Sequence[Detection], max_points: int = 1000
) -> plt.Figure:
    norm = robust_normalize(image)
    fig, ax = plt.subplots(figsize=(8, 8))
    vmin, vmax = np.percentile(norm, [5, 99])
    ax.imshow(norm, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    if detections:
        sorted_dets = sorted(detections, key=lambda d: d.probability, reverse=True)
        subset = sorted_dets[:max_points]
        xs = [det.x for det in subset]
        ys = [det.y for det in subset]
        probs = [det.probability for det in subset]
        sizes = [20 + 80 * (p - min(probs)) / (max(probs) - min(probs) + 1e-6) for p in probs]
        ax.scatter(xs, ys, s=sizes, edgecolors="red", facecolors="none", linewidths=0.8)
    ax.set_title("Detected stars (red circles)")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def save_detection_overlay(
    image: np.ndarray,
    detections: Sequence[Detection],
    output_path: Path,
    max_points: int = 1000,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = render_detection_overlay(image, detections, max_points=max_points)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved overlay visualization to {output_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fits_paths = list_fits_files(args.data_dir)
    if not fits_paths:
        raise RuntimeError(f"No FITS files found under {args.data_dir}")
    rng = np.random.default_rng(args.seed)
    print(f"Found {len(fits_paths)} FITS files. Mining training patches...")
    cv_config = None
    if args.candidate_method == "opencv":
        cv_config = CVCandidateConfig(
            blur_ksize=args.cv_blur_ksize,
            thresh_factor=args.cv_thresh_factor,
            thresh_offset=args.cv_thresh_offset,
            min_area=args.cv_min_area,
            max_area=args.cv_max_area,
            min_distance=args.cv_min_distance,
            tile_size=args.cv_tile_size,
            max_per_tile=args.cv_max_per_tile,
        )
    patch_dataset = build_patch_dataset(
        fits_paths,
        patch_size=args.patch_size,
        pos_per_image=args.pos_per_image,
        neg_per_image=args.neg_per_image,
        threshold=args.threshold,
        candidate_method=args.candidate_method,
        cv_params=cv_config,
        rng=rng,
    )
    train_data, val_data = train_val_split(patch_dataset, args.val_fraction, rng)
    print(f"Training patches: {len(train_data.labels)}, Validation patches: {len(val_data.labels)}")
    train_ds = StarPatchDataset(train_data.patches, train_data.labels, augment=True, seed=args.seed)
    val_ds = StarPatchDataset(val_data.patches, val_data.labels, augment=False, seed=args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = StarCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    best_val_loss = math.inf
    metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    weights_path = args.output_dir / "star_detector.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, weights_path)
            print(f"  Saved checkpoint to {weights_path}")

    metrics_path = args.output_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Wrote metrics to {metrics_path}")

    if args.predict is not None:
        image = load_fits(args.predict)
        prob_map, detections = slide_and_predict(
            model,
            image,
            patch_size=args.patch_size,
            stride=args.stride,
            device=device,
            detect_threshold=args.detect_threshold,
        )
        prob_path = args.output_dir / f"{args.predict.stem}_probability_map.fits"
        save_probability_map(prob_map, prob_path)
        catalog_path = args.output_dir / f"{args.predict.stem}_detections.json"
        save_detection_catalog(detections, catalog_path)
        if args.save_overlay:
            overlay_path = args.output_dir / f"{args.predict.stem}_overlay.png"
            save_detection_overlay(image, detections, overlay_path)
        if args.show_overlay:
            fig = render_detection_overlay(image, detections)
            print("Displaying detection overlay (close the window to continue)...")
            plt.show(block=True)
            plt.close(fig)


if __name__ == "__main__":
    main()
