#!/usr/bin/env python3
"""
Extract masked keyframes from visible/infrared video sequences, cluster them,
and save cluster centers as common feature maps.

Expected dataset layout:

dataset_root/
  sequence_001/
    vi/
      000001.jpg
      ...
    ir/
      000001.png
      ...
    init.txt        # one bbox per line: x y w h

mask_root/
  sequence_001/
    000001.png
    ...

The visible stream is loaded as RGB. The infrared stream is loaded as one
channel. Masks are loaded as grayscale and multiplied into the frame before
keyframe extraction and clustering.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class KeyFrame:
    sequence: str
    frame_index: int
    image_path: str
    mask_path: str
    feature: np.ndarray


def natural_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.stem.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def parse_size(value: str) -> Tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[x,]\s*(\d+)\s*", value.lower())
    if not match:
        raise argparse.ArgumentTypeError("size must look like 224x224 or 224,224")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size values must be positive")
    return width, height


def list_images(directory: Path, recursive_fallback: bool = False) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        return []

    images = [
        item
        for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images and recursive_fallback:
        images = [
            item
            for item in directory.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return sorted(images, key=natural_key)


def read_init_boxes(init_path: Path) -> List[Tuple[float, float, float, float]]:
    if not init_path.exists():
        return []

    boxes: List[Tuple[float, float, float, float]] = []
    for line_number, line in enumerate(init_path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = re.split(r"[\s,]+", stripped)
        if len(parts) != 4:
            raise ValueError(f"{init_path}:{line_number} must contain x y w h")
        boxes.append(tuple(float(part) for part in parts))  # type: ignore[arg-type]
    return boxes


def sequence_dirs(dataset_root: Path) -> List[Path]:
    sequences = []
    for item in sorted(dataset_root.iterdir(), key=natural_key):
        if item.is_dir() and (item / "vi").is_dir() and (item / "ir").is_dir():
            sequences.append(item)
    return sequences


def build_mask_lookup(mask_paths: Sequence[Path]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for path in mask_paths:
        lookup.setdefault(path.stem.lower(), path)
    return lookup


def pick_mask(
    mask_paths: Sequence[Path],
    mask_lookup: Dict[str, Path],
    frame_path: Path,
    frame_index: int,
) -> Optional[Path]:
    by_name = mask_lookup.get(frame_path.stem.lower())
    if by_name is not None:
        return by_name
    if frame_index < len(mask_paths):
        return mask_paths[frame_index]
    return None


def resize_array(array: np.ndarray, size: Tuple[int, int], is_rgb: bool) -> np.ndarray:
    clipped = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if is_rgb:
        image = Image.fromarray(clipped, mode="RGB")
    else:
        image = Image.fromarray(clipped[:, :, 0], mode="L")
    image = image.resize(size, Image.Resampling.BILINEAR)
    resized = np.asarray(image, dtype=np.float32) / 255.0
    if not is_rgb:
        resized = resized[:, :, None]
    return resized


def load_masked_feature(
    image_path: Path,
    mask_path: Path,
    is_rgb: bool,
    feature_size: Tuple[int, int],
    binary_mask: bool,
) -> np.ndarray:
    image_mode = "RGB" if is_rgb else "L"
    with Image.open(image_path) as image_handle:
        image = image_handle.convert(image_mode)
        image_size = image.size
        image_array = np.asarray(image, dtype=np.float32) / 255.0

    if not is_rgb:
        image_array = image_array[:, :, None]

    with Image.open(mask_path) as mask_handle:
        mask = mask_handle.convert("L").resize(image_size, Image.Resampling.NEAREST)
        mask_array = np.asarray(mask, dtype=np.float32) / 255.0

    if binary_mask:
        mask_array = (mask_array > 0.5).astype(np.float32)

    masked = image_array * mask_array[:, :, None]
    return resize_array(masked, feature_size, is_rgb=is_rgb)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(diff * diff))


def save_feature_image(feature: np.ndarray, path: Path, is_rgb: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_array = np.clip(feature * 255.0, 0, 255).astype(np.uint8)
    if is_rgb:
        image = Image.fromarray(image_array, mode="RGB")
    else:
        image = Image.fromarray(image_array[:, :, 0], mode="L")
    image.save(path)


def extract_stream_keyframes(
    dataset_sequences: Sequence[Path],
    mask_root: Path,
    stream_name: str,
    is_rgb: bool,
    feature_size: Tuple[int, int],
    threshold: float,
    binary_mask: bool,
    save_dir: Optional[Path],
) -> Tuple[List[KeyFrame], List[str]]:
    keyframes: List[KeyFrame] = []
    warnings: List[str] = []

    for sequence_path in dataset_sequences:
        sequence_name = sequence_path.name
        frame_paths = list_images(sequence_path / stream_name)
        mask_paths = list_images(mask_root / sequence_name, recursive_fallback=True)
        mask_lookup = build_mask_lookup(mask_paths)
        boxes = read_init_boxes(sequence_path / "init.txt")

        if not frame_paths:
            warnings.append(f"{sequence_name}/{stream_name}: no frames found")
            continue
        if not mask_paths:
            warnings.append(f"{sequence_name}/{stream_name}: no masks found")
            continue
        if boxes and len(boxes) != len(frame_paths):
            warnings.append(
                f"{sequence_name}/{stream_name}: init.txt has {len(boxes)} boxes, "
                f"but {len(frame_paths)} frames were found"
            )

        last_key_feature: Optional[np.ndarray] = None
        sequence_key_count = 0

        for frame_index, frame_path in enumerate(frame_paths):
            mask_path = pick_mask(mask_paths, mask_lookup, frame_path, frame_index)
            if mask_path is None:
                warnings.append(
                    f"{sequence_name}/{stream_name}: missing mask for frame {frame_path.name}"
                )
                continue

            feature = load_masked_feature(
                frame_path,
                mask_path,
                is_rgb=is_rgb,
                feature_size=feature_size,
                binary_mask=binary_mask,
            )

            should_keep = last_key_feature is None or mse(last_key_feature, feature) > threshold
            if should_keep:
                keyframe = KeyFrame(
                    sequence=sequence_name,
                    frame_index=frame_index,
                    image_path=str(frame_path),
                    mask_path=str(mask_path),
                    feature=feature,
                )
                keyframes.append(keyframe)
                last_key_feature = feature
                sequence_key_count += 1

                if save_dir is not None:
                    filename = f"{sequence_name}__{frame_index:06d}.png"
                    save_feature_image(feature, save_dir / filename, is_rgb=is_rgb)

        if sequence_key_count == 0:
            warnings.append(f"{sequence_name}/{stream_name}: no keyframes kept")

    return keyframes, warnings


def assign_clusters(features: np.ndarray, centers: np.ndarray, batch_size: int) -> np.ndarray:
    labels = np.empty(features.shape[0], dtype=np.int64)
    center_norms = np.sum(centers * centers, axis=1)

    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size]
        batch_norms = np.sum(batch * batch, axis=1, keepdims=True)
        distances = batch_norms + center_norms[None, :] - 2.0 * batch @ centers.T
        labels[start : start + batch.shape[0]] = np.argmin(distances, axis=1)

    return labels


def kmeans(
    samples: Sequence[np.ndarray],
    k: int,
    max_iter: int,
    seed: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("cannot cluster an empty keyframe collection")
    if k <= 0:
        raise ValueError("cluster count must be positive")

    feature_shape = samples[0].shape
    features = np.stack([sample.reshape(-1) for sample in samples]).astype(np.float32)
    sample_count = features.shape[0]
    k = min(k, sample_count)

    rng = np.random.default_rng(seed)
    initial_indices = rng.choice(sample_count, size=k, replace=False)
    centers = features[initial_indices].copy()
    labels = np.full(sample_count, -1, dtype=np.int64)

    for _ in range(max_iter):
        new_labels = assign_clusters(features, centers, batch_size=batch_size)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        nearest_distances = None
        for cluster_index in range(k):
            members = features[labels == cluster_index]
            if members.size:
                centers[cluster_index] = np.mean(members, axis=0)
            else:
                if nearest_distances is None:
                    nearest_distances = cluster_distances(features, centers, batch_size)
                replacement = int(np.argmax(nearest_distances))
                centers[cluster_index] = features[replacement]

    return centers.reshape((k,) + feature_shape), labels


def cluster_distances(features: np.ndarray, centers: np.ndarray, batch_size: int) -> np.ndarray:
    distances_out = np.empty(features.shape[0], dtype=np.float32)
    center_norms = np.sum(centers * centers, axis=1)

    for start in range(0, features.shape[0], batch_size):
        batch = features[start : start + batch_size]
        batch_norms = np.sum(batch * batch, axis=1, keepdims=True)
        distances = batch_norms + center_norms[None, :] - 2.0 * batch @ centers.T
        distances_out[start : start + batch.shape[0]] = np.min(distances, axis=1)

    return distances_out


def save_cluster_outputs(
    centers: np.ndarray,
    labels: np.ndarray,
    keyframes: Sequence[KeyFrame],
    output_dir: Path,
    is_rgb: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for cluster_index, center in enumerate(centers):
        save_feature_image(center, output_dir / f"cluster_{cluster_index:03d}.png", is_rgb=is_rgb)

    cluster_items = []
    for cluster_index in range(centers.shape[0]):
        members = [
            {
                "sequence": keyframes[item_index].sequence,
                "frame_index": keyframes[item_index].frame_index,
                "image_path": keyframes[item_index].image_path,
                "mask_path": keyframes[item_index].mask_path,
            }
            for item_index, label in enumerate(labels)
            if int(label) == cluster_index
        ]
        cluster_items.append(
            {
                "cluster": cluster_index,
                "image": str(output_dir / f"cluster_{cluster_index:03d}.png"),
                "member_count": len(members),
                "members": members,
            }
        )

    (output_dir / "clusters.json").write_text(
        json.dumps(cluster_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    sequence_names: Sequence[str],
    vi_keyframes: Sequence[KeyFrame],
    ir_keyframes: Sequence[KeyFrame],
    warnings: Sequence[str],
) -> None:
    metadata = {
        "dataset_root": str(args.dataset_root),
        "mask_root": str(args.mask_root),
        "threshold": args.threshold,
        "feature_size": list(args.feature_size),
        "sequence_count": len(sequence_names),
        "sequences": list(sequence_names),
        "vi_keyframe_count": len(vi_keyframes),
        "ir_keyframe_count": len(ir_keyframes),
        "binary_mask": args.binary_mask,
        "warnings": list(warnings),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply masks to VI/IR sequences, extract MSE keyframes, cluster each "
            "modality, and save cluster centers as common feature maps."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="dataset root directory")
    parser.add_argument("--mask-root", type=Path, required=True, help="mask root directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for outputs")
    parser.add_argument(
        "--threshold",
        "-t",
        type=positive_float,
        default=0.5,
        help="normalized MSE threshold for keyframe extraction; default: 0.5",
    )
    parser.add_argument(
        "--feature-size",
        type=parse_size,
        default=(224, 224),
        help="common feature image size before MSE/KMeans, e.g. 224x224; default: 224x224",
    )
    parser.add_argument("--max-iter", type=int, default=100, help="KMeans max iterations")
    parser.add_argument("--seed", type=int, default=0, help="random seed for KMeans init")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="batch size for KMeans distance calculation",
    )
    parser.add_argument(
        "--binary-mask",
        action="store_true",
        help="threshold masks at 0.5 before applying them",
    )
    parser.add_argument(
        "--save-keyframes",
        action="store_true",
        help="also save masked keyframes used for clustering",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not args.dataset_root.is_dir():
        parser.error(f"--dataset-root does not exist or is not a directory: {args.dataset_root}")
    if not args.mask_root.is_dir():
        parser.error(f"--mask-root does not exist or is not a directory: {args.mask_root}")

    sequences = sequence_dirs(args.dataset_root)
    if not sequences:
        parser.error("no sequence folders with vi/ and ir/ were found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_names = [sequence.name for sequence in sequences]
    cluster_count = len(sequences)

    vi_keyframe_dir = args.output_dir / "vi_keyframes" if args.save_keyframes else None
    ir_keyframe_dir = args.output_dir / "ir_keyframes" if args.save_keyframes else None

    vi_keyframes, vi_warnings = extract_stream_keyframes(
        sequences,
        args.mask_root,
        stream_name="vi",
        is_rgb=True,
        feature_size=args.feature_size,
        threshold=args.threshold,
        binary_mask=args.binary_mask,
        save_dir=vi_keyframe_dir,
    )
    ir_keyframes, ir_warnings = extract_stream_keyframes(
        sequences,
        args.mask_root,
        stream_name="ir",
        is_rgb=False,
        feature_size=args.feature_size,
        threshold=args.threshold,
        binary_mask=args.binary_mask,
        save_dir=ir_keyframe_dir,
    )

    warnings = vi_warnings + ir_warnings

    if not vi_keyframes:
        parser.error("no VI keyframes were extracted; check vi frames, masks, and paths")
    if not ir_keyframes:
        parser.error("no IR keyframes were extracted; check ir frames, masks, and paths")

    if len(vi_keyframes) < cluster_count:
        warnings.append(
            f"vi: requested {cluster_count} clusters, but only {len(vi_keyframes)} keyframes exist; "
            f"using {len(vi_keyframes)} clusters"
        )
    if len(ir_keyframes) < cluster_count:
        warnings.append(
            f"ir: requested {cluster_count} clusters, but only {len(ir_keyframes)} keyframes exist; "
            f"using {len(ir_keyframes)} clusters"
        )

    vi_centers, vi_labels = kmeans(
        [item.feature for item in vi_keyframes],
        k=cluster_count,
        max_iter=args.max_iter,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    ir_centers, ir_labels = kmeans(
        [item.feature for item in ir_keyframes],
        k=cluster_count,
        max_iter=args.max_iter,
        seed=args.seed + 1,
        batch_size=args.batch_size,
    )

    save_cluster_outputs(
        vi_centers,
        vi_labels,
        vi_keyframes,
        args.output_dir / "vi_common_features",
        is_rgb=True,
    )
    save_cluster_outputs(
        ir_centers,
        ir_labels,
        ir_keyframes,
        args.output_dir / "ir_common_features",
        is_rgb=False,
    )
    write_run_metadata(
        args.output_dir,
        args,
        sequence_names,
        vi_keyframes,
        ir_keyframes,
        warnings,
    )

    print(f"Sequences: {len(sequences)}")
    print(f"VI keyframes: {len(vi_keyframes)}")
    print(f"IR keyframes: {len(ir_keyframes)}")
    print(f"VI common features: {args.output_dir / 'vi_common_features'}")
    print(f"IR common features: {args.output_dir / 'ir_common_features'}")
    if warnings:
        print(f"Warnings: {len(warnings)}; see run_metadata.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
