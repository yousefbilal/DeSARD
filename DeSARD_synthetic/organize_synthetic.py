import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SYNTH_ROOT = ROOT / "dataset" / "synthetic"
DEFAULT_DESTINATION = ROOT / "dataset" / "synthetic"


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def unique_name(height: int, category: str, pose: str, source_name: str) -> str:
    stem = Path(source_name).stem
    if category == "human":
        return f"human_h{height}_{pose}_{stem}.jpg"
    return f"without_human_h{height}_{stem}.jpg"


def discover_heights(synth_root: Path) -> list[tuple[float, Path]]:
    heights = []
    for path in synth_root.glob("height_*"):
        if not path.is_dir():
            continue
        raw_height = path.name.removeprefix("height_")
        try:
            height = float(raw_height)
        except ValueError:
            fail(f"invalid height folder name: {path.name}")
        heights.append((height, path))
    if not heights:
        fail(f"no height_* directories found under {synth_root}")
    return sorted(heights, key=lambda item: item[0])


def collect_source_rows(synth_root: Path) -> list[dict]:
    rows = []
    for height, height_dir in discover_heights(synth_root):
        metadata_height = int(height) if height.is_integer() else height

        for pose in ("sitting", "standing"):
            folder = height_dir / "with human" / pose
            for image_path in sorted(folder.glob("*.jpg")):
                label_path = image_path.with_suffix(".txt")
                if not label_path.exists():
                    fail(f"missing YOLO label for {image_path}")
                rows.append({
                    "source_image": image_path,
                    "source_label": label_path,
                    "image": unique_name(metadata_height, "human", pose, image_path.name),
                    "altitude": metadata_height,
                    "category": "human",
                    "pose": pose,
                })

        folder = height_dir / "without human"
        for image_path in sorted(folder.glob("*.jpg")):
            label_path = image_path.with_suffix(".txt")
            if not label_path.exists():
                fail(f"missing YOLO label for {image_path}")
            rows.append({
                "source_image": image_path,
                "source_label": label_path,
                "image": unique_name(metadata_height, "without human", "", image_path.name),
                "altitude": metadata_height,
                "category": "without human",
                "pose": "",
            })

    if not rows:
        fail(f"no synthetic JPG files found under {synth_root}")

    names = [row["image"] for row in rows]
    duplicates = pd.Series(names)[pd.Series(names).duplicated()].unique().tolist()
    if duplicates:
        fail(f"generated filename collisions detected; examples: {duplicates[:5]}")
    return rows


def link_or_copy(source: Path, destination: Path, copy_files: bool) -> None:
    if copy_files:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def verify_output(rows: list[dict], images_dir: Path, labels_dir: Path,
                  metadata_path: Path) -> None:
    expected_names = {row["image"] for row in rows}
    image_names = {p.name for p in images_dir.glob("*.jpg")}
    label_names = {p.with_suffix(".jpg").name for p in labels_dir.glob("*.txt")}

    if image_names != expected_names:
        fail(
            f"image verification failed: expected={len(expected_names)}, "
            f"found={len(image_names)}, missing={len(expected_names - image_names)}, "
            f"extra={len(image_names - expected_names)}"
        )
    if label_names != expected_names:
        fail(
            f"label verification failed: expected={len(expected_names)}, "
            f"found={len(label_names)}, missing={len(expected_names - label_names)}, "
            f"extra={len(label_names - expected_names)}"
        )

    metadata = pd.read_csv(metadata_path)
    expected_columns = ["image", "altitude", "category"]
    if list(metadata.columns) != expected_columns:
        fail(f"metadata columns are {list(metadata.columns)}, expected {expected_columns}")
    if len(metadata) != len(rows):
        fail(f"metadata has {len(metadata)} rows, expected {len(rows)}")
    if set(metadata["image"]) != expected_names:
        fail("metadata image names do not exactly match the files in images/")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=SYNTH_ROOT)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing images/, labels/, metadata.csv, and stale bbox metadata.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of using space-saving hardlinks when possible.",
    )
    args = parser.parse_args()

    synth_root = args.root.resolve()
    destination_root = args.destination.resolve()
    images_dir = destination_root / "images"
    labels_dir = destination_root / "labels"
    metadata_path = destination_root / "metadata.csv"
    stale_bbox_metadata = destination_root / "metadata_with_bboxes.csv"

    rows = collect_source_rows(synth_root)
    counts = pd.DataFrame(rows).groupby(["altitude", "category"]).size()
    print(f"Source images: {len(rows)}")
    print(counts.to_string())
    print(f"\nSource root: {synth_root}")
    print(f"Destination root: {destination_root}")

    existing = [
        p
        for p in (images_dir, labels_dir, metadata_path, stale_bbox_metadata)
        if p.exists()
    ]
    if existing and not args.force:
        fail(
            "generated output already exists: "
            + ", ".join(str(p) for p in existing)
            + ". Use --force to replace it."
        )

    if args.force:
        if images_dir.exists():
            shutil.rmtree(images_dir)
        if labels_dir.exists():
            shutil.rmtree(labels_dir)
        if metadata_path.exists():
            metadata_path.unlink()
        if stale_bbox_metadata.exists():
            stale_bbox_metadata.unlink()

    destination_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=False)
    labels_dir.mkdir(parents=True, exist_ok=False)

    metadata_rows = []
    for row in tqdm(rows, desc="organizing synthetic data"):
        destination_image = images_dir / row["image"]
        destination_label = labels_dir / Path(row["image"]).with_suffix(".txt").name
        link_or_copy(row["source_image"], destination_image, args.copy)
        link_or_copy(row["source_label"], destination_label, args.copy)

        metadata_rows.append({
            "image": row["image"],
            "altitude": row["altitude"],
            "category": row["category"],
        })

    metadata = pd.DataFrame(
        metadata_rows,
        columns=["image", "altitude", "category"],
    )
    temporary_metadata = metadata_path.with_suffix(".csv.tmp")
    metadata.to_csv(temporary_metadata, index=False)
    temporary_metadata.replace(metadata_path)

    verify_output(rows, images_dir, labels_dir, metadata_path)
    print(
        f"\nDone: {len(rows)} images, {len(rows)} labels, "
        f"and {len(metadata)} metadata rows."
    )


if __name__ == "__main__":
    main()
