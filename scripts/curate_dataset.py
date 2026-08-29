#!/usr/bin/env python
"""Filter and remap a downloaded Roboflow dataset to a verified-clean subset.

Written after discovering that a real Roboflow Universe dataset
("football-w8hds") was an unharmonized merge of multiple original annotation
projects: cross-tabulating class index usage against filename source groups
showed completely different, non-overlapping class sets per source batch
(e.g. one batch used classes {0,2,3,8,9,10,11}, another used {1,9,11}, another
{4,5,6,7} — meaning e.g. class "9" almost certainly means a different real-
world object depending which original source an image came from). Its
data.yaml `names:` list was also corrupted (literal Roboflow marketing
boilerplate text instead of real class names). Training on the raw merge as
one unified label space would mean training on contradictory labels.

This script keeps only images matching a filename prefix you've visually
verified has a coherent, correct class convention (--keep-prefix), and
remaps/filters class indices to just the ones you've confirmed the meaning
of (--class-map), dropping any label line for an unconfirmed class rather
than guessing. Defaults match what was verified for football-w8hds: the
"youtube-*"-prefixed images use class 0=ball, 2=player correctly (confirmed
by drawing their boxes and eyeballing them — see the conversation this was
built in, not something to just trust blindly on a new dataset).

Usage:
    python scripts/curate_dataset.py \
        --source data/datasets/detection \
        --out data/datasets/detection_clean \
        --keep-prefix youtube- \
        --class-map 0:0,2:1 \
        --names ball,player
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPLIT_DIR_NAMES = {"train": "train", "valid": "valid", "test": "test"}


def curate_split(source: Path, out: Path, split_dir: str, keep_prefix: str, class_map: dict[int, int]) -> tuple[int, int]:
    images_dir = source / split_dir / "images"
    labels_dir = source / split_dir / "labels"
    if not images_dir.exists():
        return 0, 0

    out_images = out / split_dir / "images"
    out_labels = out / split_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    kept_images, kept_boxes = 0, 0
    for img_path in images_dir.iterdir():
        if not img_path.name.startswith(keep_prefix):
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        lines = label_path.read_text().splitlines() if label_path.exists() else []

        remapped = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            old_class = int(parts[0])
            if old_class not in class_map:
                continue  # unconfirmed class for this source batch — drop the box, not the image
            remapped.append(" ".join([str(class_map[old_class])] + parts[1:]))

        shutil.copy2(img_path, out_images / img_path.name)
        (out_labels / (img_path.stem + ".txt")).write_text("\n".join(remapped) + ("\n" if remapped else ""))
        kept_images += 1
        kept_boxes += len(remapped)

    return kept_images, kept_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/datasets/detection")
    parser.add_argument("--out", default="data/datasets/detection_clean")
    parser.add_argument("--keep-prefix", default="youtube-")
    parser.add_argument("--class-map", default="0:0,2:1",
                         help="old_class:new_class pairs, comma-separated. Classes not listed are dropped.")
    parser.add_argument("--names", default="ball,player", help="Comma-separated names, in new-class-index order")
    args = parser.parse_args()

    class_map = {}
    for pair in args.class_map.split(","):
        old, new = pair.split(":")
        class_map[int(old)] = int(new)
    names = args.names.split(",")

    source, out = Path(args.source), Path(args.out)
    total_images, total_boxes = 0, 0
    for split_dir in SPLIT_DIR_NAMES.values():
        n_images, n_boxes = curate_split(source, out, split_dir, args.keep_prefix, class_map)
        print(f"{split_dir}: {n_images} images, {n_boxes} boxes")
        total_images += n_images
        total_boxes += n_boxes

    if total_images == 0:
        raise SystemExit(f"No images matched prefix '{args.keep_prefix}' under {source} — nothing written")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(names)}\nnames: {names}\n"
    )
    print(f"\nWrote {data_yaml} ({total_images} images, {total_boxes} boxes total)")
    print(f"Next: python scripts/train_detector.py --data {data_yaml}")


if __name__ == "__main__":
    main()
