#!/usr/bin/env python3
"""Monta el dataset de YOLO con las fotos etiquetadas y escribe el YAML (enlace con la práctica 3.1).

La validación se separa POR VÍDEO: todas las fotos de los vídeos indicados en --val van a
validación y el resto a entrenamiento. Así no hay fotos gemelas en los dos lados.

Uso:
  python prepare_dataset.py images labels --val v03
  python prepare_dataset.py images labels --val v03 v04 --out datasets/parts

Resultado:
  datasets/parts/images/train  datasets/parts/labels/train
  datasets/parts/images/val    datasets/parts/labels/val
  parts.yaml
"""
import argparse
import collections
import glob
import os
import shutil
import sys

import yaml

EXTS = (".jpg", ".jpeg", ".png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images")
    p.add_argument("labels")
    p.add_argument("--val", nargs="+", help="video prefixes for validation (e.g. v03). Default: the last one")
    p.add_argument("--out", default=os.path.join("datasets", "parts"))
    p.add_argument("--yaml", default="parts.yaml")
    args = p.parse_args()

    classes_file = os.path.join(args.labels, "classes.txt")
    if not os.path.exists(classes_file):
        sys.exit(f"Missing {classes_file} (created by label_sam.py)")
    classes = [l.strip() for l in open(classes_file, encoding="utf-8") if l.strip()]

    # Solo las fotos que tienen su .txt de etiquetas
    pairs = []
    for f in sorted(glob.glob(os.path.join(args.images, "*"))):
        if f.lower().endswith(EXTS):
            txt = os.path.join(args.labels, os.path.splitext(os.path.basename(f))[0] + ".txt")
            if os.path.exists(txt):
                pairs.append((f, txt))
    if not pairs:
        sys.exit("No labelled images. Label them first with label_sam.py")

    videos = sorted({os.path.basename(f).split("_")[0] for f, _ in pairs})
    val = args.val or [videos[-1]]
    if len(videos) < 2:
        print("Warning: all images come from the same video; validation will not be independent.")

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = os.path.join(args.out, kind, split)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)

    counts = {"train": collections.Counter(), "val": collections.Counter()}
    images_per_split = collections.Counter()
    for f, txt in pairs:
        split = "val" if os.path.basename(f).split("_")[0] in val else "train"
        shutil.copy2(f, os.path.join(args.out, "images", split))
        shutil.copy2(txt, os.path.join(args.out, "labels", split))
        images_per_split[split] += 1
        for line in open(txt, encoding="utf-8"):
            if line.strip():
                counts[split][int(line.split()[0])] += 1

    data = {"path": os.path.abspath(args.out), "train": "images/train", "val": "images/val",
            "names": dict(enumerate(classes))}
    with open(args.yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"Videos: {', '.join(videos)}  ->  validation: {', '.join(val)}")
    for split in ("train", "val"):
        objs = ", ".join(f"{classes[c] if c < len(classes) else c}={n}" for c, n in sorted(counts[split].items()))
        print(f"  {split:5s}: {images_per_split[split]} images · {objs or 'no objects'}")
    missing = [classes[c] for c in range(len(classes)) if counts["val"][c] == 0]
    if missing:
        print(f"Warning: validation has no {', '.join(missing)}. Pick another video for --val.")
    print(f"YAML: {args.yaml}  ·  dataset: {args.out}")


if __name__ == "__main__":
    main()
