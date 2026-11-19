#!/usr/bin/env python3
"""Saca una foto de cada N de uno o varios vídeos (práctica 2.1).

Cada foto se llama vXX_NNNNN.jpg: número de vídeo + número de fotograma.
Así luego se puede separar la validación por vídeo (nada de fotos gemelas).

Uso:
  python extract_frames.py                      # todos los vídeos de la carpeta actual
  python extract_frames.py parts_01.mp4 parts_02.mp4
  python extract_frames.py videos\\*.mp4 --every 10 --out images
"""
import argparse
import glob
import os

import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="*", help="videos (default: all videos in the current folder)")
    p.add_argument("--every", type=int, default=15, help="save one frame every N frames (15)")
    p.add_argument("--out", default="images", help="output folder (images)")
    args = p.parse_args()

    # En Windows la consola no expande *.mp4: lo hacemos aquí. Sin argumentos, todos los vídeos de la carpeta.
    patterns = args.videos or ["*"]
    exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    videos = []
    for pat in patterns:
        found = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        videos += [v for v in found if v.lower().endswith(exts) or not any(c in pat for c in "*?[") and pat == v]
    videos = list(dict.fromkeys(videos))
    if not videos:
        p.exit(1, "No videos found. Pass their names: python extract_frames.py parts_01.mp4 parts_02.mp4\n")

    os.makedirs(args.out, exist_ok=True)
    total = 0
    with open(os.path.join(args.out, "videos.txt"), "w", encoding="utf-8") as index:
        for v, path in enumerate(videos, start=1):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"Could not open {path}")
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            n = saved = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if n % args.every == 0:
                    cv2.imwrite(os.path.join(args.out, f"v{v:02d}_{n:05d}.jpg"), frame)
                    saved += 1
                n += 1
            cap.release()
            print(f"v{v:02d} = {path}: {n} frames at {fps:.0f} fps -> {saved} images")
            index.write(f"v{v:02d}\t{path}\n")
            total += saved
    print(f"Total: {total} images in {args.out}/  (which video is each vXX: {args.out}/videos.txt)")


if __name__ == "__main__":
    main()
