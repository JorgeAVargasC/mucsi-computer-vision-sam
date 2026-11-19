#!/usr/bin/env python3
"""Herramienta de etiquetado con SAM 2.1: unos clics por pieza -> etiquetas YOLO (prácticas 2.2 y 2.3).

Uso:
  python label_sam.py images --classes mug book tape screwdriver wrench
  python label_sam.py images                       # reutiliza labels/classes.txt
  python label_sam.py images --model sam2.1_t.pt   # más rápido en portátiles sin GPU

Ratón:
  clic izquierdo   punto "esto sí" de la pieza que estás marcando
  clic derecho     punto "esto no" (para quitar sombras, mesa, la pieza de al lado...)
  arrastrar        (botón izquierdo) dibujar una caja alrededor de la pieza; se puede
                   combinar con puntos "sí" / "no" para afinar
Teclado:
  0-9              elegir la clase
  ESPACIO / ENTER  aceptar la pieza (se guarda su caja)
  c                cancelar la pieza que estás marcando
  z                borrar la última caja aceptada de esta foto
  d                guardar y pasar a la foto siguiente
  a                guardar y volver a la anterior
  (ventana)        estírala o maximízala: la foto se adapta sin deformarse
  q / ESC          guardar y salir

Cada foto images/v01_00030.jpg genera labels/v01_00030.txt con una línea por pieza:
  clase  centro_x  centro_y  ancho  alto      (todo dividido por el tamaño de la foto)
Si vuelves a abrir una foto ya etiquetada, se cargan sus cajas y puedes seguir.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
from ultralytics import SAM

COLORS = [
    (170, 95, 0),
    (60, 170, 60),
    (40, 130, 230),
    (180, 60, 180),
    (30, 200, 220),
    (200, 200, 60),
    (90, 90, 200),
    (150, 150, 150),
    (60, 60, 60),
    (0, 0, 200),
]
EXTS = (".jpg", ".jpeg", ".png")
WINDOW = "Label"


def read_labels(path, w, h):
    """Lee un .txt YOLO y devuelve [(clase, x1, y1, x2, y2)] en píxeles."""
    boxes = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            p = line.split()
            if len(p) == 5:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                boxes.append(
                    (
                        c,
                        (cx - bw / 2) * w,
                        (cy - bh / 2) * h,
                        (cx + bw / 2) * w,
                        (cy + bh / 2) * h,
                    )
                )
    return boxes


def write_labels(path, boxes, w, h):
    """Guarda las cajas en formato YOLO: clase cx cy ancho alto, normalizado entre 0 y 1."""
    with open(path, "w", encoding="utf-8") as f:
        for c, x1, y1, x2, y2 in boxes:
            f.write(
                f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n"
            )


class Labeler:
    def __init__(self, images, out_dir, classes, model, max_width):
        self.images, self.out_dir, self.classes = images, out_dir, classes
        self.sam = SAM(model)
        self.max_width = max_width
        self.i, self.cls = 0, 0
        self.open(0)

    # ---------- foto actual ----------
    def open(self, i):
        self.i = i
        self.img = cv2.imread(self.images[i])
        self.h, self.w = self.img.shape[:2]
        # se recalculan al dibujar, según el tamaño de la ventana
        self.scale, self.dx, self.dy = 1.0, 0, 0
        self.txt = os.path.join(
            self.out_dir, os.path.splitext(os.path.basename(self.images[i]))[0] + ".txt"
        )
        self.boxes = read_labels(self.txt, self.w, self.h)
        self.cancel()

    def save(self):
        # Una foto que solo has mirado no se guarda: un .txt vacío le diría a YOLO "aquí no hay piezas".
        if self.boxes or os.path.exists(self.txt):
            write_labels(self.txt, self.boxes, self.w, self.h)

    # ---------- pieza que se está marcando ----------
    def cancel(self):
        self.points, self.signs, self.contour, self.box = [], [], None, None
        self.hint_box = None  # caja dibujada arrastrando el ratón
        self.drag = None  # (x0, y0, x1, y1) mientras se arrastra, en la ventana

    def add_point(self, x, y, sign):
        self.points.append([int(x), int(y)])
        self.signs.append(sign)
        self.segment()

    def set_box(self, x1, y1, x2, y2):
        self.hint_box = [
            int(min(x1, x2)),
            int(min(y1, y2)),
            int(max(x1, x2)),
            int(max(y1, y2)),
        ]
        self.segment()

    def segment(self):
        """Pide a SAM la máscara con las pistas que haya: caja, puntos o ambas."""
        # sin caja ni ningún "sí" no hay pieza
        if self.hint_box is None and 1 not in self.signs:
            return
        prompts = {}
        if self.hint_box is not None:
            prompts["bboxes"] = [self.hint_box]
        if self.points:
            # Varios puntos para UNA pieza: points=[[...]], labels=[[...]] (formato de Ultralytics)
            prompts["points"], prompts["labels"] = [self.points], [self.signs]
        try:
            r = self.sam(self.img, verbose=False, **prompts)[0]
        except Exception as e:
            # por si esta versión no admite ambas a la vez
            if "bboxes" in prompts and "points" in prompts:
                print(f"Could not combine box and points ({e}); using the box only")
                r = self.sam(self.img, bboxes=[self.hint_box], verbose=False)[0]
            else:
                raise
        if r.masks is None or len(r.masks.xy) == 0 or len(r.masks.xy[0]) < 3:
            self.contour, self.box = None, None
            print("SAM found nothing there; try another point")
            return
        pol = r.masks.xy[0]  # contorno en píxeles de la foto original
        self.contour = pol.astype(np.int32)
        (x1, y1), (x2, y2) = pol.min(0), pol.max(0)
        self.box = (float(x1), float(y1), float(x2), float(y2))

    def accept(self):
        if self.box is not None:
            self.boxes.append((self.cls, *self.box))
            self.save()
        self.cancel()

    # ---------- dibujo ----------
    def name(self, c):
        return self.classes[c] if c < len(self.classes) else f"class {c}"

    def draw(self, win_w, win_h):
        """Devuelve una imagen del tamaño exacto de la ventana, con la foto escalada sin deformar."""
        vis = self.img.copy()
        for c, x1, y1, x2, y2 in self.boxes:
            col = COLORS[c % len(COLORS)]
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(
                vis,
                self.name(c),
                (int(x1), max(15, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                col,
                2,
            )
        if self.contour is not None:
            layer = vis.copy()
            cv2.fillPoly(layer, [self.contour], COLORS[self.cls % len(COLORS)])
            vis = cv2.addWeighted(layer, 0.4, vis, 0.6, 0)
            x1, y1, x2, y2 = map(int, self.box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)
        if self.hint_box is not None:
            x1, y1, x2, y2 = self.hint_box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 255), 2)
        for (x, y), s in zip(self.points, self.signs):
            cv2.circle(vis, (x, y), 6, (0, 200, 0) if s else (0, 0, 230), -1)
            cv2.circle(vis, (x, y), 6, (255, 255, 255), 1)
        header = (
            f"{self.i + 1}/{len(self.images)} {os.path.basename(self.images[self.i])}  |  "
            f"class {self.cls}: {self.name(self.cls)}  |  objects: {len(self.boxes)}"
        )
        # Misma escala en ancho y alto (la menor de las dos): la foto cabe entera y no se deforma.
        # Lo que sobra se rellena con bandas grises, y el lienzo mide justo lo que la ventana.
        self.scale = min(win_w / self.w, win_h / self.h)
        nw, nh = max(1, round(self.w * self.scale)), max(1, round(self.h * self.scale))
        interp = cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR
        canvas = np.full((win_h, win_w, 3), 40, np.uint8)
        self.dx, self.dy = (win_w - nw) // 2, (win_h - nh) // 2
        canvas[self.dy : self.dy + nh, self.dx : self.dx + nw] = cv2.resize(
            vis, (nw, nh), interpolation=interp
        )
        # caja que se está dibujando (coordenadas de ventana)
        if self.drag is not None:
            ax0, ay0, ax1, ay1 = self.drag
            cv2.rectangle(canvas, (ax0, ay0), (ax1, ay1), (0, 220, 255), 1)
        header += f"  |  {round(self.scale * 100)} %"
        for thickness, color in ((4, (0, 0, 0)), (1, (255, 255, 255))):
            cv2.putText(
                canvas, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness
            )
            cv2.putText(
                canvas,
                "left: yes  right: no  drag: box  SPACE: accept  c: cancel  z: undo  a/d: image  q: quit",
                (10, win_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                thickness,
            )
        return canvas

    def to_image(self, x, y):
        """De coordenadas de la ventana a píxeles de la foto original (quitando bandas y escala)."""
        return (x - self.dx) / self.scale, (y - self.dy) / self.scale

    def inside(self, x, y):
        fx, fy = self.to_image(x, y)
        return 0 <= fx < self.w and 0 <= fy < self.h

    # ---------- eventos ----------
    def mouse(self, ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:  # puede ser un clic o el inicio de una caja
            self.start = (x, y) if self.inside(x, y) else None
        elif ev == cv2.EVENT_MOUSEMOVE and getattr(self, "start", None) is not None:
            x0, y0 = self.start
            if abs(x - x0) > 8 or abs(y - y0) > 8:  # se ha movido: es una caja
                self.drag = (x0, y0, x, y)
        elif ev == cv2.EVENT_LBUTTONUP and getattr(self, "start", None) is not None:
            x0, y0 = self.start
            self.start = None
            fx0, fy0 = self.to_image(x0, y0)  # de la ventana a la foto original
            fx, fy = self.to_image(x, y)
            fx, fy = min(max(fx, 0), self.w - 1), min(max(fy, 0), self.h - 1)
            if self.drag is not None:
                self.drag = None
                self.set_box(fx0, fy0, fx, fy)
            else:
                self.add_point(fx, fy, 1)
        elif ev == cv2.EVENT_RBUTTONDOWN and self.inside(x, y):
            self.add_point(*self.to_image(x, y), 0)

    def key(self, k):
        """Devuelve False para salir."""
        if k in (ord("q"), 27):
            self.save()
            return False
        if ord("0") <= k <= ord("9"):
            self.cls = k - ord("0")
        elif k in (ord(" "), 13, 10):
            self.accept()
        elif k == ord("c"):
            self.cancel()
        elif k == ord("z") and self.boxes:
            self.boxes.pop()
            self.save()
        elif k == ord("d") and self.i < len(self.images) - 1:
            self.save()
            self.open(self.i + 1)
        elif k == ord("a") and self.i > 0:
            self.save()
            self.open(self.i - 1)
        return True


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("images", nargs="?", default="images", help="folder with the images (default: images)")
    p.add_argument("--out", default="labels", help="folder for the .txt files (labels)")
    p.add_argument("--classes", nargs="+", help="class names, in order: the first one is 0")
    p.add_argument("--model", default="sam2.1_b.pt", help="sam2.1_t.pt, _s, _b or _l (auto-downloaded)")
    p.add_argument("--width", type=int, default=1280, help="initial max window width")
    p.add_argument("--height", type=int, default=760, help="initial max window height")
    p.add_argument("--start", type=int, default=1, help="start at image number N")
    args = p.parse_args()

    images = sorted(
        f for f in glob.glob(os.path.join(args.images, "*")) if f.lower().endswith(EXTS)
    )
    if not images:
        sys.exit(f"No .jpg/.png images in {args.images}")
    os.makedirs(args.out, exist_ok=True)

    classes_file = os.path.join(args.out, "classes.txt")
    if args.classes:
        with open(classes_file, "w", encoding="utf-8") as f:
            f.write("\n".join(args.classes) + "\n")
        classes = args.classes
    elif os.path.exists(classes_file):
        classes = [l.strip() for l in open(classes_file, encoding="utf-8") if l.strip()]
    else:
        sys.exit(
            "First time, pass the classes in order: python label_sam.py images --classes mug book wrench"
        )
    print("Classes:", ", ".join(f"{i}={c}" for i, c in enumerate(classes)))
    print(f"{len(images)} images. Loading {args.model} (downloaded the first time)...")

    lb = Labeler(images, args.out, classes, args.model, args.width)
    lb.open(max(0, min(len(images) - 1, args.start - 1)))
    # Ventana redimensionable. Le damos cada vez una imagen del tamaño exacto de la ventana,
    # con la foto escalada por igual en ancho y alto: así OpenCV no tiene que estirarla.
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    s0 = min(1.0, args.width / lb.w, args.height / lb.h)
    size = (max(320, round(lb.w * s0)), max(240, round(lb.h * s0)))
    cv2.resizeWindow(WINDOW, *size)
    cv2.setMouseCallback(WINDOW, lb.mouse)
    while True:
        try:
            _, _, aw, ah = cv2.getWindowImageRect(WINDOW)
            if aw > 0 and ah > 0:
                size = (aw, ah)
        except cv2.error:
            pass  # si no se puede leer, mantenemos el último tamaño
        cv2.imshow(WINDOW, lb.draw(*size))
        k = cv2.waitKey(30) & 0xFF
        if k != 255 and not lb.key(k):
            break
        # ventana cerrada con la X
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            lb.save()
            break
    cv2.destroyAllWindows()
    done = sum(
        1
        for f in images
        if os.path.exists(os.path.join(args.out, os.path.splitext(os.path.basename(f))[0] + ".txt"))
    )
    print(f"Labelled images: {done}/{len(images)} in {args.out}/")


if __name__ == "__main__":
    main()
