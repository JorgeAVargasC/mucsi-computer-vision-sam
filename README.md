# SAM labeling → YOLO dataset

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Usage

1. **Extract frames** from your videos (one every 15 frames by default):

   ```bash
   python extract_frames.py video_01.mp4 video_02.mp4 --every 15 --out images
   ```

2. **Label with SAM** (click on each object, SPACE to accept, `d`/`a` next/previous, `q` to quit):

   ```bash
   python label_sam.py images --classes mug book tape screwdriver wrench
   ```

   Next time you can omit `--classes` (they are saved in `labels/classes.txt`).
   Use `--model sam2.1_t.pt` for a faster model on CPU.

3. **Build the YOLO dataset** (validation split by video):

   ```bash
   python prepare_dataset.py images labels --val v03
   ```

   This creates `datasets/parts/` and `parts.yaml`, ready for YOLO training.

Run any script with `-h` to see all options.
