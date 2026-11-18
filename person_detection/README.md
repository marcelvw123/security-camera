# Night Person Detector

A local browser app that accepts CCTV video and produces:

- an annotated MP4 containing tracked person bounding boxes;
- a JSON file containing frame number, timestamp, track ID, confidence, and pixel coordinates.

The detector combines normal and contrast-enhanced inference with overlapping high-resolution tiles. A temporal tracker keeps weak detections stable across nearby frames.

## Install

Python 3.10 or newer and FFmpeg are recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the selected YOLO weights. By default the app uses `yolo11m.pt`. A larger model may improve difficult footage:

```bash
PERSON_MODEL=yolo11x.pt python app.py
```

Otherwise start normally:

```bash
python app.py
```

Open `http://127.0.0.1:7860`, select a video, and press **Detect people**.

## Output coordinates

`bounding_boxes.json` stores boxes as `[left, top, right, bottom]` in source-video pixels. `track_id` aims to retain the same identity between frames; it is not facial recognition.

## Important tuning note

The supplied example has detection rectangles already baked into the pixels. Results will be better when the original clean camera stream is used, before motion boxes or text overlays are drawn.
