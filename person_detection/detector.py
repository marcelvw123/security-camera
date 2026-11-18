from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    box: np.ndarray
    confidence: float
    source: str


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    confidence: float
    hits: int = 1
    missed: int = 0
    history: list[np.ndarray] = field(default_factory=list)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter))


def nms(detections: list[Detection], threshold: float = 0.45) -> list[Detection]:
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(iou(candidate.box, existing.box) < threshold for existing in kept):
            kept.append(candidate)
    return kept


class TemporalTracker:
    def __init__(self, min_hits: int = 2, max_missed: int = 8):
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: list[Track] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched = set(range(len(detections)))
        for track in self.tracks:
            choices = [(iou(track.box, detections[i].box), i) for i in unmatched]
            score, match = max(choices, default=(0.0, -1))
            if score >= 0.18:
                det = detections[match]
                track.history.append(track.box.copy())
                track.history = track.history[-5:]
                track.box = 0.65 * det.box + 0.35 * track.box
                track.confidence = max(det.confidence, track.confidence * 0.94)
                track.hits += 1
                track.missed = 0
                unmatched.remove(match)
            else:
                track.missed += 1
                track.confidence *= 0.92

        for index in unmatched:
            det = detections[index]
            self.tracks.append(Track(self.next_id, det.box.copy(), det.confidence))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [t for t in self.tracks if t.hits >= self.min_hits and t.missed <= 3]


class NightPersonDetector:
    def __init__(self, model_name: str = "yolo11m.pt", confidence: float = 0.08, imgsz: int = 960):
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.imgsz = imgsz

    @staticmethod
    def enhance(frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        light, a, b = cv2.split(lab)
        light = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(light)
        return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)

    def _predict(self, image: np.ndarray, offset=(0, 0), scale=1.0, source="full") -> list[Detection]:
        result = self.model.predict(
            image,
            classes=[0],
            conf=self.confidence,
            iou=0.55,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        found: list[Detection] = []
        if result.boxes is None:
            return found
        for coordinates, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            coordinates = coordinates / scale
            coordinates[[0, 2]] += offset[0]
            coordinates[[1, 3]] += offset[1]
            found.append(Detection(coordinates.astype(float), float(confidence), source))
        return found

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        enhanced = self.enhance(frame)
        detections = self._predict(frame, source="full")
        detections += self._predict(enhanced, source="enhanced")

        # Overlapping tiles preserve small people that disappear during whole-frame resizing.
        tile_w, tile_h = int(width * 0.62), int(height * 0.62)
        starts_x = sorted(set((0, width - tile_w)))
        starts_y = sorted(set((0, height - tile_h)))
        for y in starts_y:
            for x in starts_x:
                tile = enhanced[y:y + tile_h, x:x + tile_w]
                detections += self._predict(tile, offset=(x, y), source="tile")
        return nms(detections)


def draw_track(frame: np.ndarray, track: Track) -> None:
    x1, y1, x2, y2 = np.rint(track.box).astype(int)
    color = (40, 220, 70)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    label = f"person #{track.track_id} {track.confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 12)), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def process_video(input_path: Path, output_path: Path, json_path: Path, model: str = "yolo11m.pt", progress=None) -> dict:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_output = output_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    detector = NightPersonDetector(model_name=model)
    tracker = TemporalTracker()
    records = []
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        boxes = []
        for track in tracks:
            draw_track(frame, track)
            x1, y1, x2, y2 = np.rint(track.box).astype(int).tolist()
            boxes.append({"track_id": track.track_id, "confidence": round(track.confidence, 4), "box_xyxy": [x1, y1, x2, y2]})
        records.append({"frame": frame_index, "time_seconds": round(frame_index / fps, 4), "people": boxes})
        writer.write(frame)
        frame_index += 1
        if progress and (frame_index == 1 or frame_index % 3 == 0):
            progress(frame_index, frame_count)

    capture.release()
    writer.release()
    if frame_index == 0:
        raise ValueError("The uploaded video contains no readable frames.")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(raw_output), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)], check=True)
        raw_output.unlink(missing_ok=True)
    else:
        raw_output.replace(output_path)

    summary = {
        "source": input_path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frame_index,
        "model": model,
        "coordinate_format": "[left, top, right, bottom] pixels",
        "detections": records,
    }
    json_path.write_text(json.dumps(summary, indent=2))
    if progress:
        progress(frame_index, frame_index)
    return {"frames": frame_index, "expected_frames": frame_count, "unique_tracks": tracker.next_id - 1}
