from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from detector import process_video

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def update_job(job_id: str, **changes) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(changes)


def run_detection(job_id: str, source: Path, annotated: Path, boxes: Path) -> None:
    def report(frame: int, total: int) -> None:
        percent = round(frame * 100 / total) if total else 0
        update_job(job_id, frame=frame, total=total, percent=min(percent, 100))

    try:
        summary = process_video(
            source,
            annotated,
            boxes,
            model=os.getenv("PERSON_MODEL", "yolo11m.pt"),
            progress=report,
        )
        update_job(job_id, status="complete", percent=100, summary=summary)
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc))


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/detect")
def detect_page():
    # Opening /detect directly is not an upload submission. Send the browser
    # back to the form instead of returning a confusing 405 response.
    return redirect(url_for("index"))


@app.post("/detect")
def detect():
    upload = request.files.get("video")
    if not upload or not upload.filename:
        return render_template("index.html", error="Choose an MP4 video first."), 400
    filename = secure_filename(upload.filename)
    if Path(filename).suffix.lower() not in {".mp4", ".mov", ".m4v", ".avi"}:
        return render_template("index.html", error="Please upload an MP4, MOV, M4V, or AVI video."), 400

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir()
    source = job_dir / filename
    annotated = job_dir / "people_detected.mp4"
    boxes = job_dir / "bounding_boxes.json"
    upload.save(source)
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "processing", "frame": 0, "total": 0, "percent": 0}
    threading.Thread(target=run_detection, args=(job_id, source, annotated, boxes), daemon=True).start()
    return redirect(url_for("job_status_page", job_id=job_id))


@app.get("/status/<job_id>")
def job_status_page(job_id: str):
    if job_id not in JOBS:
        return redirect(url_for("index"))
    return render_template("progress.html", job_id=job_id)


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"status": "missing", "error": "Job not found"}), 404
        return jsonify(job)


@app.get("/result/<job_id>")
def result(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or job.get("status") != "complete":
            return redirect(url_for("job_status_page", job_id=job_id))
        summary = job["summary"]
    return render_template("result.html", job_id=job_id, summary=summary)


@app.get("/jobs/<job_id>/<filename>")
def job_file(job_id: str, filename: str):
    return send_from_directory(JOBS_DIR / job_id, filename, as_attachment=filename.endswith(".json"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
