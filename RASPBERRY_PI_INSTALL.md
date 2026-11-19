# Raspberry Pi motion testing

This branch focuses on local motion recording and evaluation. It does not send WhatsApp notifications or upload clips from the Python application. The API branch retains clip upload endpoints for future use, but this Python workflow does not call them.

Use a 64-bit Raspberry Pi OS and the existing Python virtual environment. Install `requirements-headless.txt` if creating a new environment. Configure camera connection settings from `.env.example` in a private `.env` file. Start the web app with `.venv/bin/python webapp.py` and select cameras and motion sensitivity in the web interface. Check whether another instance is already running before starting it. Do not overwrite a running installation without a backup and a tested rollback.

Clips are stored in `~/SecurityCamera/clips`. At startup and after reconnect, the detector learns the background for `MOTION_WARMUP_SECONDS` before creating clips. Motion detection runs while recording. When `MOTION_MAX_CLIPS` is reached or free disk space drops below `MOTION_MIN_FREE_MB`, new recording stops; existing clips are not deleted. The limits are checked when starting a clip, so they are not strict limits during concurrent or ongoing recordings. Review disk space regularly. Recordings include up to three seconds of compressed pre-trigger footage when available. The pre-trigger frames are sampled at 4 fps and repeated to match the clip frame rate. The buffer is capped at 5 MiB per camera. Clip encoding runs on a separate thread; its live-frame queue is limited to 48 MiB or 32 frames per camera, then applies backpressure if encoding cannot keep up.

Set `DETECTION_INTERVAL_SECONDS` to control sampling. Faster sampling increases load and must be benchmarked on the Pi. An existing `.env` value overrides the default shown in `.env.example`.

See [motion evaluation](scripts/MOTION_EVALUATION.md) for replay and event scoring. Motion-triggered clips alone cannot measure missed events; use independently recorded continuous footage for that. The legacy installer and service scripts have not been validated for this motion-only branch.

## Camera motion settings

Each camera can use **Frame difference (test)** or **Original background**. Frame difference is experimental and may miss small, slow, brief, or low-contrast movement; it has not been validated at night. A processing error falls back to the original background method. The sensitivity control adjusts the minimum detected motion area.

For each running camera, open **Draw excluded areas** on its preview. Drag rectangles over regions to ignore, then choose **Save areas**. Undo and Clear all let you revise the selection. The shaded regions suppress all motion there, including people and vehicles. Changes apply to the running camera immediately, and the Pi stores them per camera for future starts. No areas are excluded by default.

### Experimental object clips

The optional object worker uses one shared inference thread for all cameras and reuses their existing frames. Install `ultralytics` in the Pi environment, place the YOLO11n weights at `models/yolo11n.pt`, and set `OBJECT_DETECTION_ENABLED=true` in `.env` before starting the web app. The preview reports whether object detection is ready, unavailable, or recording. Motion recording continues independently if object inference fails.

Person detections must be seen in two successive analyzed frames. Vehicle detections also need to move between samples, to reduce triggers from parked cars. An event creates a separate clip in `~/SecurityCamera/object_clips`, with about three seconds of pre-roll and 20 seconds after confirmation. It may overlap the corresponding motion clip. The clip is finalized when recording finishes or the camera stops. A continuing event does not repeatedly start clips; a quiet interval rearms it. Files are not automatically deleted, and the disk reserve applies to both recording types.

This is an experimental filter, not a reliable security alarm. Inference is sampled and asynchronous; rapid, small, obscured or low-light objects may be missed. In particular, the tested vehicles behind the gate on cameras 701 and 801 were missed. The original motion clip is retained for review. WhatsApp is not triggered by object clips.
