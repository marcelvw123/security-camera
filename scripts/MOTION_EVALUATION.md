# Motion evaluation

Use independently recorded continuous footage, not motion-triggered footage, to assess misses. Keep source clips private and outside Git. Review footage and separately note the start and end of each event; include empty and background-motion periods. Review each camera and day/night conditions separately.

Replay the actual application motion detector without sending notifications or modifying the recording:

```sh
.venv/bin/python scripts/evaluate_motion.py /path/to/continuous.mp4 --interval 1 --sensitivity 5 --output /tmp/motion-results.json
```

Compare the timestamps and bounding boxes against the independently reviewed events. Repeat with different intervals and sensitivity levels. Samples during the initial background-learning period are marked `warmup`; they are not proof of a real event. Detector CPU time excludes video decoding and does not predict total Pi load. Replay cannot establish live performance: test CPU, memory, storage, dropped frames and temperature on the Pi separately. Sampled frames and a replayed background model may differ from the live stream, especially during reconnects or recording.

## Score reviewed events

Review the original footage independently and save an annotation JSON file. Time is in seconds from the start of the recording; use half-open intervals `[start, end)`. Each `roi` is a rectangle `[x, y, width, height]` in original video pixels containing the moving object throughout the event. Avoid including unrelated moving areas; if the object travels widely, a large ROI can produce misleading hits. Example:

```json
{"events": [
  {"label": "passing vehicle", "start": 12.2, "end": 15.4, "roi": [450, 100, 500, 200]},
  {"label": "person", "start": 42.0, "end": 48.0, "roi": [100, 150, 200, 300]}
]}
```

Pass `--annotations /path/to/events.json` to include event-level results in the output. A hit means at least one non-warmup motion box intersects the event region during its interval; a miss means at least one eligible frame was sampled but no box intersected. An event with no eligible samples is `unsampled`, not a miss. Report all three counts. This is an approximate spatial match, not proof that the correct object caused the motion: visually check hits, including trees, shadows and other movement. Event recall alone does not measure false alarms; also review labelled empty or background-only periods. Do not use motion-triggered recordings to calculate recall, because their missing events are unknowable.

## Experimental frame-difference replay

`evaluate_difference.py` tests the consecutive-frame detector used by the current Pi deployment. Example using the live settings and saved exclusion areas:

```sh
.venv/bin/python scripts/evaluate_difference.py /path/to/continuous.mp4 --interval 0.5 --min-area 200 --pixel-threshold 15 --exclusions-file motion_exclusions.json --camera manual:701 --output /tmp/difference-results.json
```

Use `manual:801` for camera 801. The example's minimum area corresponds to sensitivity 5; the live minimum area is `200 * 5 / sensitivity`. Vary the minimum area to compare settings without changing the camera. Use independently recorded footage to measure misses; clips captured by motion detection alone cannot establish recall. The live detector may sample at different times or fall back to background subtraction after an error.

Its area threshold is measured after downscaling to 480×270; a different input resolution or viewpoint changes the size of distant people and vehicles in that image. Benchmarks have shown that the filter can miss small, slow, low-contrast or short-lived motion, and may trigger when the camera shakes. The results are experiments, not a proven safety filter. Count event hits and misses against independently reviewed source footage and separately count background-only triggers; box overlap alone does not prove the correct object caused the motion.
