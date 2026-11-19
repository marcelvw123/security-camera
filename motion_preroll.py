"""Small, compressed per-camera frame buffer for motion-triggered clips."""
from collections import deque
import math

import cv2
import numpy as np


class MotionPreRoll:
    def __init__(self, seconds=3.0, sample_fps=4.0, max_bytes=5 * 1024 * 1024):
        if seconds < 0 or sample_fps <= 0 or max_bytes < 0:
            raise ValueError('invalid pre-trigger buffer settings')
        self.seconds = seconds
        self.interval = 1.0 / sample_fps
        self.max_bytes = max_bytes
        self.frames = deque()
        self.total_bytes = 0
        self.next_sample = None

    def clear(self):
        self.frames.clear()
        self.total_bytes = 0
        self.next_sample = None

    def add(self, timestamp, frame):
        if not self.seconds or not self.max_bytes:
            return
        if self.next_sample is not None and timestamp < self.next_sample:
            return
        self.next_sample = timestamp + self.interval
        ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        data = encoded.tobytes()
        if len(data) <= self.max_bytes:
            self.frames.append((timestamp, data))
            self.total_bytes += len(data)
        cutoff = timestamp - self.seconds
        while self.frames and (self.frames[0][0] < cutoff or self.total_bytes > self.max_bytes):
            self.total_bytes -= len(self.frames.popleft()[1])

    def write_before(self, writer, trigger_time, fps):
        """Write buffered frames before trigger at the clip FPS; return seconds written.

        Frames are sampled sparsely and repeated to preserve elapsed time. Omit the
        trigger frame itself because the caller writes it in the normal live path.
        """
        frames = [(t, data) for t, data in self.frames if trigger_time-self.seconds <= t < trigger_time]
        if not frames or fps <= 0:
            return 0.0
        count = 0
        for i, (timestamp, data) in enumerate(frames):
            end = frames[i + 1][0] if i + 1 < len(frames) else trigger_time
            decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                continue
            frame_count = max(0, int(round((end - timestamp) * fps)))
            for _ in range(frame_count):
                writer.write(decoded)
            count += frame_count
        return count / fps


class AsyncPreRollWriter:
    """Write pre-roll off the capture thread; bound queued live-frame memory.

    When encoding falls behind, the producer blocks rather than dropping frames.
    The byte limit is per camera; a single oversized frame is allowed when empty.
    """
    def __init__(self, writer, preroll, trigger_time, fps,
                 max_queue_bytes=48 * 1024 * 1024, max_frames=32):
        import threading
        if max_queue_bytes <= 0 or max_frames <= 0:
            raise ValueError('invalid video write queue limits')
        self.writer = writer
        self.path = writer.path
        self.max_queue_bytes = max_queue_bytes
        self.max_frames = max_frames
        self.queue = deque()
        self.queued_bytes = 0
        self.condition = threading.Condition()
        self.closed = False
        self.error = None
        self.buffered_seconds = 0.0
        snapshot = MotionPreRoll(preroll.seconds, 1 / preroll.interval, preroll.max_bytes)
        snapshot.frames = deque(preroll.frames)
        snapshot.total_bytes = preroll.total_bytes
        self.thread = threading.Thread(target=self._run, args=(snapshot, trigger_time, fps), daemon=True)
        self.thread.start()

    def _run(self, preroll, trigger_time, fps):
        try:
            self.buffered_seconds = preroll.write_before(self.writer, trigger_time, fps)
            while True:
                with self.condition:
                    while not self.queue and not self.closed:
                        self.condition.wait()
                    if not self.queue and self.closed:
                        break
                    frame, size = self.queue.popleft()
                    self.queued_bytes -= size
                    self.condition.notify_all()
                self.writer.write(frame)
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.queue.clear()
                self.queued_bytes = 0
                self.condition.notify_all()
        finally:
            try:
                self.writer.release()
            except Exception as exc:
                with self.condition:
                    self.error = self.error or exc
                    self.condition.notify_all()

    def write(self, frame):
        size = int(frame.nbytes)
        with self.condition:
            while (not self.error and not self.closed and self.queue and
                   (len(self.queue) >= self.max_frames or self.queued_bytes + size > self.max_queue_bytes)):
                self.condition.wait()
            if self.error:
                raise RuntimeError('motion clip writer failed') from self.error
            if self.closed:
                raise RuntimeError('motion clip writer is closed')
            self.queue.append((frame, size))
            self.queued_bytes += size
            self.condition.notify_all()

    def release(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join()
        if self.error:
            raise RuntimeError('motion clip writer failed') from self.error
