"""Compare an experimental motion signal without changing recording decisions."""
import logging
from motion_difference import FrameDifferenceMotion


class MotionShadowComparison:
    def __init__(self, camera, pixel_threshold=15, min_area=200, report_interval=30):
        self.camera = camera
        self.filter = FrameDifferenceMotion(pixel_threshold=pixel_threshold, min_area=min_area)
        self.report_interval = report_interval
        self.reset()

    def reset(self):
        self.filter.reset()
        self.start = None
        self.failed = False
        self.total = self.raw = self.candidate = self.raw_only = self.candidate_only = 0

    def observe(self, frame, timestamp, raw_motion, eligible=True):
        if self.failed:
            return
        try:
            primed = self.filter.previous_time is not None and 0 < timestamp-self.filter.previous_time <= self.filter.max_gap_seconds
            filtered = bool(self.filter.observe(frame, timestamp))
            if not eligible or not primed:
                return
            if self.start is None:
                self.start = timestamp
            self.total += 1
            self.raw += int(bool(raw_motion))
            self.candidate += int(filtered)
            self.raw_only += int(bool(raw_motion) and not filtered)
            self.candidate_only += int(filtered and not raw_motion)
            if timestamp - self.start >= self.report_interval:
                logging.info('MOTION_SHADOW camera=%s samples=%s raw=%s candidate=%s raw_only=%s candidate_only=%s',
                             self.camera, self.total, self.raw, self.candidate, self.raw_only, self.candidate_only)
                self.start = timestamp
                self.total = self.raw = self.candidate = self.raw_only = self.candidate_only = 0
        except Exception:
            # The comparison must never stop, suppress, or trigger recording.
            logging.exception('Motion shadow comparison disabled for camera=%s', self.camera)
            self.failed = True
