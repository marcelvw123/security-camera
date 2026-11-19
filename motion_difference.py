"""Experimental consecutive-frame motion signal for evaluation only.

This signal has not been validated for night, weather, small objects, or slow
motion. It must not replace the live recorder until those cases are evaluated.
"""
import math
import cv2
import numpy as np


class FrameDifferenceMotion:
    def __init__(self, width=480, height=270, pixel_threshold=25,
                 min_area=500, max_area_ratio=.60, max_gap_seconds=2.0, exclusions=None):
        if (width <= 0 or height <= 0 or not 0 < pixel_threshold < 255 or min_area <= 0
                or not 0 < max_area_ratio <= 1 or max_gap_seconds <= 0):
            raise ValueError('invalid frame difference settings')
        self.exclusions = self.validate_exclusions(exclusions or [])
        self.width, self.height = width, height
        self.pixel_threshold, self.min_area = pixel_threshold, min_area
        self.max_area_ratio, self.max_gap_seconds = max_area_ratio, max_gap_seconds
        self.reset()

    @staticmethod
    def validate_exclusions(exclusions):
        if not isinstance(exclusions, list) or len(exclusions) > 20:
            raise ValueError('exclusions must be a list of up to 20 rectangles')
        result = []
        for rectangle in exclusions:
            if (not isinstance(rectangle, list) or len(rectangle) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in rectangle)):
                raise ValueError('each exclusion must be [x, y, width, height]')
            x, y, w, h = rectangle
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > 1 or y+h > 1:
                raise ValueError('exclusion coordinates must fit within 0 to 1')
            result.append((x, y, w, h))
        return result

    def reset(self):
        self.previous = None
        self.previous_time = None

    def observe(self, frame, timestamp):
        if not math.isfinite(timestamp) or frame is None or frame.ndim != 3 or not frame.size:
            raise ValueError('a non-empty color frame and finite timestamp are required')
        if self.previous_time is not None and timestamp <= self.previous_time:
            raise ValueError('timestamps must increase')
        gray = cv2.cvtColor(cv2.resize(frame, (self.width, self.height)), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        previous, last = self.previous, self.previous_time
        self.previous, self.previous_time = gray, timestamp
        if previous is None or timestamp - last > self.max_gap_seconds:
            return []
        diff = cv2.absdiff(gray, previous)
        mask = cv2.threshold(diff, self.pixel_threshold, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(mask, None, iterations=2)
        for x, y, w, h in self.exclusions:
            x1, y1 = int(x * self.width), int(y * self.height)
            x2, y2 = math.ceil((x + w) * self.width), math.ceil((y + h) * self.height)
            mask[y1:y2, x1:x2] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = frame.shape[:2]
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.width * self.height * self.max_area_ratio:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            x1, y1 = int(x * width / self.width), int(y * height / self.height)
            x2, y2 = int(math.ceil((x+w) * width / self.width)), int(math.ceil((y+h) * height / self.height))
            boxes.append([x1, y1, min(width, x2)-x1, min(height, y2)-y1])
        return boxes
