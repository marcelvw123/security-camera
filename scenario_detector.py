from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set


VEHICLE_OBJECTS = {"car", "truck", "bus", "motorcycle"}
PERSON_OBJECTS = {"person"}
SCENARIO_OBJECTS = VEHICLE_OBJECTS | PERSON_OBJECTS


def scenario_relevant_objects(detected_objects: Set[str]) -> Set[str]:
    return detected_objects & SCENARIO_OBJECTS


@dataclass(frozen=True)
class ScenarioMatch:
    name: str
    camera_name: str
    objects: Set[str]
    matched_at: datetime
    window_seconds: int

    @property
    def label(self) -> str:
        return self.name.replace("_", "-")


class ScenarioDetector:
    def __init__(self, window_seconds: int = 20, cooldown_seconds: int = 0):
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self._events_by_camera: Dict[str, List[tuple[datetime, Set[str]]]] = {}
        self._last_match_by_camera: Dict[str, datetime] = {}

    def record_detection(
        self,
        camera_name: str,
        detected_objects: Set[str],
        detected_at: Optional[datetime] = None,
    ) -> Optional[ScenarioMatch]:
        relevant_objects = scenario_relevant_objects(detected_objects)
        if not relevant_objects:
            return None

        detected_at = detected_at or datetime.now()
        window_start = detected_at - timedelta(seconds=self.window_seconds)
        events = self._events_by_camera.setdefault(camera_name, [])
        events.append((detected_at, set(relevant_objects)))
        events[:] = [event for event in events if event[0] >= window_start]

        if not self._has_vehicle_and_person(events):
            return None

        last_match = self._last_match_by_camera.get(camera_name)
        if last_match and detected_at - last_match < timedelta(seconds=self.cooldown_seconds):
            return None

        self._last_match_by_camera[camera_name] = detected_at
        return ScenarioMatch(
            name=f"vehicle_person_within_{self.window_seconds}s",
            camera_name=camera_name,
            objects=self._combined_objects(events),
            matched_at=detected_at,
            window_seconds=self.window_seconds,
        )

    @staticmethod
    def _has_vehicle_and_person(events: List[tuple[datetime, Set[str]]]) -> bool:
        combined_objects = ScenarioDetector._combined_objects(events)
        return bool(combined_objects & VEHICLE_OBJECTS) and bool(combined_objects & PERSON_OBJECTS)

    @staticmethod
    def _combined_objects(events: List[tuple[datetime, Set[str]]]) -> Set[str]:
        combined_objects: Set[str] = set()
        for _event_time, objects in events:
            combined_objects.update(objects)
        return combined_objects
