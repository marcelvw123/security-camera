from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Set


VEHICLE_OBJECTS = {"car", "truck", "bus", "motorcycle"}
PERSON_OBJECTS = {"person"}
SCENARIO_OBJECTS = VEHICLE_OBJECTS | PERSON_OBJECTS
ScenarioEvent = tuple[datetime, Set[str], int]


def scenario_relevant_objects(detected_objects: Iterable[str]) -> Set[str]:
    return set(detected_objects) & SCENARIO_OBJECTS


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


@dataclass(frozen=True)
class ScenarioRule:
    name: str
    predicate: Callable[[List[ScenarioEvent], datetime], bool]


def has_two_or_more_persons(events: List[ScenarioEvent], _detected_at: datetime) -> bool:
    return any(person_count >= 2 for _event_time, _objects, person_count in events)


def has_vehicle_and_person(events: List[ScenarioEvent], _detected_at: datetime) -> bool:
    combined_objects = combined_event_objects(events)
    return bool(combined_objects & VEHICLE_OBJECTS) and bool(combined_objects & PERSON_OBJECTS)


def has_person_after_hours(events: List[ScenarioEvent], detected_at: datetime) -> bool:
    if not is_after_hours(detected_at):
        return False

    return any(person_count >= 1 for _event_time, _objects, person_count in events)


def is_after_hours(detected_at: datetime) -> bool:
    current_time = detected_at.time()
    return current_time >= time(23, 0) or current_time < time(5, 0)


def combined_event_objects(events: List[ScenarioEvent]) -> Set[str]:
    combined_objects: Set[str] = set()
    for _event_time, objects, _person_count in events:
        combined_objects.update(objects)
    return combined_objects


DEFAULT_SCENARIO_RULES = (
    ScenarioRule("person_after_23h00", has_person_after_hours),
    ScenarioRule("two_or_more_persons", has_two_or_more_persons),
    ScenarioRule("vehicle_person", has_vehicle_and_person),
)


class ScenarioDetector:
    def __init__(
        self,
        window_seconds: int = 20,
        cooldown_seconds: int = 0,
        rules: Iterable[ScenarioRule] = DEFAULT_SCENARIO_RULES,
    ):
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.rules = tuple(rules)
        self._events_by_camera: Dict[str, List[ScenarioEvent]] = {}
        self._last_match_by_camera: Dict[tuple[str, str], datetime] = {}

    def record_detection(
        self,
        camera_name: str,
        detected_objects: Iterable[str],
        detected_at: Optional[datetime] = None,
    ) -> Optional[ScenarioMatch]:
        detected_object_list = list(detected_objects)
        relevant_objects = scenario_relevant_objects(detected_object_list)
        if not relevant_objects:
            return None

        detected_at = detected_at or datetime.now()
        window_start = detected_at - timedelta(seconds=self.window_seconds)
        events = self._events_by_camera.setdefault(camera_name, [])
        events.append((detected_at, set(relevant_objects), detected_object_list.count("person")))
        events[:] = [event for event in events if event[0] >= window_start]

        for rule in self.rules:
            if not rule.predicate(events, detected_at):
                continue

            match_key = (camera_name, rule.name)
            last_match = self._last_match_by_camera.get(match_key)
            if last_match and detected_at - last_match < timedelta(seconds=self.cooldown_seconds):
                continue

            self._last_match_by_camera[match_key] = detected_at
            return ScenarioMatch(
                name=f"{rule.name}_within_{self.window_seconds}s",
                camera_name=camera_name,
                objects=combined_event_objects(events),
                matched_at=detected_at,
                window_seconds=self.window_seconds,
            )

        return None
