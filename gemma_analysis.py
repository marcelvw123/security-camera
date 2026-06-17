import base64
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import cv2

from app_config import load_dotenv


DEFAULT_FRAME_COUNT = 6
DEFAULT_JPEG_QUALITY = 75
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_MAX_TOKENS = 1200
DEFAULT_MODEL = "gemma-4"
PROMPT = """
You are analyzing a security camera clip from a home or business.
Describe the visible scene and decide whether any people appear to be trying to break in.
Consider suspicious signs such as forced entry, checking doors or windows, hiding, tools,
climbing fences, damaging property, or coordinated intrusion behavior.
Return only JSON with these fields:
scene_description: string
break_in_likely: boolean
confidence: number from 0 to 1
reasoning: string
""".strip()


@dataclass(frozen=True)
class GemmaAnalysisConfig:
    api_url: str
    model: str
    frame_count: int
    jpeg_quality: int
    timeout_seconds: int
    max_tokens: int

    @classmethod
    def from_environment(cls) -> Optional["GemmaAnalysisConfig"]:
        load_dotenv()
        api_url = os.getenv("GEMMA_API_URL", "").strip()
        base_url = os.getenv("GEMMA_BASE_URL", "").strip().rstrip("/")

        if not api_url and base_url:
            api_url = urljoin(f"{base_url}/", "chat/completions" if base_url.endswith("/v1") else "v1/chat/completions")

        if not api_url:
            return None

        return cls(
            api_url=api_url,
            model=os.getenv("GEMMA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            frame_count=parse_int_env("GEMMA_FRAME_COUNT", DEFAULT_FRAME_COUNT),
            jpeg_quality=parse_int_env("GEMMA_JPEG_QUALITY", DEFAULT_JPEG_QUALITY),
            timeout_seconds=parse_int_env("GEMMA_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            max_tokens=parse_int_env("GEMMA_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        )


def parse_int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default

    try:
        return int(value)
    except ValueError:
        logging.warning("Invalid %s value %r; using %s", name, value, default)
        return default


def analyze_clip_with_gemma(clip_path: Path, trigger_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    config = GemmaAnalysisConfig.from_environment()
    if config is None:
        logging.info("Gemma clip analysis skipped because GEMMA_API_URL or GEMMA_BASE_URL is not configured")
        return None

    frame_data_urls = sample_video_frames(
        clip_path,
        frame_count=config.frame_count,
        jpeg_quality=config.jpeg_quality,
    )
    if not frame_data_urls:
        logging.warning("Gemma clip analysis skipped because no frames could be sampled from %s", clip_path)
        return None

    payload = build_gemma_payload(config, frame_data_urls)
    try:
        log_gemma(trigger_id, "STEP_7_GEMMA_REQUEST_SENT", "clip=%s", clip_path)
        response = post_json(config.api_url, payload, timeout_seconds=config.timeout_seconds)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        logging.warning("Gemma clip analysis request failed for %s: %s", clip_path, exc)
        return None
    except json.JSONDecodeError as exc:
        logging.warning("Gemma clip analysis returned invalid JSON response for %s: %s", clip_path, exc)
        return None

    log_gemma(trigger_id, "STEP_8_GEMMA_RAW_RESPONSE", "clip=%s response=%s", clip_path, json.dumps(response, ensure_ascii=False))
    content = extract_message_content(response)
    if not content:
        logging.warning("Gemma clip analysis returned no message content for %s", clip_path)
        return None

    log_gemma(trigger_id, "STEP_8_GEMMA_RESPONSE_CONTENT", "clip=%s content=%s", clip_path, content)
    analysis = parse_analysis_content(content)
    analysis["finish_reason"] = extract_finish_reason(response)
    log_gemma(
        trigger_id,
        "STEP_8_GEMMA_PARSED_ANALYSIS",
        "clip=%s scene=%r break_in_likely=%r confidence=%r reasoning=%r finish_reason=%r",
        clip_path,
        analysis.get("scene_description"),
        analysis.get("break_in_likely"),
        analysis.get("confidence"),
        analysis.get("reasoning"),
        analysis.get("finish_reason"),
    )
    print_gemma_analysis(clip_path, analysis)
    return analysis


def log_gemma(trigger_id: Optional[str], step: str, message: str, *args) -> None:
    if args:
        message = message % args

    if trigger_id:
        logging.info("%s %s %s", trigger_id, step, message)
    else:
        logging.info("%s %s", step, message)


def sample_video_frames(clip_path: Path, frame_count: int, jpeg_quality: int) -> List[str]:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return []

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            return sample_sequential_frames(cap, frame_count, jpeg_quality)

        indexes = evenly_spaced_indexes(total_frames, frame_count)
        frame_data_urls = []
        for frame_index in indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue

            data_url = encode_frame_as_data_url(frame, jpeg_quality)
            if data_url:
                frame_data_urls.append(data_url)

        return frame_data_urls
    finally:
        cap.release()


def sample_sequential_frames(cap, frame_count: int, jpeg_quality: int) -> List[str]:
    frame_data_urls = []
    while len(frame_data_urls) < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        data_url = encode_frame_as_data_url(frame, jpeg_quality)
        if data_url:
            frame_data_urls.append(data_url)

    return frame_data_urls


def evenly_spaced_indexes(total_frames: int, frame_count: int) -> List[int]:
    frame_count = max(1, min(frame_count, total_frames))
    if frame_count == 1:
        return [total_frames // 2]

    last_index = total_frames - 1
    return sorted({round(index * last_index / (frame_count - 1)) for index in range(frame_count)})


def encode_frame_as_data_url(frame, jpeg_quality: int) -> Optional[str]:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(jpeg_quality, 100))]
    encoded, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not encoded:
        return None

    image_base64 = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{image_base64}"


def build_gemma_payload(config: GemmaAnalysisConfig, frame_data_urls: List[str]) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": PROMPT}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": frame_data_url},
        }
        for frame_data_url in frame_data_urls
    )

    return {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": 0,
        "max_tokens": config.max_tokens,
    }


def post_json(url: str, payload: Dict[str, Any], timeout_seconds: int) -> Dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_message_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        content = content.strip()
        if content:
            return content

    if isinstance(content, list):
        content_text = "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
        if content_text:
            return content_text

    reasoning_content = message.get("reasoning_content", "")
    if isinstance(reasoning_content, str):
        return reasoning_content.strip()

    return ""


def extract_finish_reason(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""

    return str(choices[0].get("finish_reason", "") or "")


def parse_analysis_content(content: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(extract_json_object(content))
    except json.JSONDecodeError:
        return {
            "scene_description": content,
            "break_in_likely": None,
            "confidence": None,
            "reasoning": "Gemma did not return parseable JSON.",
        }

    return {
        "scene_description": str(parsed.get("scene_description", "")).strip(),
        "break_in_likely": parsed.get("break_in_likely"),
        "confidence": parsed.get("confidence"),
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }


def print_gemma_analysis(clip_path: Path, analysis: Dict[str, Any]) -> None:
    print(f"\nGemma analysis for {clip_path}:")
    print(f"Scene: {analysis.get('scene_description')}")
    print(f"Break-in likely: {analysis.get('break_in_likely')}")
    print(f"Confidence: {analysis.get('confidence')}")
    print(f"Reasoning: {analysis.get('reasoning')}")
    finish_reason = analysis.get("finish_reason")
    if finish_reason:
        print(f"Finish reason: {finish_reason}")
    print()


def extract_json_object(content: str) -> str:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return content
    return content[start : end + 1]
