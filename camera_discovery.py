import base64
import hashlib
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen


ONVIF_TIMEOUT_SECONDS = 4
RTSP_PROBE_TEMPLATES = (
    "/Streaming/Channels/{channel}",
    "/unicast/c{camera}/s1/live",
    "/unicast/c{camera}/s0/live",
    "/cam/realmonitor?channel={camera}&subtype=1",
    "/cam/realmonitor?channel={camera}&subtype=0",
    "/live/ch{camera}",
    "/h264Preview_01_sub",
)

SOAP_NS = "http://www.w3.org/2003/05/soap-envelope"
WSSE_NS = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
WSU_NS = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
TD_NS = "http://www.onvif.org/ver10/device/wsdl"
TRT_NS = "http://www.onvif.org/ver10/media/wsdl"
TT_NS = "http://www.onvif.org/ver10/schema"


@dataclass(frozen=True)
class DiscoveredStream:
    label: str
    stream_id: str
    rtsp_url: Optional[str] = None
    rtsp_path_template: Optional[str] = None
    channel: Optional[str] = None
    source: str = "rtsp_probe"


def discover_nvr_streams(settings, can_open_channel, progress_callback=None) -> List[DiscoveredStream]:
    report_progress(progress_callback, "ONVIF discovery: starting...")
    onvif_streams = discover_onvif_streams(settings, progress_callback=progress_callback)
    if onvif_streams:
        report_progress(progress_callback, f"ONVIF discovery: succeeded, found {len(onvif_streams)} stream(s).")
        return onvif_streams

    report_progress(progress_callback, "ONVIF discovery: failed, no streams found.")
    report_progress(progress_callback, "RTSP probing: starting common URL patterns...")
    rtsp_streams = probe_rtsp_templates(settings, can_open_channel, progress_callback=progress_callback)
    if rtsp_streams:
        report_progress(progress_callback, f"RTSP probing: succeeded, found {len(rtsp_streams)} stream(s).")
    else:
        report_progress(progress_callback, "RTSP probing: failed, no streams found.")
    return rtsp_streams


def discover_onvif_streams(settings, progress_callback=None) -> List[DiscoveredStream]:
    username = settings["username"]
    password = settings["password"]
    ip_address = settings["ip_address"]
    device_urls = (
        f"http://{ip_address}/onvif/device_service",
        f"http://{ip_address}:80/onvif/device_service",
        f"http://{ip_address}:8899/onvif/device_service",
        f"http://{ip_address}:8000/onvif/device_service",
    )

    for device_url in device_urls:
        try:
            report_progress(progress_callback, f"ONVIF discovery: trying {device_url}")
            media_url = get_onvif_media_url(device_url, username, password)
            if not media_url:
                report_progress(progress_callback, f"ONVIF discovery: no media service at {device_url}")
                continue

            report_progress(progress_callback, f"ONVIF discovery: media service found at {media_url}")
            profiles = get_onvif_profiles(media_url, username, password)
            streams = []
            for index, profile_token in enumerate(profiles, start=1):
                report_progress(progress_callback, f"ONVIF discovery: requesting stream URI for profile {index}")
                rtsp_url = get_onvif_stream_uri(media_url, username, password, profile_token)
                if not rtsp_url:
                    continue
                rtsp_url = add_credentials_to_rtsp_url(rtsp_url, username, password)

                stream_id = f"onvif:{profile_token}"
                streams.append(
                    DiscoveredStream(
                        label=f"ONVIF Stream {index}",
                        stream_id=stream_id,
                        rtsp_url=rtsp_url,
                        source="onvif",
                    )
                )

            if streams:
                logging.info("ONVIF discovered %s stream(s) from %s", len(streams), media_url)
                return streams
        except Exception as exc:
            report_progress(progress_callback, f"ONVIF discovery: failed at {device_url} ({type(exc).__name__})")
            logging.info("ONVIF discovery failed for %s: %s", device_url, exc)

    return []


def probe_rtsp_templates(settings, can_open_channel, progress_callback=None) -> List[DiscoveredStream]:
    streams = []
    seen_urls = set()

    for template in RTSP_PROBE_TEMPLATES:
        report_progress(progress_callback, f"RTSP probing: trying template {template}")
        probe_settings = dict(settings)
        probe_settings["rtsp_path_template"] = template
        for camera_number, stream_name, channel in iter_probe_channels(template):
            try:
                rtsp_url = build_probe_rtsp_url(probe_settings, channel)
                if rtsp_url in seen_urls:
                    continue
                seen_urls.add(rtsp_url)

                if not can_open_channel(probe_settings, channel):
                    continue

                stream = DiscoveredStream(
                    label=f"Camera {camera_number} {stream_name} ({template})",
                    stream_id=f"probe:{template}:{channel}",
                    rtsp_path_template=template,
                    channel=channel,
                    source="rtsp_probe",
                )
                streams.append(stream)
                report_progress(progress_callback, f"RTSP probing: found {stream.label}")
            except Exception as exc:
                logging.info("RTSP probe failed for template=%s channel=%s: %s", template, channel, exc)

    return streams


def report_progress(progress_callback, message):
    logging.info(message)
    if progress_callback:
        progress_callback(message)


def iter_probe_channels(template: str):
    if "{channel}" in template:
        for camera_number in range(1, 17):
            yield camera_number, "Main Stream", f"{camera_number}01"
            yield camera_number, "Sub Stream", f"{camera_number}02"
        return

    if "{camera}" in template:
        for camera_number in range(1, 17):
            yield camera_number, "Stream", f"{camera_number}01"
        return

    yield 1, "Stream", "101"


def build_probe_rtsp_url(settings, channel):
    username = quote(settings["username"], safe="")
    password = quote(settings["password"], safe="")
    ip_address = settings["ip_address"]
    port = settings["port"]
    camera, stream = channel_parts(channel)
    template = settings["rtsp_path_template"]
    rendered_template = template.format(
        username=username,
        password=password,
        ip=ip_address,
        ip_address=ip_address,
        port=port,
        channel=channel or "",
        camera=camera,
        stream=stream,
    )

    if rendered_template.startswith("rtsp://"):
        return rendered_template
    if not rendered_template.startswith("/"):
        rendered_template = f"/{rendered_template}"
    return f"rtsp://{username}:{password}@{ip_address}:{port}{rendered_template}"


def add_credentials_to_rtsp_url(rtsp_url, username, password):
    parsed_url = urlsplit(rtsp_url)
    if parsed_url.scheme != "rtsp" or "@" in parsed_url.netloc:
        return rtsp_url

    username = quote(username, safe="")
    password = quote(password, safe="")
    return urlunsplit(
        (
            parsed_url.scheme,
            f"{username}:{password}@{parsed_url.netloc}",
            parsed_url.path,
            parsed_url.query,
            parsed_url.fragment,
        )
    )


def channel_parts(channel):
    channel_text = str(channel or "").strip()
    if channel_text.isdigit() and len(channel_text) >= 3:
        return channel_text[:-2], channel_text[-1]
    return channel_text, channel_text


def get_onvif_media_url(device_url, username, password):
    body = f"""
    <tds:GetCapabilities>
      <tds:Category>Media</tds:Category>
    </tds:GetCapabilities>
    """
    response = post_onvif(device_url, username, password, body)
    root = ET.fromstring(response)
    xaddr = find_first_text(root, "XAddr")
    return xaddr


def get_onvif_profiles(media_url, username, password):
    response = post_onvif(media_url, username, password, "<trt:GetProfiles />")
    root = ET.fromstring(response)
    profiles = []
    for element in root.iter():
        if local_name(element.tag) == "Profiles":
            token = element.attrib.get("token")
            if token:
                profiles.append(token)
    return profiles


def get_onvif_stream_uri(media_url, username, password, profile_token):
    body = f"""
    <trt:GetStreamUri>
      <trt:StreamSetup>
        <tt:Stream>RTP-Unicast</tt:Stream>
        <tt:Transport>
          <tt:Protocol>RTSP</tt:Protocol>
        </tt:Transport>
      </trt:StreamSetup>
      <trt:ProfileToken>{escape_xml(profile_token)}</trt:ProfileToken>
    </trt:GetStreamUri>
    """
    response = post_onvif(media_url, username, password, body)
    root = ET.fromstring(response)
    return find_first_text(root, "Uri")


def post_onvif(url, username, password, body):
    envelope = soap_envelope(username, password, body)
    request = Request(
        url,
        data=envelope.encode("utf-8"),
        headers={"Content-Type": "application/soap+xml; charset=utf-8"},
        method="POST",
    )
    with urlopen(request, timeout=ONVIF_TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", errors="replace")


def soap_envelope(username, password, body):
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    nonce = os.urandom(16)
    password_digest = base64.b64encode(
        hashlib.sha1(nonce + created.encode("utf-8") + password.encode("utf-8")).digest()
    ).decode("ascii")
    nonce_text = base64.b64encode(nonce).decode("ascii")

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="{SOAP_NS}" xmlns:wsse="{WSSE_NS}" xmlns:wsu="{WSU_NS}" xmlns:tds="{TD_NS}" xmlns:trt="{TRT_NS}" xmlns:tt="{TT_NS}">
  <s:Header>
    <wsse:Security>
      <wsse:UsernameToken>
        <wsse:Username>{escape_xml(username)}</wsse:Username>
        <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{password_digest}</wsse:Password>
        <wsse:Nonce>{nonce_text}</wsse:Nonce>
        <wsu:Created>{created}</wsu:Created>
      </wsse:UsernameToken>
    </wsse:Security>
  </s:Header>
  <s:Body>{body}</s:Body>
</s:Envelope>"""


def find_first_text(root, wanted_local_name):
    for element in root.iter():
        if local_name(element.tag) == wanted_local_name and element.text:
            return element.text.strip()
    return None


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def escape_xml(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
