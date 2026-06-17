import os
import re
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Set, Tuple

from app_config import load_dotenv


SCAN_PORTS = (554, 80, 8000, 8080, 8899)
PORT_SCORES = {
    554: 8,
    8899: 5,
    8000: 4,
    80: 2,
    8080: 2,
}
SCAN_CONNECT_TIMEOUT_SECONDS = 0.15
SCAN_WORKERS = 64


def likely_dvr_ip() -> Optional[str]:
    load_dotenv()
    configured_ip = first_env_value("CAMERA_IP", "NVR_IP", "DVR_IP")
    if configured_ip:
        return configured_ip

    return scan_for_likely_dvr_ip() or default_gateway_ip()


def first_env_value(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def default_gateway_ip() -> Optional[str]:
    return linux_default_gateway_ip() or macos_default_gateway_ip() or socket_route_gateway_ip()


def scan_for_likely_dvr_ip() -> Optional[str]:
    local_ips = local_ipv4_addresses()
    gateway_ip = default_gateway_ip()
    candidates = sorted(candidate_ips_for_local_addresses(local_ips))
    if gateway_ip:
        candidates = [candidate for candidate in candidates if candidate != gateway_ip]

    if not candidates:
        return None

    scored_hosts = []
    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as executor:
        future_to_ip = {
            executor.submit(open_dvr_ports, ip_address): ip_address
            for ip_address in candidates
        }
        for future in as_completed(future_to_ip):
            ip_address = future_to_ip[future]
            try:
                open_ports = future.result()
            except OSError:
                continue

            score = host_score(open_ports)
            if score > 0:
                scored_hosts.append((score, ip_address, open_ports))

    if not scored_hosts:
        return None

    scored_hosts.sort(key=lambda host: (-host[0], ip_sort_key(host[1])))
    return scored_hosts[0][1]


def local_ipv4_addresses() -> Set[str]:
    addresses = linux_local_ipv4_addresses() | macos_local_ipv4_addresses()
    socket_ip = socket_local_ip()
    if socket_ip:
        addresses.add(socket_ip)

    return {
        address
        for address in addresses
        if not address.startswith("127.") and not address.startswith("169.254.")
    }


def linux_local_ipv4_addresses() -> Set[str]:
    try:
        completed_process = subprocess.run(
            ["ip", "-o", "-4", "addr", "show", "scope", "global"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()

    return set(re.findall(r"\binet\s+(\d+\.\d+\.\d+\.\d+)/", completed_process.stdout))


def macos_local_ipv4_addresses() -> Set[str]:
    try:
        completed_process = subprocess.run(
            ["ifconfig"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()

    return set(re.findall(r"\binet\s+(\d+\.\d+\.\d+\.\d+)\s+netmask", completed_process.stdout))


def socket_local_ip() -> Optional[str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def candidate_ips_for_local_addresses(local_ips: Set[str]) -> Set[str]:
    candidates = set()
    for local_ip in local_ips:
        octets = local_ip.split(".")
        if len(octets) != 4:
            continue

        prefix = ".".join(octets[:3])
        for host_number in range(1, 255):
            candidate = f"{prefix}.{host_number}"
            if candidate != local_ip:
                candidates.add(candidate)
    return candidates


def open_dvr_ports(ip_address: str) -> Tuple[int, ...]:
    open_ports = []
    for port in SCAN_PORTS:
        if can_connect(ip_address, port):
            open_ports.append(port)

    return tuple(open_ports)


def can_connect(ip_address: str, port: int) -> bool:
    try:
        with socket.create_connection((ip_address, port), timeout=SCAN_CONNECT_TIMEOUT_SECONDS):
            return True
    except OSError:
        return False


def host_score(open_ports: Tuple[int, ...]) -> int:
    score = sum(PORT_SCORES.get(port, 0) for port in open_ports)
    if 554 in open_ports and any(port in open_ports for port in (80, 8000, 8080, 8899)):
        score += 5
    return score


def ip_sort_key(ip_address: str) -> Tuple[int, int, int, int]:
    return tuple(int(part) for part in ip_address.split("."))


def linux_default_gateway_ip() -> Optional[str]:
    try:
        completed_process = subprocess.run(
            ["ip", "route", "show", "default"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    match = re.search(r"\bvia\s+(\d+\.\d+\.\d+\.\d+)", completed_process.stdout)
    return match.group(1) if match else None


def macos_default_gateway_ip() -> Optional[str]:
    try:
        completed_process = subprocess.run(
            ["route", "-n", "get", "default"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    match = re.search(r"\bgateway:\s+(\d+\.\d+\.\d+\.\d+)", completed_process.stdout)
    return match.group(1) if match else None


def socket_route_gateway_ip() -> Optional[str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
    except OSError:
        return None

    octets = local_ip.split(".")
    if len(octets) != 4:
        return None

    return ".".join(octets[:3] + ["1"])
