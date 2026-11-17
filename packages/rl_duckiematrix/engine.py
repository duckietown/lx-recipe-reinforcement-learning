"""Locate the Duckiematrix engine from wherever the code runs.

The gym-mode engine serves DTPS on port 7501. The notebooks run inside the VSCode editor
container, which is on Docker's bridge network, so ``127.0.0.1`` is not the laptop: on Docker
Desktop (macOS/Windows) the laptop is ``host.docker.internal``, on Linux it is the bridge gateway.
"""

from __future__ import annotations

import os
import socket
import struct

ENGINE_PORT = 7501


def _reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _default_gateway() -> str | None:
    """The default gateway of this container (the Docker host on Linux), from ``/proc/net/route``."""
    try:
        with open("/proc/net/route") as routes:
            next(routes)  # header
            for line in routes:
                fields = line.split()
                # destination 00000000 = default route; the gateway is a little-endian hex IPv4
                if len(fields) > 2 and fields[1] == "00000000":
                    return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    except (OSError, StopIteration, ValueError):
        return None
    return None


def candidate_hosts() -> list[str]:
    """Hosts to try, in order: ``DUCKIEMATRIX_HOST``, localhost, Docker host, default gateway."""
    candidates = []
    explicit = os.environ.get("DUCKIEMATRIX_HOST")
    if explicit:
        candidates.append(explicit)
    candidates.extend(["127.0.0.1", "host.docker.internal"])
    gateway = _default_gateway()
    if gateway:
        candidates.append(gateway)
    return list(dict.fromkeys(candidates))  # drop duplicates, keep the order


def resolve_engine_host(port: int = ENGINE_PORT) -> str:
    """Return the first candidate host with the engine's DTPS port open.

    Raises:
        ConnectionError: when no candidate answers. The message tells the learner to start the
            Duckiematrix in gym mode.

    """
    candidates = candidate_hosts()
    for host in candidates:
        if _reachable(host, port):
            return host
    message = (
        f"No Duckiematrix engine found on port {port} (tried {', '.join(candidates)}). "
        "Start it in gym mode (see the README), then run this cell again."
    )
    raise ConnectionError(message)
