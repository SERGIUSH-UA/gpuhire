"""Minimal Job that probes outbound network from the remote runtime.

Used to verify whether ``enable_internet=True`` actually grants network egress
in the chosen Kaggle account. Should complete in <30s and produce one file
``/kaggle/working/net_probe.txt`` summarizing reachability.
"""

from __future__ import annotations

from typing import Any, ClassVar

from gpurunner.core.job import Job


class NetProbeJob(Job):
    """Probe DNS + HTTP egress and write a tiny report to working dir."""

    name: ClassVar[str] = "net-probe"
    description: ClassVar[str] = "Test whether the remote kernel actually has internet."
    supported_backends: ClassVar[tuple[str, ...]] = (
        "kaggle",
        "modal",
        "colab",
        "vast",
        "lightning",
        "beam",
        "saturn",
    )

    def requirements(self) -> list[str]:
        # Nothing to install — uses only stdlib.
        return []

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        _ = params
        return {}

    def render_remote_code(
        self,
        params: dict[str, Any],
        *,
        shard_index: int = 0,
        total_shards: int = 1,
    ) -> str:
        _ = (params, shard_index, total_shards)
        return """\
import socket, subprocess, urllib.request
from pathlib import Path

lines = []
lines.append("=== net-probe ===")

# DNS
try:
    ip = socket.gethostbyname("pypi.org")
    lines.append(f"DNS pypi.org   -> {ip}  OK")
except Exception as e:
    lines.append(f"DNS pypi.org   FAIL: {e}")

try:
    ip = socket.gethostbyname("kaggle.com")
    lines.append(f"DNS kaggle.com -> {ip}  OK")
except Exception as e:
    lines.append(f"DNS kaggle.com FAIL: {e}")

# HTTP
for url in ("https://pypi.org/simple/", "https://www.kaggle.com/", "https://example.com/"):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            lines.append(f"HTTP {url}: {r.status} ({len(r.read(64))}+ bytes)")
    except Exception as e:
        lines.append(f"HTTP {url}: FAIL {type(e).__name__}: {e}")

# pip show (does pip even have an index configured?)
try:
    r = subprocess.run(
        ["pip", "config", "list"], capture_output=True, text=True, timeout=10
    )
    lines.append("--- pip config ---")
    lines.append(r.stdout.strip() or "(empty)")
except Exception as e:
    lines.append(f"pip config FAIL: {e}")

out = Path("/kaggle/working/net_probe.txt")
out.parent.mkdir(parents=True, exist_ok=True)
report = "\\n".join(lines)
out.write_text(report, encoding="utf-8")
print(report, flush=True)
"""
