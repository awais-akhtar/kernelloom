"""Local system metrics used by the hardware profiler."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any


_CPU_SAMPLE: tuple[float, float] | None = None


class _FILETIME(ctypes.Structure):
    _fields_ = [
        ("dwLowDateTime", wintypes.DWORD),
        ("dwHighDateTime", wintypes.DWORD),
    ]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def system_metrics(data_dir: str | Path) -> dict[str, Any]:
    data_path = Path(data_dir)
    disk_usage = shutil.disk_usage(data_path if data_path.exists() else data_path.anchor or ".")
    memory = _memory_metrics()
    cpu_percent = _cpu_percent()
    return {
        "status": "ok",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "cpu": {
            "count": os.cpu_count() or 1,
            "usage_percent": cpu_percent,
            "process_seconds": round(time.process_time(), 3),
        },
        "memory": memory,
        "disk": {
            "path": str(data_path),
            "total_gb": _gb(disk_usage.total),
            "used_gb": _gb(disk_usage.used),
            "free_gb": _gb(disk_usage.free),
            "used_percent": _percent(disk_usage.used, disk_usage.total),
        },
        "privacy_boundary": "System telemetry is read from the local OS and local data directory only.",
    }


def _memory_metrics() -> dict[str, Any]:
    if platform.system().lower() == "windows":
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            used = max(0, status.ullTotalPhys - status.ullAvailPhys)
            return {
                "total_gb": _gb(status.ullTotalPhys),
                "available_gb": _gb(status.ullAvailPhys),
                "used_gb": _gb(used),
                "used_percent": float(status.dwMemoryLoad),
            }
    total = _sysconf_bytes("SC_PAGE_SIZE", "SC_PHYS_PAGES")
    available = _available_memory_fallback(total)
    used = max(0, total - available)
    return {
        "total_gb": _gb(total),
        "available_gb": _gb(available),
        "used_gb": _gb(used),
        "used_percent": _percent(used, total),
    }


def _cpu_percent() -> float:
    global _CPU_SAMPLE
    if platform.system().lower() == "windows":
        idle = _FILETIME()
        kernel = _FILETIME()
        user = _FILETIME()
        if ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined]
            ctypes.byref(idle),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            idle_value = _filetime_value(idle)
            total_value = _filetime_value(kernel) + _filetime_value(user)
            previous = _CPU_SAMPLE
            _CPU_SAMPLE = (idle_value, total_value)
            if previous:
                idle_delta = idle_value - previous[0]
                total_delta = total_value - previous[1]
                if total_delta > 0:
                    return round(max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta))), 1)
        return 0.0
    try:
        load = os.getloadavg()[0]
        count = max(1, os.cpu_count() or 1)
        return round(max(0.0, min(100.0, load / count * 100.0)), 1)
    except (AttributeError, OSError):
        return 0.0


def _filetime_value(value: _FILETIME) -> float:
    return float((int(value.dwHighDateTime) << 32) + int(value.dwLowDateTime))


def _sysconf_bytes(page_size_name: str, pages_name: str) -> int:
    try:
        page_size = os.sysconf(page_size_name)
        pages = os.sysconf(pages_name)
        return int(page_size) * int(pages)
    except (AttributeError, OSError, ValueError):
        return 0


def _available_memory_fallback(total: int) -> int:
    if platform.system().lower() == "linux":
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
    return total


def _gb(value: int | float) -> float:
    return round(float(value) / (1024 ** 3), 2) if value else 0.0


def _percent(used: int | float, total: int | float) -> float:
    return round((float(used) / float(total)) * 100.0, 1) if total else 0.0
