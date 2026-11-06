"""Local structured logging. Never capture environment values or webhook tokens."""

from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time


def redact(value):
    return re.sub(r"https?://(?:\w+\.)?discord(?:app)?\.com/api(?:/v\d+)?/webhooks/[^\s\"'<>]+",
                  "[DISCORD_WEBHOOK_REDACTED]", str(value))


def resources():
    result = {"pid": os.getpid()}
    try:
        import psutil
        p = psutil.Process()
        result.update(rss_bytes=p.memory_info().rss, cpu_seconds=sum(p.cpu_times()[:2]),
                      available_ram_bytes=psutil.virtual_memory().available,
                      total_ram_bytes=psutil.virtual_memory().total)
    except ImportError:
        pass
    return result


def event(path, kind, **details):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_seconds": time.monotonic(),
              "event": kind, **details}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(redact(json.dumps(record, default=str, allow_nan=False)) + "\n")


def environment():
    versions = {}
    for package in ("numpy", "torch", "scikit-learn", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "platform": platform.platform(), "cpu": platform.processor(),
            "logical_cores": os.cpu_count(), "versions": versions, "resources": resources()}


class Tee:
    def __init__(self, original, stream):
        self.original, self.stream = original, stream
    def write(self, value):
        value = redact(value)
        self.original.write(value)
        self.stream.write(value)
        self.stream.flush()
        return len(value)
    def flush(self):
        self.original.flush(); self.stream.flush()
    def isatty(self):
        return False


@contextmanager
def capture_console(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_out, old_err = sys.stdout, sys.stderr
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        sys.stdout, sys.stderr = Tee(old_out, stream), Tee(old_err, stream)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


@contextmanager
def heartbeat(path, interval=60, **details):
    stopped = threading.Event()
    def pulse():
        while not stopped.wait(interval):
            event(path, "heartbeat", **details, resources=resources())
    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=2)
