"""Local pipe client for the isolated native accelerator worker.

The host process can remain free of OpenVINO binary
dependencies. A single long-lived Python 3.12 child owns native devices,
compiled models, infer requests, and GenAI pipelines. Requests are serialized
over inherited pipes: there is no port, remote transport, or telemetry path.
"""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import threading
import time
from typing import Any, Callable, Iterator
from uuid import uuid4

from .hardware import HardwareProfiler, accelerator_environment


class DirectHardwareError(RuntimeError):
    """Raised when the isolated native worker cannot safely serve a request."""


class DirectHardwareClient:
    """Own one serialized, restartable accelerator process.

    Inference and generation requests are never automatically replayed. A
    native failure may happen after a device already executed the request, so
    retrying would violate at-most-once action semantics for callers that bind
    generation to tools.
    """

    def __init__(self, profiler: HardwareProfiler) -> None:
        self.profiler = profiler
        self._process: subprocess.Popen[str] | None = None
        self._responses: Queue[dict[str, Any]] = Queue()
        self._stderr: deque[str] = deque(maxlen=80)
        self._request_lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._started_at = 0.0

    @property
    def accelerator_python(self) -> str:
        return self.profiler.accelerator_python_path()

    def available(self) -> bool:
        return bool(self.accelerator_python)

    def status(self, *, start: bool = False) -> dict[str, Any]:
        process = self._process
        running = bool(process and process.poll() is None)
        if start or running:
            live = self.request("status", timeout=30)
            return {
                **live,
                "running": True,
                "transport": "inherited-jsonl-pipes",
            }
        return {
            "status": "running" if running else ("available" if self.available() else "unavailable"),
            "running": running,
            "pid": process.pid if running and process else None,
            "accelerator_python": self.accelerator_python,
            "uptime_seconds": round(time.time() - self._started_at, 3) if running else 0.0,
            "stderr_tail": list(self._stderr)[-8:] if not running else [],
            "transport": "inherited-jsonl-pipes",
            "privacy_boundary": "No socket is opened. Native commands and results stay between parent and child processes on this machine.",
        }

    def request(
        self,
        action: str,
        *,
        timeout: float = 120.0,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        request_id = uuid4().hex
        command = {**payload, "action": action, "request_id": request_id}
        with self._request_lock:
            self._ensure_started()
            process = self._process
            if process is None or process.stdin is None:
                raise DirectHardwareError("The native accelerator worker did not expose a command pipe.")
            try:
                process.stdin.write(json.dumps(command, separators=(",", ":"), default=_json_safe) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._terminate()
                raise DirectHardwareError(self._failure_message("Native worker command pipe closed", exc)) from exc
            deadline = time.monotonic() + max(0.1, float(timeout))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._terminate()
                    raise DirectHardwareError(
                        f"Native worker timed out during '{action}'. The worker was stopped; the request was not replayed."
                    )
                try:
                    response = self._responses.get(timeout=min(remaining, 0.25))
                except Empty:
                    if process.poll() is not None:
                        raise DirectHardwareError(self._failure_message("Native worker exited", process.returncode))
                    continue
                if str(response.get("request_id", "")) != request_id:
                    self._terminate()
                    raise DirectHardwareError("Native worker response ordering was violated; the worker was stopped.")
                if response.get("status") == "event":
                    event = response.get("event")
                    if on_event is not None and isinstance(event, dict):
                        on_event(event)
                    continue
                if response.get("status") != "ok":
                    error = str(response.get("error", "Native worker rejected the request"))
                    error_type = str(response.get("type", "RuntimeError"))
                    raise DirectHardwareError(f"{error_type}: {error}")
                result = response.get("result")
                if not isinstance(result, dict):
                    raise DirectHardwareError("Native worker returned a malformed result.")
                return result

    def load_model(
        self,
        path: str,
        *,
        device: str,
        model_id: str = "",
        cache_dir: str = "",
        config: dict[str, Any] | None = None,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        return self.request(
            "load", path=path, device=device, model_id=model_id,
            cache_dir=cache_dir, config=config or {}, timeout=timeout,
        )

    def load_llm(
        self,
        path: str,
        *,
        device: str,
        model_id: str = "",
        cache_dir: str = "",
        config: dict[str, Any] | None = None,
        scheduler: dict[str, Any] | None = None,
        timeout: float = 900.0,
    ) -> dict[str, Any]:
        return self.request(
            "load_llm", path=path, device=device, model_id=model_id,
            cache_dir=cache_dir, config=config or {}, scheduler=scheduler or {}, timeout=timeout,
        )

    def infer(
        self,
        model_id: str,
        *,
        inputs: dict[str, Any] | str | None = None,
        output_mode: str = "summary",
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        return self.request(
            "infer", model_id=model_id, inputs=inputs, output_mode=output_mode, timeout=timeout,
        )

    def generate(
        self,
        model_id: str,
        *,
        messages: list[dict[str, str]] | None = None,
        prompt: str = "",
        generation: dict[str, Any] | None = None,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        return self.request(
            "generate", model_id=model_id, messages=messages,
            prompt=prompt, generation=generation or {}, timeout=timeout,
        )

    def stream_generate(
        self,
        model_id: str,
        *,
        messages: list[dict[str, str]] | None = None,
        prompt: str = "",
        generation: dict[str, Any] | None = None,
        timeout: float = 600.0,
    ) -> Iterator[dict[str, Any]]:
        events: Queue[tuple[str, Any]] = Queue()
        finished = threading.Event()

        def on_event(event: dict[str, Any]) -> None:
            events.put(("event", event))

        def execute() -> None:
            try:
                result = self.request(
                    "generate_stream", model_id=model_id, messages=messages,
                    prompt=prompt, generation=generation or {}, timeout=timeout,
                    on_event=on_event,
                )
                events.put(("done", result))
            except Exception as exc:
                events.put(("error", exc))
            finally:
                finished.set()

        thread = threading.Thread(target=execute, name="openagent-direct-stream", daemon=True)
        thread.start()
        try:
            while True:
                kind, value = events.get()
                if kind == "event":
                    yield dict(value)
                elif kind == "done":
                    yield {"type": "done", **dict(value)}
                    break
                else:
                    raise value
        finally:
            if not finished.is_set():
                self._terminate()

    def benchmark(
        self,
        model_id: str,
        *,
        iterations: int = 20,
        warmup: int = 2,
        inputs: dict[str, Any] | str | None = None,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        return self.request(
            "benchmark", model_id=model_id, iterations=iterations,
            warmup=warmup, inputs=inputs, timeout=timeout,
        )

    def calibrate(
        self,
        path: str,
        *,
        devices: list[str],
        iterations: int = 10,
        absolute_tolerance: float = 0.03,
        cache_dir: str = "",
        keep_model_id: str = "",
        timeout: float = 1800.0,
    ) -> dict[str, Any]:
        return self.request(
            "calibrate", path=path, devices=devices, iterations=iterations,
            absolute_tolerance=absolute_tolerance, cache_dir=cache_dir,
            keep_model_id=keep_model_id, timeout=timeout,
        )

    def unload(self, model_id: str) -> dict[str, Any]:
        return self.request("unload", model_id=model_id, timeout=30)

    def close(self) -> None:
        with self._request_lock:
            process = self._process
            if not process or process.poll() is not None:
                self._clear_process()
                return
            try:
                if process.stdin:
                    process.stdin.write(json.dumps({"action": "shutdown", "request_id": uuid4().hex}) + "\n")
                    process.stdin.flush()
                process.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._terminate()
            finally:
                self._clear_process()

    def _ensure_started(self) -> None:
        with self._lifecycle_lock:
            if self._process and self._process.poll() is None:
                return
            self._clear_process()
            python = self.accelerator_python
            if not python:
                raise DirectHardwareError(
                    "The accelerator runtime is unavailable. Install the OpenVINO extras and set KERNELLOOM_ACCELERATOR_PYTHON."
                )
            root = self.profiler.project_root
            environment = accelerator_environment(root)
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            try:
                process = subprocess.Popen(
                    [
                        python, "-m", "openagent_engine.worker", "--protocol", "jsonl",
                        "--parent-pid", str(os.getpid()),
                    ],
                    cwd=str(root), env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                    bufsize=1, creationflags=creationflags,
                )
            except OSError as exc:
                raise DirectHardwareError(f"Could not start the isolated accelerator worker: {exc}") from exc
            self._process = process
            self._started_at = time.time()
            self._stdout_thread = threading.Thread(
                target=self._read_stdout, args=(process,), name="openagent-accelerator-output", daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._read_stderr, args=(process,), name="openagent-accelerator-errors", daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is None:
            return
        for line in stream:
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    self._responses.put(value)
                else:
                    self._responses.put({"status": "error", "error": "Worker emitted non-object JSON"})
            except json.JSONDecodeError:
                self._stderr.append(f"non-json stdout: {line.strip()[:400]}")

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        stream = process.stderr
        if stream is None:
            return
        for line in stream:
            clean = line.rstrip()
            if clean:
                self._stderr.append(clean[:1000])

    def _terminate(self) -> None:
        process = self._process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._clear_process()

    def _clear_process(self) -> None:
        process = self._process
        threads = (self._stdout_thread, self._stderr_thread)
        for thread in threads:
            if thread and thread is not threading.current_thread() and thread.is_alive():
                thread.join(timeout=1)
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except OSError:
                        pass
        self._process = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._started_at = 0.0
        while True:
            try:
                self._responses.get_nowait()
            except Empty:
                break

    def _failure_message(self, prefix: str, detail: object) -> str:
        tail = " | ".join(list(self._stderr)[-6:])
        suffix = f"; native stderr: {tail}" if tail else ""
        return f"{prefix} ({detail}){suffix}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)
