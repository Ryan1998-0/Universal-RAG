from __future__ import annotations

import threading
from typing import Callable, Optional


class LeaseHeartbeat:
    def __init__(
        self,
        heartbeat_fn: Callable[[str], None],
        *,
        initial_stage: str,
        interval_seconds: float,
    ):
        self.heartbeat_fn = heartbeat_fn
        self.interval_seconds = max(0.01, float(interval_seconds))
        self._stage = str(initial_stage)
        self._stage_lock = threading.Lock()
        self._stop = threading.Event()
        self._failure: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run,
            name="rag-lease-heartbeat",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def set_stage(self, stage: str) -> None:
        with self._stage_lock:
            self._stage = str(stage)

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("background lease heartbeat failed") from self._failure

    def stop(self) -> None:
        self.cancel()
        self.raise_if_failed()

    def cancel(self) -> None:
        self._stop.set()
        if self._started and self._thread.is_alive():
            self._thread.join(timeout=min(5.0, self.interval_seconds + 1.0))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                with self._stage_lock:
                    stage = self._stage
                self.heartbeat_fn(stage)
            except BaseException as exc:
                self._failure = exc
                self._stop.set()
                return
