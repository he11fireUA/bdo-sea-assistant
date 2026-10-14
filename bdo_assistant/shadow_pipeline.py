"""Bounded, non-authoritative candidate analysis for live shadow mode."""
from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
import threading
from typing import Callable, Mapping, Protocol

import numpy as np

from .monitoring import MonitoringSample


class ShadowProcessor(Protocol):
    def process(self, sample: MonitoringSample) -> Mapping[str, object]: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class ShadowAccounting:
    submitted: int = 0
    completed: int = 0
    dropped: int = 0
    failed: int = 0


@dataclass(frozen=True)
class ShadowMetrics:
    """Structured candidate output; image arrays are deliberately forbidden."""

    source_sequence: int
    processing_ordinal: int
    received_monotonic: float
    native_source_timestamp: float | None
    calibration_epoch: int
    values: dict[str, str | int | float | bool | None]


def _structured_values(values: Mapping[str, object]) -> dict[str, str | int | float | bool | None]:
    if not isinstance(values, Mapping):
        raise ValueError("shadow processor must return a mapping of structured metrics")
    result: dict[str, str | int | float | bool | None] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("shadow metric names must be non-empty strings")
        if isinstance(value, np.ndarray) or not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError("shadow metrics must be scalar structured values, never frames")
        result[key] = value
    return result


class ShadowPipelineRunner:
    """Run candidate work behind a one-way, bounded queue.

    Candidate output only becomes ``ShadowMetrics``.  It cannot produce a
    monitoring result or notification and therefore cannot affect legacy state.
    """

    def __init__(
        self,
        processor: ShadowProcessor,
        *,
        queue_capacity: int = 1,
        on_metrics: Callable[[], None] | None = None,
    ) -> None:
        if not callable(getattr(processor, "process", None)):
            raise ValueError("shadow_processor must provide process")
        if not callable(getattr(processor, "reset", None)):
            raise ValueError("shadow_processor must provide reset")
        if type(queue_capacity) is not int or queue_capacity < 1:
            raise ValueError("shadow_queue_capacity must be a positive integer")
        if on_metrics is not None and not callable(on_metrics):
            raise ValueError("on_metrics must be callable or None")
        self._processor = processor
        self._on_metrics = on_metrics
        self._queue: Queue[tuple[int, MonitoringSample] | None] = Queue(maxsize=queue_capacity)
        self._lock = threading.RLock()
        self._processor_lock = threading.RLock()
        self._generation = 0
        self._accounting = ShadowAccounting()
        self._latest: ShadowMetrics | None = None
        self._worker = threading.Thread(target=self._run, name="candidate-shadow", daemon=True)
        self._worker.start()

    @property
    def accounting(self) -> ShadowAccounting:
        with self._lock:
            return self._accounting

    @property
    def latest_metrics(self) -> ShadowMetrics | None:
        with self._lock:
            return self._latest

    def submit(self, sample: MonitoringSample) -> bool:
        if type(sample) is not MonitoringSample:
            raise ValueError("shadow sample must be MonitoringSample")
        with self._lock:
            generation = self._generation
        try:
            self._queue.put_nowait((generation, sample))
        except Full:
            with self._lock:
                self._accounting = ShadowAccounting(
                    self._accounting.submitted,
                    self._accounting.completed,
                    self._accounting.dropped + 1,
                    self._accounting.failed,
                )
            return False
        with self._lock:
            self._accounting = ShadowAccounting(
                self._accounting.submitted + 1,
                self._accounting.completed,
                self._accounting.dropped,
                self._accounting.failed,
            )
        return True

    def reset(self) -> None:
        with self._processor_lock:
            with self._lock:
                self._generation += 1
                self._latest = None
            while True:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
                else:
                    with self._lock:
                        self._accounting = ShadowAccounting(
                            self._accounting.submitted,
                            self._accounting.completed,
                            self._accounting.dropped + 1,
                            self._accounting.failed,
                        )
                    self._queue.task_done()
            self._processor.reset()

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                return
            generation, sample = task
            try:
                with self._processor_lock:
                    with self._lock:
                        if generation != self._generation:
                            self._accounting = ShadowAccounting(
                                self._accounting.submitted,
                                self._accounting.completed,
                                self._accounting.dropped + 1,
                                self._accounting.failed,
                            )
                            continue
                    values = _structured_values(self._processor.process(sample))
                metrics = ShadowMetrics(
                    source_sequence=sample.source_sequence,
                    processing_ordinal=sample.processing_ordinal,
                    received_monotonic=sample.received_monotonic,
                    native_source_timestamp=sample.native_source_timestamp,
                    calibration_epoch=sample.calibration_epoch,
                    values=values,
                )
            except Exception:
                with self._lock:
                    self._accounting = ShadowAccounting(
                        self._accounting.submitted,
                        self._accounting.completed,
                        self._accounting.dropped,
                        self._accounting.failed + 1,
                    )
            else:
                published = False
                with self._lock:
                    if generation == self._generation:
                        self._latest = metrics
                        self._accounting = ShadowAccounting(
                            self._accounting.submitted,
                            self._accounting.completed + 1,
                            self._accounting.dropped,
                            self._accounting.failed,
                        )
                        published = True
                    else:
                        self._accounting = ShadowAccounting(
                            self._accounting.submitted,
                            self._accounting.completed,
                            self._accounting.dropped + 1,
                            self._accounting.failed,
                        )
                if published and self._on_metrics is not None:
                    try:
                        self._on_metrics()
                    except Exception:
                        pass
            finally:
                self._queue.task_done()
