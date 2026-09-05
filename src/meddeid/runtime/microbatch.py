from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, replace
from threading import Condition, Thread
from time import monotonic
from typing import Any

from meddeid.pipeline.types import PreparedWindow, WindowPrediction

from .base import InferenceRuntime


class MicroBatcherOverloaded(RuntimeError):
    """Raised when a bounded inference queue cannot admit another request."""


@dataclass(frozen=True)
class _QueueItem:
    window: PreparedWindow
    future: Future[WindowPrediction]
    bucket_length: int
    enqueued_at: float


class MicroBatchRuntime(InferenceRuntime):
    """Coalesce windows from concurrent callers into bounded runtime batches.

    Sequence buckets are optional. When configured, a window is placed in the
    smallest bucket that contains it and is padded to that exact length before
    the wrapped runtime is called. Predictions are sliced back to the original
    length before they leave this wrapper.
    """

    supports_concurrent_requests = True

    def __init__(
        self,
        runtime: InferenceRuntime,
        *,
        max_windows: int,
        max_tokens: int,
        max_wait_ms: float,
        queue_max_windows: int = 8192,
        queue_max_requests: int = 256,
        sequence_buckets: tuple[int, ...] = (),
    ) -> None:
        if int(max_windows) < 1 or int(max_tokens) < 1:
            raise ValueError("microbatch window and token limits must be positive")
        if float(max_wait_ms) < 0:
            raise ValueError("microbatch wait must be non-negative")
        if int(queue_max_windows) < 1 or int(queue_max_requests) < 1:
            raise ValueError("microbatch queue limits must be positive")
        self.runtime = runtime
        self.max_windows = int(max_windows)
        self.max_tokens = int(max_tokens)
        self.max_wait_seconds = float(max_wait_ms) / 1000.0
        self.queue_max_windows = int(queue_max_windows)
        self.queue_max_requests = int(queue_max_requests)
        self.sequence_buckets = self._normalize_buckets(sequence_buckets)
        self._condition = Condition()
        self._queues: dict[int, deque[_QueueItem]] = {}
        self._outstanding_windows = 0
        self._active_requests = 0
        self._closed = False
        self._batches = 0
        self._windows = 0
        self._padded_tokens = 0
        self._worker = Thread(
            target=self._run,
            name="meddeid-microbatcher",
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _normalize_buckets(values: tuple[int, ...]) -> tuple[int, ...]:
        buckets = tuple(sorted({int(value) for value in values}))
        if any(value < 1 for value in buckets):
            raise ValueError("sequence buckets must contain positive integers")
        return buckets

    def _bucket_for(self, sequence_length: int) -> int:
        if not self.sequence_buckets:
            return 0
        for bucket in self.sequence_buckets:
            if sequence_length <= bucket:
                return bucket
        raise ValueError(
            f"window length {sequence_length} exceeds the largest sequence bucket "
            f"({self.sequence_buckets[-1]})"
        )

    def infer_windows(self, windows: list[PreparedWindow]) -> list[WindowPrediction]:
        if not windows:
            return []
        futures: list[Future[WindowPrediction]] = []
        queued_at = monotonic()
        with self._condition:
            if self._closed:
                raise RuntimeError("microbatch runtime is closed")
            if (
                self._active_requests + 1 > self.queue_max_requests
                or self._outstanding_windows + len(windows) > self.queue_max_windows
            ):
                raise MicroBatcherOverloaded("inference microbatch queue is full")
            items: list[_QueueItem] = []
            for window in windows:
                future: Future[WindowPrediction] = Future()
                futures.append(future)
                items.append(
                    _QueueItem(
                        window=window,
                        future=future,
                        bucket_length=self._bucket_for(window.sequence_length),
                        enqueued_at=queued_at,
                    )
                )
            self._active_requests += 1
            self._outstanding_windows += len(items)
            for item in items:
                self._queues.setdefault(item.bucket_length, deque()).append(item)
            self._condition.notify_all()

        try:
            return [future.result() for future in futures]
        finally:
            with self._condition:
                self._active_requests -= 1
                self._condition.notify_all()

    def _oldest_bucket_locked(self) -> int | None:
        candidates = [
            (queue[0].enqueued_at, bucket)
            for bucket, queue in self._queues.items()
            if queue
        ]
        if not candidates:
            return None
        return min(candidates)[1]

    def _batch_capacity_locked(self, bucket: int) -> int:
        queue = self._queues[bucket]
        total_tokens = 0
        longest_sequence = 0
        capacity = 0
        for item in queue:
            if bucket:
                candidate_tokens = total_tokens + bucket
            else:
                longest_sequence = max(
                    longest_sequence,
                    item.window.sequence_length,
                )
                candidate_tokens = longest_sequence * (capacity + 1)
            if capacity and candidate_tokens > self.max_tokens:
                break
            total_tokens = candidate_tokens
            capacity += 1
            if capacity >= self.max_windows or total_tokens >= self.max_tokens:
                break
        return max(1, capacity)

    def _take_batch_locked(self, bucket: int) -> list[_QueueItem]:
        queue = self._queues[bucket]
        limit = self._batch_capacity_locked(bucket)
        batch = [queue.popleft() for _ in range(min(limit, len(queue)))]
        if not queue:
            del self._queues[bucket]
        return batch

    @staticmethod
    def _padded_window(window: PreparedWindow, target_length: int) -> PreparedWindow:
        padding = target_length - window.sequence_length
        if padding <= 0:
            return window
        return replace(
            window,
            input_ids=window.input_ids + [0] * padding,
            attention_mask=window.attention_mask + [0] * padding,
            special_tokens_mask=window.special_tokens_mask + [1] * padding,
        )

    def _execute(self, batch: list[_QueueItem]) -> None:
        try:
            runtime_windows = [
                self._padded_window(item.window, item.bucket_length)
                if item.bucket_length
                else item.window
                for item in batch
            ]
            predictions = self.runtime.infer_windows(runtime_windows)
            if len(predictions) != len(batch):
                raise RuntimeError(
                    "inference runtime returned the wrong prediction count"
                )
            for item, prediction in zip(batch, predictions, strict=True):
                length = item.window.sequence_length
                item.future.set_result(
                    WindowPrediction(
                        bio_logits=prediction.bio_logits[:length],
                        label_logits=prediction.label_logits[:length],
                    )
                )
            with self._condition:
                self._batches += 1
                self._windows += len(batch)
                self._padded_tokens += sum(
                    (item.bucket_length or item.window.sequence_length)
                    - item.window.sequence_length
                    for item in batch
                )
        except Exception as exc:  # noqa: BLE001 - propagate backend failures to callers
            for item in batch:
                item.future.set_exception(exc)
        finally:
            with self._condition:
                self._outstanding_windows -= len(batch)
                self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                bucket = self._oldest_bucket_locked()
                while bucket is None:
                    if self._closed:
                        return
                    self._condition.wait()
                    bucket = self._oldest_bucket_locked()

                first = self._queues[bucket][0]
                deadline = first.enqueued_at + self.max_wait_seconds
                while not self._closed:
                    capacity = self._batch_capacity_locked(bucket)
                    target_capacity = self.max_windows
                    if bucket:
                        target_capacity = min(
                            target_capacity,
                            max(1, self.max_tokens // bucket),
                        )
                    if len(self._queues[bucket]) >= target_capacity or capacity < len(
                        self._queues[bucket]
                    ):
                        break
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)
                    if bucket not in self._queues:
                        break
                if bucket not in self._queues:
                    continue
                batch = self._take_batch_locked(bucket)
            self._execute(batch)

    def healthcheck(self) -> dict[str, Any]:
        health = dict(self.runtime.healthcheck())
        with self._condition:
            health["microbatching"] = {
                "enabled": True,
                "max_windows": self.max_windows,
                "max_tokens": self.max_tokens,
                "max_wait_ms": self.max_wait_seconds * 1000.0,
                "sequence_buckets": list(self.sequence_buckets),
                "outstanding_windows": self._outstanding_windows,
                "active_requests": self._active_requests,
                "batches_completed": self._batches,
                "windows_completed": self._windows,
                "padded_tokens": self._padded_tokens,
            }
        return health

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=60.0)
        if self._worker.is_alive():
            raise RuntimeError("microbatch worker did not stop within 60 seconds")
        self.runtime.close()
