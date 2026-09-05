from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from meddeid.pipeline.types import PreparedWindow, WindowPrediction
from meddeid.runtime.microbatch import MicroBatcherOverloaded, MicroBatchRuntime


def window(length: int, *, doc_index: int = 0) -> PreparedWindow:
    return PreparedWindow(
        doc_index=doc_index,
        begin=0,
        end=max(0, length - 2),
        input_ids=[1] * length,
        attention_mask=[1] * length,
        special_tokens_mask=[0] * length,
    )


class RecordingRuntime:
    def __init__(self) -> None:
        self.batches: list[list[PreparedWindow]] = []
        self.closed = False

    def infer_windows(self, windows: list[PreparedWindow]) -> list[WindowPrediction]:
        self.batches.append(windows)
        return [
            WindowPrediction(
                bio_logits=np.zeros((item.sequence_length, 3)),
                label_logits=np.zeros((item.sequence_length, 14)),
            )
            for item in windows
        ]

    def healthcheck(self):
        return {"ready": True, "backend": "recording"}

    def close(self) -> None:
        self.closed = True


def test_microbatch_runtime_coalesces_callers_and_restores_lengths() -> None:
    underlying = RecordingRuntime()
    runtime = MicroBatchRuntime(
        underlying,
        max_windows=4,
        max_tokens=32,
        max_wait_ms=100,
        sequence_buckets=(8, 16),
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(runtime.infer_windows, [window(3)])
            second = executor.submit(runtime.infer_windows, [window(7, doc_index=1)])
            assert first.result()[0].bio_logits.shape == (3, 3)
            assert second.result()[0].label_logits.shape == (7, 14)

        assert len(underlying.batches) == 1
        assert [item.sequence_length for item in underlying.batches[0]] == [8, 8]
        assert underlying.batches[0][0].attention_mask == [1, 1, 1, 0, 0, 0, 0, 0]
        assert runtime.healthcheck()["microbatching"]["padded_tokens"] == 6
    finally:
        runtime.close()
    assert underlying.closed is True


class BlockingRuntime(RecordingRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def infer_windows(self, windows: list[PreparedWindow]) -> list[WindowPrediction]:
        self.started.set()
        self.release.wait(timeout=2)
        return super().infer_windows(windows)


def test_microbatch_runtime_rejects_requests_beyond_bounded_capacity() -> None:
    underlying = BlockingRuntime()
    runtime = MicroBatchRuntime(
        underlying,
        max_windows=1,
        max_tokens=8,
        max_wait_ms=0,
        queue_max_windows=1,
        queue_max_requests=1,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            admitted = executor.submit(runtime.infer_windows, [window(3)])
            assert underlying.started.wait(timeout=1)
            with pytest.raises(MicroBatcherOverloaded):
                runtime.infer_windows([window(3, doc_index=1)])
            underlying.release.set()
            assert len(admitted.result()) == 1
    finally:
        underlying.release.set()
        runtime.close()


def test_microbatch_runtime_rejects_window_larger_than_bucket_range() -> None:
    runtime = MicroBatchRuntime(
        RecordingRuntime(),
        max_windows=2,
        max_tokens=32,
        max_wait_ms=0,
        sequence_buckets=(8, 16),
    )
    try:
        with pytest.raises(ValueError, match="largest sequence bucket"):
            runtime.infer_windows([window(17)])
    finally:
        runtime.close()


def test_unbucketed_token_limit_accounts_for_batch_padding() -> None:
    underlying = RecordingRuntime()
    runtime = MicroBatchRuntime(
        underlying,
        max_windows=4,
        max_tokens=8,
        max_wait_ms=100,
    )
    try:
        assert len(
            runtime.infer_windows([window(5), window(3, doc_index=1)])
        ) == 2
        assert [len(batch) for batch in underlying.batches] == [1, 1]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("max_windows", 0, "window and token"),
        ("max_tokens", 0, "window and token"),
        ("max_wait_ms", -1, "wait"),
        ("queue_max_windows", 0, "queue"),
        ("queue_max_requests", 0, "queue"),
    ],
)
def test_microbatch_runtime_rejects_invalid_limits(setting, value, message) -> None:
    options = {
        "max_windows": 2,
        "max_tokens": 32,
        "max_wait_ms": 1,
        "queue_max_windows": 8,
        "queue_max_requests": 4,
    }
    options[setting] = value
    with pytest.raises(ValueError, match=message):
        MicroBatchRuntime(RecordingRuntime(), **options)
