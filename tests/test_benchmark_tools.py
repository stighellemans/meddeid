import importlib.util
import json
from pathlib import Path

_BENCHMARK_PATH = Path(__file__).parents[1] / "deploy" / "benchmark_http.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_http", _BENCHMARK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCHMARK)
load_batches = _BENCHMARK.load_batches

_IMAGE_SIZE_PATH = Path(__file__).parents[1] / "deploy" / "measure_container_image.py"
_IMAGE_SIZE_SPEC = importlib.util.spec_from_file_location(
    "measure_container_image", _IMAGE_SIZE_PATH
)
assert _IMAGE_SIZE_SPEC is not None and _IMAGE_SIZE_SPEC.loader is not None
_IMAGE_SIZE = importlib.util.module_from_spec(_IMAGE_SIZE_SPEC)
_IMAGE_SIZE_SPEC.loader.exec_module(_IMAGE_SIZE)
parse_docker_size = _IMAGE_SIZE.parse_docker_size
load_budget = _IMAGE_SIZE.load_budget
size_budget_failures = _IMAGE_SIZE.size_budget_failures


def test_benchmark_loader_reads_suite_metadata_json_and_repeats_ids(tmp_path) -> None:
    fixture = tmp_path / "benchmark.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "document_id": "note-1",
                "text": "synthetic note",
                "metadata_json": json.dumps(
                    {
                        "lang": "nl-BE",
                        "patient": {"given_name": "Ada"},
                        "benchmark_robustness": {"source": "not API metadata"},
                    }
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    batches = load_batches(fixture, batch_size=2, repeat_fixture=2)

    assert [item["document_id"] for item in batches[0]] == ["note-1-r0", "note-1-r1"]
    assert all(
        item["metadata"] == {"lang": "nl-BE", "patient": {"given_name": "Ada"}}
        for item in batches[0]
    )


def test_image_measurement_parses_docker_history_sizes() -> None:
    assert parse_docker_size("0B") == 0
    assert parse_docker_size("503MB") == 503_000_000
    assert parse_docker_size("6.75GB") == 6_750_000_000


def test_checked_image_size_budgets_cover_each_published_runtime() -> None:
    budget_file = Path(__file__).parents[1] / "deploy" / "image-size-budgets.json"
    for key in ("cpu", "pytorch-cuda", "tensorrt-server", "tensorrt-gateway"):
        budget = load_budget(budget_file, key)
        report = {
            "portable_save_gzip_bytes": budget["max_portable_save_gzip_bytes"],
            "content_size_bytes": budget["max_content_size_bytes"],
        }
        assert size_budget_failures(report, budget) == []

        report["portable_save_gzip_bytes"] += 1
        assert size_budget_failures(report, budget)
