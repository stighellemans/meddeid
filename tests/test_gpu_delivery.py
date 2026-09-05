import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
render_config = runpy.run_path(ROOT / "deploy/render_triton_config.py")["render_config"]


def test_image_build_metadata_reads_the_supported_project_version() -> None:
    output = subprocess.check_output(
        [sys.executable, "deploy/read_project_version.py"],
        cwd=ROOT,
        text=True,
    )

    assert output.strip() == "0.3.0"


def test_cpu_workflow_passes_the_project_version_to_both_builds() -> None:
    workflow = (ROOT / ".github/workflows/container.yml").read_text()

    assert 'output.write(f"PACKAGE_VERSION={version}\\n")' in workflow
    assert workflow.count("MEDDEID_VERSION=${{ env.PACKAGE_VERSION }}") == 2
    assert "python deploy/read_project_version.py" in workflow


def test_pytorch_cuda_release_pins_stay_aligned() -> None:
    versions = (ROOT / "deploy/pytorch-cuda/versions.env").read_text()
    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "compose.cuda.yaml").read_text()

    assert "MEDDEID_PYTORCH_VERSION=2.13.0" in versions
    assert "MEDDEID_PYTORCH_CUDA_VERSION=12.9" in versions
    assert (
        "MEDDEID_PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu129" in versions
    )
    assert "ARG TORCH_VERSION=2.13.0" in dockerfile
    assert "io.meddeid.accelerator" in dockerfile
    assert 'org.opencontainers.image.version="${MEDDEID_VERSION}"' in dockerfile
    assert "MEDDEID_DEVICE: cuda" in compose
    assert "MEDDEID_TORCH_PRECISION: ${MEDDEID_TORCH_PRECISION:-fp16}" in compose
    assert (
        compose.count("MEDDEID_TORCH_COMPILE_MODE: ${MEDDEID_TORCH_COMPILE_MODE:-off}")
        == 1
    )
    assert "ghcr.io/stighellemans/meddeid-api:0.3.0-cuda12.9" in compose
    assert "site-packages/triton" in dockerfile
    assert "site-packages/torch/include" in dockerfile
    assert "! -name torch_shm_manager -delete" in dockerfile
    assert "libbackend_with_compiler.so" in dockerfile
    assert "MEDDEID_MAX_CONCURRENT_REQUESTS=auto" in dockerfile
    assert dockerfile.index("LABEL org.opencontainers.image.title") > dockerfile.index(
        "COPY --from=builder /opt/venv"
    )


def test_gpu_compose_requests_one_selected_nvidia_device() -> None:
    compose = (ROOT / "compose.cuda.yaml").read_text()

    assert 'device_ids: ["${MEDDEID_GPU_DEVICE_ID:-0}"]' in compose
    assert "capabilities: [gpu]" in compose
    assert "NVIDIA_DRIVER_CAPABILITIES: compute,utility" in compose


def test_gpu_validation_exercises_cuda_and_the_http_api() -> None:
    checker = (ROOT / "deploy/check_pytorch_cuda.py").read_text()
    validator = (ROOT / "deploy/validate_pytorch_cuda_image.sh").read_text()

    assert "torch.cuda.is_available()" in checker
    assert "left @ right" in checker
    assert '("tensorrt", "tritonclient", "onnxruntime")' in checker
    assert 'find_spec("triton") is not None' in checker
    assert "scripts/container_smoke.py:/smoke.py:ro" in validator
    assert "--read-only" in validator


def test_triton_tooling_avoids_local_package_shadowing() -> None:
    builder = (ROOT / "deploy/build_triton_repository.sh").read_text()
    exporter = (ROOT / "deploy/export_onnx.py").read_text()
    renderer = (ROOT / "deploy/render_triton_config.py").read_text()
    preflight = (ROOT / "deploy/preflight_triton_host.sh").read_text()
    versions = (ROOT / "deploy/triton/versions.env").read_text()

    assert "MEDDEID_TRT_PRECISION=fp16" in versions
    assert "MEDDEID_TRT_OUTPUT_PRECISION=fp32" in versions
    assert "MEDDEID_TRITON_THROUGHPUT_DYNAMIC_BATCHING=false" in versions
    assert "python -m deploy.export_onnx" in builder
    assert '--precision "${MEDDEID_TRT_PRECISION}"' in builder
    assert '--output-precision "${MEDDEID_TRT_OUTPUT_PRECISION}"' in builder
    assert "model.half()" in exporter
    assert "onnx.checker.check_model(exported)" in exporter
    assert "output.bio_logits.float()" in exporter
    assert 'args.output_precision == "fp16" else "TYPE_FP32"' in renderer
    assert "--fp16" not in builder
    assert "--skipInference" in builder
    assert "trtexec --help >/dev/null" in preflight
    assert "trtexec --version" not in preflight


def test_triton_scheduler_can_enable_dynamic_batching_explicitly() -> None:
    common = {
        "model_name": "meddeid",
        "model_version": "1",
        "max_batch_size": 64,
        "instance_count": 1,
        "bio_labels": 3,
        "entity_labels": 17,
        "output_type": "TYPE_FP32",
    }

    latency = render_config(**common, queue_delay_microseconds=None)
    request_local_throughput = render_config(
        **common,
        queue_delay_microseconds=None,
    )
    throughput = render_config(**common, queue_delay_microseconds=1000)

    assert "dynamic_batching" not in latency
    assert request_local_throughput == latency
    assert "dynamic_batching" in throughput
    assert "max_queue_delay_microseconds: 1000" in throughput
    assert "preferred_batch_size" not in throughput


def test_triton_runtime_uses_current_boolean_flag_syntax() -> None:
    dockerfile = (ROOT / "deploy/triton-model.Dockerfile").read_text()
    runtime_dockerfile = (ROOT / "deploy/triton-runtime.Dockerfile").read_text()
    slim_builder = (ROOT / "deploy/build_slim_triton_runtime.sh").read_text()

    assert '"--disable-auto-complete-config"' in dockerfile
    assert 'org.opencontainers.image.version="${MEDDEID_VERSION}"' in dockerfile
    assert 'io.meddeid.model-revision="${MODEL_REVISION}"' in dockerfile
    assert 'org.opencontainers.image.revision="${VCS_REF}"' in dockerfile
    assert dockerfile.index("LABEL org.opencontainers.image.title") > dockerfile.index(
        "COPY --from=model_repository"
    )
    assert '"--disable-auto-complete-config=true"' not in dockerfile
    assert "--backend tensorrt" in slim_builder
    assert "MEDDEID_TRITON_MIN_IMAGE" in slim_builder
    assert "MEDDEID_TRITON_COMPOSE_REVISION" in slim_builder
    assert "MEDDEID_CUDA_BASE_IMAGE" in slim_builder
    assert "import distro, requests" in slim_builder
    assert "libnvinfer_builder_resource" in runtime_dockerfile
    assert "/opt/tritonserver/backends/tensorrt" in runtime_dockerfile
    assert "libnvinfer_plugin.so.11.1.0" in runtime_dockerfile
    assert "ln -s libnvinfer.so.11.1.0" in runtime_dockerfile
    assert "COPY --from=composed /opt/tritonserver/include" not in runtime_dockerfile
    assert "apt-get upgrade" not in runtime_dockerfile
    assert "apt-get upgrade" not in dockerfile


def test_triton_image_packager_validates_nested_scheduler_configs() -> None:
    builder = (ROOT / "deploy/build_triton_image.sh").read_text()

    assert "def iter_artifacts" in builder
    assert 'iter_artifacts(payload["artifacts"])' in builder


def test_triton_gateway_is_weight_free_and_does_not_install_torch() -> None:
    dockerfile = (ROOT / "deploy/triton-gateway.Dockerfile").read_text()
    compose = (ROOT / "compose.triton.yaml").read_text()

    assert "prepare_triton_gateway_model.py" in dockerfile
    assert "('torch', 'tensorrt', 'onnxruntime')" in dockerfile
    assert "tritonclient[http]" in dockerfile
    assert 'org.opencontainers.image.version="${MEDDEID_VERSION}"' in dockerfile
    assert dockerfile.index("LABEL org.opencontainers.image.title") > dockerfile.index(
        "COPY --from=builder /opt/venv"
    )
    assert "meddeid-triton-gateway:0.3.0" in compose
    assert "MEDDEID_WORKERS=4" in dockerfile
    assert "MEDDEID_TRITON_TRANSPORT=binary" in dockerfile
    assert "MEDDEID_WINDOW_BATCH_SIZE=64" in dockerfile
    assert "--model-config-name=${MEDDEID_SERVING_PROFILE:-latency}" in compose
    assert "MEDDEID_MICROBATCH_ENABLED=auto" in dockerfile


def test_cpu_image_gate_rejects_gpu_and_alternate_frameworks() -> None:
    workflow = (ROOT / ".github/workflows/container.yml").read_text()

    assert "torch.version.cuda is None" in workflow
    assert "('tensorrt', 'tritonclient', 'onnxruntime', 'triton')" in workflow
    assert "--budget-key cpu" in workflow


def test_every_published_runtime_has_an_enforced_size_budget() -> None:
    cuda_workflow = (ROOT / ".github/workflows/pytorch-cuda.yml").read_text()
    triton_workflow = (ROOT / ".github/workflows/triton-gpu-validation.yml").read_text()

    assert "--budget-key pytorch-cuda" in cuda_workflow
    for key in ("tensorrt-server", "tensorrt-gateway", "cpu", "pytorch-cuda"):
        assert f"--budget-key {key}" in triton_workflow


def test_azure_runner_bootstrap_is_ephemeral_and_integrity_pinned() -> None:
    bootstrap = (ROOT / "deploy/azure/bootstrap_t4_runner_host.sh").read_text()
    create_config = (ROOT / "deploy/azure/create_jit_config.sh").read_text()
    run_runner = (ROOT / "deploy/azure/run_ephemeral_runner.sh").read_text()

    assert "runner_version=2.337.0" in bootstrap
    assert (
        "runner_sha256=70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613"
        in bootstrap
    )
    assert "sha256sum --check" in bootstrap
    assert "--gpus all" in bootstrap
    assert "generate-jitconfig" in create_config
    assert "labels[]=t4-sm75" in create_config
    assert "umask 077" in create_config
    assert ".meddeid-dedicated-release-runner" in bootstrap
    assert ".meddeid-dedicated-release-runner" in run_runner
    assert "docker system prune --all --force --volumes" in run_runner
    assert 'exec ./run.sh --jitconfig "${encoded_config}"' in run_runner
    assert "shred --remove" in run_runner


def test_published_gpu_templates_expose_one_performance_choice() -> None:
    expected_common = {
        "MEDDEID_API_KEY",
        "MEDDEID_SERVING_PROFILE",
        "MEDDEID_BIND_ADDRESS",
        "MEDDEID_PORT",
    }
    expected = {
        ".env.cuda.example": expected_common | {"MEDDEID_PYTORCH_CUDA_IMAGE"},
        ".env.triton.example": expected_common
        | {"MEDDEID_TRITON_IMAGE", "MEDDEID_API_IMAGE"},
    }
    for filename, expected_keys in expected.items():
        configured = {
            line.split("=", 1)[0]
            for line in (ROOT / filename).read_text().splitlines()
            if line and not line.startswith("#")
        }
        assert configured == expected_keys

    cuda = (ROOT / "compose.cuda.yaml").read_text()
    triton = (ROOT / "compose.triton.yaml").read_text()
    for compose in (cuda, triton):
        assert "MEDDEID_SERVING_PROFILE: ${MEDDEID_SERVING_PROFILE:-latency}" in compose
        assert 'MEDDEID_REQUIRE_API_KEY: "true"' in compose
        assert "MEDDEID_ALLOWED_MODELS: /opt/meddeid-model" in compose
    assert "MEDDEID_MICROBATCH_ENABLED:" not in cuda
    assert "MEDDEID_MICROBATCH_ENABLED:" not in triton


def test_triton_gate_builds_its_reference_from_the_same_checkout() -> None:
    workflow = (ROOT / ".github/workflows/triton-gpu-validation.yml").read_text()

    assert (
        'reference_candidate="meddeid-api:${image_version}-cpu-reference-ci"'
        in workflow
    )
    assert 'docker build --tag "${CANDIDATE_REFERENCE_IMAGE}" .' in workflow
    assert (
        './deploy/build_triton_gateway_image.sh "${CANDIDATE_GATEWAY_IMAGE}"'
        in workflow
    )
    assert "'.[dev]' distro requests onnx onnxscript" in workflow
    assert "MEDDEID_API_IMAGE: ghcr.io/stighellemans/meddeid-api:0.3.0" not in workflow
