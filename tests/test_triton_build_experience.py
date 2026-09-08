from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from meddeid.triton_artifact import load_catalog
from meddeid import triton_artifact
from meddeid.triton_revision import resolve_revision
from meddeid.triton_gateway import pin_gateway_revision

ROOT = Path(__file__).resolve().parents[1]


def test_missing_revision_resolves_main_and_continues(monkeypatch):
    calls = []

    def model_info(self, model, **kwargs):
        calls.append((model, kwargs))
        return SimpleNamespace(sha="a" * 40)

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)
    assert resolve_revision("owner/model", "") == "a" * 40
    assert calls == [
        ("owner/model", {"revision": "main", "expand": ["sha"], "timeout": 10})
    ]


def test_pinned_and_local_revisions_never_query_hub(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("should not access the Hub")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", unexpected)
    assert resolve_revision("owner/model", "a" * 40) == "a" * 40
    assert resolve_revision("/input-model", "local-v1", local_model=True) == "local-v1"
    assert resolve_revision("/input-model", "", local_model=True) == ""


def test_offline_revision_hint_does_not_expose_network_error(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("secret credential in network error")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", failed)
    with pytest.raises(ValueError) as error:
        resolve_revision("owner/model", "")
    assert "secret credential" not in str(error.value)
    assert "approved commit" in str(error.value)


@pytest.mark.parametrize("revision", ["", "main", "v1", "a" * 40])
def test_gateway_uses_plan_commit_without_a_second_hub_lookup(revision, monkeypatch):
    def unexpected(*a, **kw):
        pytest.fail("gateway must not resolve main independently")
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", unexpected)
    env = {"MEDDEID_MODEL": "owner/model", "MEDDEID_REVISION": revision}
    pin_gateway_revision(env, {"model": {"id": "owner/model", "revision": "a" * 40}})
    assert env["MEDDEID_REVISION"] == "a" * 40


def test_gateway_rejects_different_pinned_commit():
    with pytest.raises(ValueError, match="revision does not match"):
        pin_gateway_revision(
            {"MEDDEID_MODEL": "owner/model", "MEDDEID_REVISION": "b" * 40},
            {"model": {"id": "owner/model", "revision": "a" * 40}},
        )


@pytest.mark.parametrize("matches", [True, False])
def test_local_gateway_uses_bundle_without_hub_revision(monkeypatch, matches):
    from meddeid import bundle

    monkeypatch.setattr(bundle, "load_model_bundle", lambda *a, **kw:
                        SimpleNamespace(contract_hash=lambda: "expected" if matches else "other"))
    env = {"MEDDEID_MODEL": "/input-model", "MEDDEID_REVISION": "local-v1",
           "MEDDEID_LOCAL_BUNDLE": "true"}
    manifest = {"model": {"bundle_sha256": "expected"}}
    if matches:
        pin_gateway_revision(env, manifest)
        assert env["MEDDEID_REVISION"] == ""
    else:
        with pytest.raises(ValueError, match="bundle does not match"):
            pin_gateway_revision(env, manifest)


def test_fetch_cli_passes_automatically_resolved_revision(monkeypatch, tmp_path):
    monkeypatch.delenv("MEDDEID_LOCAL_BUNDLE", raising=False)
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info",
                        lambda *a, **kw: SimpleNamespace(sha="a" * 40))
    monkeypatch.setattr(triton_artifact, "detect_nvidia_gpu", lambda: None)
    observed = {}

    def prepare(catalog, **kwargs):
        observed.update(kwargs)
        return "local"

    monkeypatch.setattr(triton_artifact, "prepare_plan", prepare)
    monkeypatch.setattr(sys, "argv", ["plan", "fetch", "--catalog",
        str(ROOT / "deploy/triton/release.json"), "--model", "owner/model",
        "--language-profile", "nl-BE", "--output", str(tmp_path)])
    triton_artifact.main()
    assert observed["revision"] == "a" * 40
    assert observed["model"] == "owner/model"



def test_builder_is_noninteractive_and_has_no_contribution_step():
    script = (ROOT / "deploy/build_triton_local.sh").read_text()
    compose = (ROOT / "compose.triton.build.yaml").read_text()
    assert "contribut" not in script
    assert "stdin_open" not in compose
    assert "tty:" not in compose


def test_builder_installs_virtual_environment_support():
    dockerfile = (ROOT / "deploy/triton-builder.Dockerfile").read_text()
    assert "python3-venv" in dockerfile
    assert dockerfile.index("python3-venv") < dockerfile.index("python -m venv")
