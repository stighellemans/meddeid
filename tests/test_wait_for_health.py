import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wait_health', ROOT / 'deploy/wait_for_health.py')
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)


@pytest.mark.parametrize('status,restarts', [('exited', 0), ('restarting', 1), ('running', 2)])
def test_dead_container_fails_without_waiting_for_timeout(monkeypatch, tmp_path, status, restarts):
    output = tmp_path / 'startup.json'
    monkeypatch.setattr(sys, 'argv', ['wait', '--container', 'candidate', '--started-at-ns', '0',
                                    '--output', str(output)])
    monkeypatch.setattr(health, 'time_ns', lambda: 1)
    monkeypatch.setattr(health.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0,
        stdout=json.dumps([{'State': {'Status': status}, 'RestartCount': restarts}])))
    monkeypatch.setattr(health, 'urlopen', lambda *a, **k: pytest.fail('must not poll dead service'))
    with pytest.raises(RuntimeError, match='stopped or restarted'):
        health.main()
    report = json.loads(output.read_text())
    assert not report['passed'] and report['attempts'] == 1


def test_timeout_preserves_error_evidence(monkeypatch, tmp_path):
    output = tmp_path / 'startup.json'
    monkeypatch.setattr(sys, 'argv', ['wait', '--started-at-ns', '0', '--timeout-seconds', '1',
                                    '--output', str(output)])
    monkeypatch.setattr(health, 'time_ns', lambda: 2_000_000_000)
    with pytest.raises(RuntimeError, match='not attempted'):
        health.main()
    assert json.loads(output.read_text())['passed'] is False
