from __future__ import annotations

from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start-local.sh"


def run_launcher(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sh", str(SCRIPT), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_local_launcher_help_describes_selection() -> None:
    result = run_launcher("--help")

    assert result.returncode == 0
    assert "--model HUB_ID" in result.stdout
    assert "--language-profile PROFILE" in result.stdout
    assert "--build" in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        (),
        ("--model", "example/model"),
        ("--language-profile", "nl-BE"),
    ],
)
def test_local_launcher_requires_model_and_profile_before_docker(
    arguments: tuple[str, ...],
) -> None:
    result = run_launcher(*arguments)

    assert result.returncode == 2
    assert "Select both --model and --language-profile" in result.stderr
