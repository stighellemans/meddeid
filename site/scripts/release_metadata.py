"""Expose the package version to the documentation templates."""

from pathlib import Path
import subprocess
import sys


def on_config(config):
    project_root = Path(config.config_file_path).resolve().parent.parent
    config.extra["package_version"] = subprocess.check_output(
        [sys.executable, "deploy/read_project_version.py"], cwd=project_root, text=True
    ).strip()
