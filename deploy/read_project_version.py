from __future__ import annotations

import re
from pathlib import Path


def main() -> None:
    project = (
        Path("pyproject.toml").read_text(encoding="utf-8").split("[project]", 1)[1]
    )
    match = re.search(r'^version\s*=\s*"([^"]+)"', project, re.MULTILINE)
    if match is None:
        raise SystemExit("pyproject.toml is missing [project].version")
    print(match.group(1))


if __name__ == "__main__":
    main()
