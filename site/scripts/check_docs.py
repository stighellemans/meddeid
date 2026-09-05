#!/usr/bin/env python3
"""Fail when local Markdown links or basic page structure are invalid."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def markdown_links(path: Path) -> list[tuple[int, str]]:
    links: list[tuple[int, str]] = []
    in_fence = False
    fence = ""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        marker = FENCE_RE.match(line)
        if marker:
            current = marker.group(1)
            if not in_fence:
                in_fence = True
                fence = current
            elif current == fence:
                in_fence = False
            continue
        if in_fence:
            continue
        links.extend((number, match.group(1)) for match in LINK_RE.finditer(line))
    return links


def h1_count(path: Path) -> int:
    count = 0
    in_fence = False
    fence = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        marker = FENCE_RE.match(line)
        if marker:
            current = marker.group(1)
            if not in_fence:
                in_fence = True
                fence = current
            elif current == fence:
                in_fence = False
            continue
        if not in_fence and re.match(r"^# [^#]", line):
            count += 1
    return count


def main() -> int:
    errors: list[str] = []
    pages = sorted(DOCS.rglob("*.md"))
    if not pages:
        errors.append("docs/: no Markdown pages found")

    for page in pages:
        relative = page.relative_to(ROOT)
        page_h1_count = h1_count(page)
        if page_h1_count != 1:
            errors.append(f"{relative}: expected one H1, found {page_h1_count}")

        for line, raw_target in markdown_links(page):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path_part = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not path_part:
                continue
            resolved = (page.parent / path_part).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                errors.append(f"{relative}:{line}: link escapes repository: {target}")
                continue
            if not resolved.exists():
                errors.append(f"{relative}:{line}: missing local target: {target}")

    if errors:
        print("Documentation checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Checked {len(pages)} documentation pages and their local links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
