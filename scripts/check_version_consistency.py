#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Version Consistency Checker
"""Pre-commit hook: verify version strings match across all carrier files.

Canonical source: pyproject.toml `version = "X.Y.Z"`
Must match: __init__.py, CITATION.cff, .zenodo.json, README.md badge.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "scpn_quantum_control" / "__init__.py"
CITATION = ROOT / "CITATION.cff"
ZENODO = ROOT / ".zenodo.json"
README = ROOT / "README.md"

PATTERNS: dict[Path, re.Pattern[str]] = {
    PYPROJECT: re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE),
    INIT: re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE),
    CITATION: re.compile(r'^version:\s*"([^"]+)"', re.MULTILINE),
    ZENODO: re.compile(r'"version":\s*"([^"]+)"'),
    README: re.compile(r"version-([0-9]+\.[0-9]+\.[0-9]+)-"),
}


def main() -> int:
    canonical = None
    errors: list[str] = []

    for path, pattern in PATTERNS.items():
        if not path.exists():
            errors.append(f"  {path.relative_to(ROOT)}: file not found")
            continue
        match = pattern.search(path.read_text(encoding="utf-8"))
        if not match:
            errors.append(f"  {path.relative_to(ROOT)}: version pattern not found")
            continue
        version = match.group(1)
        if canonical is None:
            canonical = version
        elif version != canonical:
            errors.append(f"  {path.relative_to(ROOT)}: {version} (expected {canonical})")

    if errors:
        print(f"Version mismatch (canonical: {canonical}):")
        print("\n".join(errors))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
