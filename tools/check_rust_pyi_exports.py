# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control - Rust extension typing contract check
"""Check that the PyO3 export list and local typing contract agree."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUST_MODULE = ROOT / "scpn_quantum_engine" / "src" / "lib.rs"
PYI_MODULE = ROOT / "src" / "scpn_quantum_engine.pyi"

WRAP_RE = re.compile(r"wrap_pyfunction!\((?P<path>[a-zA-Z0-9_:]+),\s*m\)")


def rust_exports() -> set[str]:
    """Return every function exported from the PyO3 module initializer."""
    text = RUST_MODULE.read_text(encoding="utf-8")
    exports: set[str] = set()
    for match in WRAP_RE.finditer(text):
        exports.add(match.group("path").rsplit("::", 1)[-1])
    return exports


def pyi_exports() -> set[str]:
    """Return every public function declared in the local .pyi contract."""
    tree = ast.parse(PYI_MODULE.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }


def main() -> int:
    """Compare Rust extension exports with Python type-stub exports."""
    rust = rust_exports()
    pyi = pyi_exports()
    missing = sorted(rust - pyi)
    stale = sorted(pyi - rust)
    if missing or stale:
        print("Rust extension typing contract drift:")
        if missing:
            print("  missing from src/scpn_quantum_engine.pyi:")
            for name in missing:
                print(f"    - {name}")
        if stale:
            print("  declared in src/scpn_quantum_engine.pyi but not exported by Rust:")
            for name in stale:
                print(f"    - {name}")
        return 1
    print(f"Rust extension typing contract OK ({len(rust)} exports)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
