#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- stable core contract fixtures exporter
"""Export deterministic stable-core contract fixtures."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_comparator_module() -> ModuleType:
    """Load comparator helpers without requiring ``scripts`` as import package."""

    spec = importlib.util.spec_from_file_location(
        "_stable_core_contract_fixture_comparator",
        Path(__file__).resolve().parent / "compare_stable_core_contract_fixtures.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load stable-core contract fixture comparator module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = REPO_ROOT / "data" / "stable_core" / "stable_core_contract_fixtures.json"
DEFAULT_DOC = REPO_ROOT / "docs" / "stable_core_contract_fixtures.md"


def main() -> int:
    """Export stable-core contract fixture artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()

    comparator = _load_comparator_module()
    digests = comparator.write_stable_core_contract_fixtures(
        json_path=args.json_path,
        markdown_path=args.doc_path,
    )
    print(
        "wrote {path} sha256={digest}".format(
            path=args.json_path.relative_to(REPO_ROOT),
            digest=digests["json_sha256"],
        )
    )
    print(
        "wrote {path} sha256={digest}".format(
            path=args.doc_path.relative_to(REPO_ROOT),
            digest=digests["markdown_sha256"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
