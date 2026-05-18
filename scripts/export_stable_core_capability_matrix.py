#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- stable core capability matrix exporter
"""Export deterministic stable core backend capability artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from scpn_quantum_control.stable_core import write_stable_core_capability_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = REPO_ROOT / "data" / "stable_core" / "backend_capability_matrix.json"
DEFAULT_DOC = REPO_ROOT / "docs" / "stable_core_backend_capability_matrix.md"


def main() -> int:
    """Export backend capability matrix artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()
    digests = write_stable_core_capability_artifacts(
        json_path=args.json_path,
        doc_path=args.doc_path,
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
            digest=digests["doc_sha256"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
