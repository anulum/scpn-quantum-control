# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — sealed qualification receipt vectors for tests
"""Build unassessed qualification contract vectors over real repository files.

Each vector copies byte-identical production source, test cohort and CI files
into a temporary checkout and seals a receipt whose engineering axes are
labelled contract inputs, never measured scientific results.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "src/scpn_quantum_control/resource_budget_gate.py"
COHORT = "tests/test_resource_budget_gate.py"
WORKFLOW = ".github/workflows/ci-assurance-policy.yml"
POLICY = "tools/ci_workflow_policy.json"
COORDINATOR = json.loads((ROOT / POLICY).read_text(encoding="utf-8"))["coordinator"]


def build_receipt(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    """Copy the actual source tree boundary and freeze an unassessed vector."""
    checkout = tmp_path / "checkout"
    paths = (SOURCE, COHORT, WORKFLOW, POLICY, COORDINATOR)
    for name in paths:
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(ROOT / name, target)
    payload: dict[str, object] = {
        "schema": "domain_qualification_receipt.v1",
        "domain_id": "resource_admission_contract_vector",
        "category": "docs_api_maintainability",
        "source_hashes": {
            name: hashlib.sha256((checkout / name).read_bytes()).hexdigest() for name in paths
        },
        "runtime": {
            "module": "sys",
            "distribution": "python",
            "version": platform.python_version(),
        },
        "axes": {
            "forward": "passed",
            "derivative": "passed",
            "composition": "passed",
            "backend": "passed",
        },
        "scientific_status": "unassessed",
        "claim_class": "simulation",
        "ci_job": "resource-budget-gate-quality",
        "test_cohort": [COHORT],
    }
    return checkout, tmp_path / "receipt.json", payload


def seal(path: Path, payload: dict[str, object]) -> str:
    """Seal the independent JSON vector as exact bytes before projection."""
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()
