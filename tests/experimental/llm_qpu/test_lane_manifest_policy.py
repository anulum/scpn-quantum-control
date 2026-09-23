# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU manifest tests
"""Exercise the opt-in lane inventory and legacy-registry boundary."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from scpn_quantum_control.analysis.research_lane_registry import (
    ResearchLaneClaimStatus,
    ResearchLaneDiffHook,
    ResearchLaneMaturity,
    ResearchLaneRecord,
)
from scpn_quantum_control.experimental.llm_qpu.manifest import (
    LANE_MANIFEST,
    LaneManifest,
    assert_lane_inventory,
    assert_worker_inventory,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENTAL_ROOT = _REPO_ROOT / "src/scpn_quantum_control/experimental"
_WORKER_ROOT = _REPO_ROOT / "experimental_workers/llm_qpu"


def test_manifest_import_is_offline_and_does_not_authorize_submit() -> None:
    """A fresh import survives poisoned provider credentials and network calls."""
    script = """
import socket
def forbidden(*args, **kwargs):
    raise AssertionError('network access during experimental import')
socket.socket.connect = forbidden
socket.create_connection = forbidden
from scpn_quantum_control.experimental.llm_qpu.manifest import LANE_MANIFEST
wire = LANE_MANIFEST.to_wire()
assert wire['hardware_submission_enabled'] is False
assert wire['claim_promotion_enabled'] is False
assert all(value == 'not_implemented' for value in wire['kernel_statuses'].values())
"""
    env = dict(os.environ)
    env.update({"IQM_TOKEN": "poison", "IBM_QUANTUM_TOKEN": "poison"})
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(_REPO_ROOT / "src"), env.get("PYTHONPATH", "")) if path
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_manifest_inventory_refuses_new_unlisted_module(tmp_path: Path) -> None:
    """The live package inventory passes and a copied unreviewed module fails."""
    assert assert_lane_inventory() == tuple(sorted(LANE_MANIFEST.module_inventory))
    copied = tmp_path / "experimental"
    shutil.copytree(_EXPERIMENTAL_ROOT, copied, ignore=shutil.ignore_patterns("__pycache__"))
    (copied / "llm_qpu/unreviewed.py").write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="inventory drift"):
        assert_lane_inventory(copied)
    assert assert_worker_inventory() == tuple(sorted(LANE_MANIFEST.worker_inventory))
    copied_worker = tmp_path / "workers"
    shutil.copytree(_WORKER_ROOT, copied_worker, ignore=shutil.ignore_patterns("__pycache__"))
    (copied_worker / "protocol/unreviewed.py").write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="inventory drift"):
        assert_worker_inventory(copied_worker)


def test_legacy_research_registry_rejects_experimental_module() -> None:
    """The existing analysis/gauge record validator keeps its old namespace."""
    with pytest.raises(ValueError, match="analysis or gauge"):
        ResearchLaneRecord(
            module="scpn_quantum_control.experimental.llm_qpu.manifest",
            summary="Experimental LLM-QPU manifest",
            maturity=ResearchLaneMaturity.RESEARCH,
            diff_hook=ResearchLaneDiffHook.NONE,
            claim_status=ResearchLaneClaimStatus.RESEARCH_ONLY,
        )


def test_root_package_does_not_reexport_experimental_lane() -> None:
    """The stable root source never imports or exports the opt-in namespace."""
    source = (_REPO_ROOT / "src/scpn_quantum_control/__init__.py").read_text()
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module == "experimental" or node.module.startswith("experimental."))
        for node in ast.walk(tree)
    )
    assert "LANE_MANIFEST" not in source
    with pytest.raises(ValueError, match="cannot authorize"):
        LaneManifest(
            schema=LANE_MANIFEST.schema,
            lane_id=LANE_MANIFEST.lane_id,
            namespace=LANE_MANIFEST.namespace,
            kernel_statuses=LANE_MANIFEST.kernel_statuses,
            module_inventory=LANE_MANIFEST.module_inventory,
            worker_inventory=LANE_MANIFEST.worker_inventory,
            write_roots=LANE_MANIFEST.write_roots,
            hardware_submission_enabled=True,
            claim_promotion_enabled=False,
            non_claims=LANE_MANIFEST.non_claims,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "unknown", "unknown experimental lane"),
        ("lane_id", "wrong", "unknown experimental lane"),
        ("namespace", "scpn_quantum_control", "namespace mismatch"),
        ("kernel_statuses", [], "immutable pairs"),
        ("kernel_statuses", (("wrong", "not_implemented"),), "kernel inventory"),
        (
            "kernel_statuses",
            (("xy_static_digital_v1", "ready"), *LANE_MANIFEST.kernel_statuses[1:]),
            "readiness requires",
        ),
        ("module_inventory", (), "module inventory"),
        ("worker_inventory", (), "worker inventory"),
        ("write_roots", (), "write roots"),
        ("hardware_submission_enabled", 1, "must be boolean"),
        ("claim_promotion_enabled", True, "cannot authorize"),
        ("non_claims", (), "non-claims must be non-empty"),
        ("non_claims", (" ",), "non-claims must be non-empty"),
    ],
)
def test_manifest_refuses_unreviewed_execution_or_scope(
    field: str, value: object, message: str
) -> None:
    """Malformed scope and execution authority fail before any provider path."""
    with pytest.raises(ValueError, match=message):
        replace(LANE_MANIFEST, **{field: value})


def test_manifest_wire_is_detached_from_approved_scope() -> None:
    """Editing a serialized description cannot change the frozen manifest."""
    wire = LANE_MANIFEST.to_wire()
    assert wire["schema"] == LANE_MANIFEST.schema
    assert wire["status"] == "experimental"
    assert wire["hardware_submission_enabled"] is False
    kernels = wire["kernel_statuses"]
    roots = wire["write_roots"]
    assert isinstance(kernels, dict)
    assert isinstance(roots, list)
    kernels["xy_static_digital_v1"] = "ready"
    roots.append("data/unreviewed")
    assert LANE_MANIFEST.kernel_statuses[0][1] == "not_implemented"
    assert "data/unreviewed" not in LANE_MANIFEST.write_roots


def test_manifest_inventory_refuses_missing_declared_modules(tmp_path: Path) -> None:
    """An incomplete deployment cannot claim the expected package or worker."""
    copied = tmp_path / "experimental"
    shutil.copytree(_EXPERIMENTAL_ROOT, copied, ignore=shutil.ignore_patterns("__pycache__"))
    (copied / "llm_qpu/manifest.py").unlink()
    with pytest.raises(ValueError, match="inventory drift"):
        assert_lane_inventory(copied)
    copied_worker = tmp_path / "workers"
    shutil.copytree(_WORKER_ROOT, copied_worker, ignore=shutil.ignore_patterns("__pycache__"))
    (copied_worker / "protocol/worker.py").unlink()
    with pytest.raises(ValueError, match="inventory drift"):
        assert_worker_inventory(copied_worker)


def test_standalone_worker_runs_without_parent_import_and_refuses_compute(
    tmp_path: Path,
) -> None:
    """The real worker executes with only stdlib and no SCPN import path."""
    worker = _WORKER_ROOT / "protocol/worker.py"
    env = {
        "PYTHONPATH": str(tmp_path),
        "IQM_TOKEN": "poison",
        "IBM_QUANTUM_TOKEN": "poison",
    }
    describe = subprocess.run(
        [sys.executable, "-S", str(worker)],
        input=b'{"op":"describe"}',
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert describe.returncode == 0, describe.stderr.decode(errors="replace")
    response = json.loads(describe.stdout)
    assert response["status"] == "experimental_no_compute"
    assert response["hardware_submission_enabled"] is False
    refused = subprocess.run(
        [sys.executable, "-S", str(worker)],
        input=b'{"op":"submit"}',
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert refused.returncode == 2
    assert json.loads(refused.stdout)["status"] == "refused"


@pytest.mark.parametrize(
    ("request", "reason"),
    [
        (b"not json", "invalid JSON"),
        (b"[]", "unsupported operation"),
        (b'{"op":"describe","submit":true}', "unsupported operation"),
        (b" " * 65_537, "request too large"),
    ],
)
def test_standalone_worker_refuses_malformed_or_oversized_requests(
    tmp_path: Path, request: bytes, reason: str
) -> None:
    """The real isolated process rejects ambiguous and unbounded requests."""
    result = subprocess.run(
        [sys.executable, "-S", str(_WORKER_ROOT / "protocol/worker.py")],
        input=request,
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path)},
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["reason"] == reason
