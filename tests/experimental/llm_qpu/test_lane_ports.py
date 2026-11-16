# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU typed ports
"""Every chapter-03 port is registered by side-effect class and refuses explicitly."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.experimental.llm_qpu import contracts, ports
from scpn_quantum_control.experimental.llm_qpu.ports import PORTS, PortRefused, PortSpec

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHAPTER_03 = (
    ("probe_model", ("ModelProbeRequest",), "ModelDescriptor"),
    ("extract_latents", ("LatentExtractionRequest",), "LatentBatch"),
    ("fit_compressor", ("LatentBatch", "SplitManifest", "CompressionSpec"), "CompressorArtifact"),
    ("transform_latents", ("LatentBatch", "CompressorArtifact"), "CompressedLatentBatch"),
    ("build_circuit_plan", ("ReservoirSpec", "CompressedLatentBatch"), "CircuitPlanSet"),
    ("plan_measurements", ("CircuitPlanSet", "MeasurementSpec"), "PlannedExperiment"),
    ("simulate", ("CircuitPlanSet", "SimulationMode"), "RawResultSet | ExactFeatureSet"),
    ("compile_bundle", ("PlannedExperiment", "TargetSnapshot"), "CompiledBundle"),
    ("quote_bundle", ("CompiledBundle", "ResourceLimits"), "CostQuote"),
    ("submit_approved", ("CompiledBundle", "OwnerApproval"), "ProviderBatchReceipt"),
    ("retrieve_existing", ("ProviderBatchReceipt",), "RawResultSet"),
    ("validate_results", ("CompiledBundle", "RawResultSet"), "ValidationReport"),
    ("estimate_features", ("RawResultSet", "MeasurementPlan"), "FeatureBatch"),
    ("compare_models", ("ExperimentProtocol", "PredictionSet"), "EvaluationReport"),
)


def test_registry_holds_exactly_the_chapter_03_signatures_in_order() -> None:
    """The fourteen ports keep the owner pack's names, inputs and outputs."""
    assert tuple((spec.name, spec.inputs, spec.output) for spec in PORTS) == _CHAPTER_03
    assert all(
        ports.port(name) is spec for (name, _, _), spec in zip(_CHAPTER_03, PORTS, strict=True)
    )


def test_side_effect_classes_keep_submit_sole_and_offline_ports_credential_free() -> None:
    """One governed submit, one read-only retrieval, two local probes, the rest offline."""
    governed = ports.ports_for("governed_submit")
    retrieval = ports.ports_for("read_only_retrieval")
    probes = ports.ports_for("offline_model_probe")
    offline = ports.ports_for("offline_planning")
    assert [spec.name for spec in governed] == ["submit_approved"]
    assert [spec.name for spec in retrieval] == ["retrieve_existing"]
    assert [spec.name for spec in probes] == ["probe_model", "extract_latents"]
    assert len(offline) == 10 and len(governed + retrieval + probes + offline) == len(PORTS)
    assert (governed[0].access, governed[0].credentials) == ("provider_network", "provider_submit")
    assert (retrieval[0].access, retrieval[0].credentials) == ("provider_network", "provider_read")
    assert all((spec.access, spec.credentials) == ("local_runtime", "none") for spec in probes)
    assert all((spec.access, spec.credentials) == ("none", "none") for spec in offline)
    with pytest.raises(ValueError, match="^unknown port side-effect class$"):
        ports.ports_for("online_planning")
    with pytest.raises(ValueError, match="^unknown lane port: train_student$"):
        ports.port("train_student")


@pytest.mark.parametrize("spec", PORTS, ids=[spec.name for spec in PORTS])
def test_every_port_refuses_with_its_reason_and_missing_contracts(spec: PortSpec) -> None:
    """A refusal names the port, its class, its reason and every undefined contract."""
    missing = [
        name for name in (*spec.inputs, *spec.output.split(" | ")) if name not in contracts.__all__
    ]
    with pytest.raises(PortRefused) as refused:
        spec.refuse()
    message = str(refused.value)
    assert message.startswith(f"{spec.name} ({spec.side_effect}) refuses: {spec.reason}")
    assert spec.missing_contracts() == tuple(sorted(set(missing)))
    if missing:
        assert message.endswith(f"; missing contracts: {', '.join(sorted(set(missing)))}")
    else:
        assert "missing contracts" not in message


def test_fully_typed_ports_refuse_and_retrieval_never_submits() -> None:
    """The four ports whose contracts all exist refuse through their typed functions."""
    anything: Any = object()
    calls = (
        (ports.transform_latents, (anything, anything), "transform_latents (offline_planning)"),
        (ports.submit_approved, (anything, anything), "submit_approved (governed_submit)"),
        (ports.retrieve_existing, (anything,), "retrieve_existing (read_only_retrieval)"),
        (ports.estimate_features, (anything, anything), "estimate_features (offline_planning)"),
    )
    for function, arguments, prefix in calls:
        with pytest.raises(PortRefused, match=f"^{re.escape(prefix)} refuses"):
            function(*arguments)
    assert "never fall back to a submission" in ports.port("retrieve_existing").reason
    assert "never start a new experiment" in ports.port("compare_models").reason
    assert "stored target only" in ports.port("compile_bundle").reason


def test_missing_contracts_follow_the_contracts_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once a contract lands, the port stops naming it as missing."""
    probe = ports.port("probe_model")
    assert probe.missing_contracts() == ("ModelProbeRequest",)
    monkeypatch.setattr(contracts, "__all__", (*contracts.__all__, "ModelProbeRequest"))
    assert probe.missing_contracts() == ()
    with pytest.raises(PortRefused) as refused:
        probe.refuse()
    assert "missing contracts" not in str(refused.value)


def test_port_specs_refuse_a_second_submit_or_an_unnamed_port() -> None:
    """No other port may join the governed class, and submit may not leave it."""
    for arguments, reason in (
        (("compile_bundle", ("A",), "B", "governed_submit", "r"), "^submit_approved is the sole"),
        (
            ("submit_approved", ("A",), "B", "offline_planning", "r"),
            "^submit_approved is the sole",
        ),
        (("probe", ("A",), "B", "online_planning", "r"), "^unknown port side-effect class$"),
        (("probe", (), "B", "offline_planning", "r"), "^a port names its inputs, its output and"),
        (
            ("probe", ("A",), "", "offline_planning", "r"),
            "^a port names its inputs, its output and",
        ),
        (
            ("probe", ("A",), "B", "offline_planning", ""),
            "^a port names its inputs, its output and",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            PortSpec(*arguments)


def test_importing_and_refusing_ports_is_offline() -> None:
    """A fresh import and every refusal survive poisoned credentials and network calls."""
    script = """
import socket
def forbidden(*args, **kwargs):
    raise AssertionError('network access during a port refusal')
socket.socket.connect = forbidden
socket.create_connection = forbidden
from scpn_quantum_control.experimental.llm_qpu.ports import PORTS, PortRefused
for spec in PORTS:
    try:
        spec.refuse()
    except PortRefused:
        continue
    raise AssertionError(spec.name)
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
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
