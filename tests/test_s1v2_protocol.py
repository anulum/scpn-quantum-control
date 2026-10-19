# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the S1v2 protocol, journal and submitter gates
"""Real tests of the frozen S1v2 circuits, analysis, journal and submission gates."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import IfElseOp
from qiskit_aer import AerSimulator

from scripts import s1v2_protocol as protocol
from scripts import s1v2_submission_common as common
from scripts import submit_s1v2_ibm as ibm
from scripts import submit_s1v2_iqm as iqm


def _if_blocks(circuit: QuantumCircuit) -> list[IfElseOp]:
    """Return the top-level ``if_else`` operations of a circuit.

    Parameters
    ----------
    circuit
        Circuit to inspect.

    """
    return [inst.operation for inst in circuit.data if isinstance(inst.operation, IfElseOp)]


@pytest.mark.parametrize(
    ("candidate", "arm"), [("c0", "CL"), ("c0", "OL"), ("c0", "NONE"), ("c2", "CL")]
)
def test_arm_structure_matches_the_frozen_design(candidate: str, arm: str) -> None:
    """Each arm has the frozen registers, conditional blocks, control bits and resets.

    Parameters
    ----------
    candidate
        Frozen candidate name.
    arm
        Arm name.

    """
    circuit = protocol.build_arm(candidate, arm, 1, "X")
    rounds = int(protocol.CANDIDATES[candidate]["rounds"])
    blocks = _if_blocks(circuit)
    assert len(blocks) == (0 if arm == "NONE" else protocol.N_SYSTEM * rounds)
    for block in blocks:
        names = {inst.operation.name for inst in block.blocks[0].data}
        assert names <= {"rx", "ry"}, names
    controlling = {circuit.find_bit(block.condition[0]).registers[0][0].name for block in blocks}
    assert controlling == (set() if arm == "NONE" else {"cb" if arm == "OL" else "mb"})
    assert [reg.name for reg in circuit.cregs] == (
        ["mb", "cb", "ro"] if arm == "OL" else ["mb", "ro"]
    )
    assert circuit.num_qubits == protocol.N_SYSTEM * (3 if arm == "OL" else 2)
    assert circuit.count_ops()["reset"] == protocol.N_SYSTEM * rounds * (2 if arm == "OL" else 1)


def test_rotation_helper_is_rz_of_minus_theta_up_to_phase() -> None:
    """The rx/ry rotation helper equals RZ(-theta), not RZ(theta)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    theta = 0.73
    circuit = QuantumCircuit(1)
    protocol._rz_from_rx_ry(circuit, theta, 0)
    reference = QuantumCircuit(1)
    reference.rz(-theta, 0)
    assert Operator(circuit).equiv(Operator(reference))
    opposite = QuantumCircuit(1)
    opposite.rz(theta, 0)
    assert not Operator(circuit).equiv(Operator(opposite))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidate": "c9", "arm": "CL", "polarity": 1, "basis": "X"},
        {"candidate": "c0", "arm": "XX", "polarity": 1, "basis": "X"},
        {"candidate": "c0", "arm": "CL", "polarity": 0, "basis": "X"},
        {"candidate": "c0", "arm": "CL", "polarity": 1, "basis": "Z"},
    ],
)
def test_arm_refuses_values_outside_the_design(kwargs: dict[str, object]) -> None:
    """Arguments outside the frozen design are refused.

    Parameters
    ----------
    kwargs
        Keyword arguments for ``build_arm``.

    """
    with pytest.raises(ValueError):
        protocol.build_arm(**kwargs)  # type: ignore[arg-type]


def test_model_latency_is_simulation_only() -> None:
    """Modelled latency adds delays only when asked and refuses negative values."""
    plain = protocol.build_arm("c0", "CL", 1, "X")
    delayed = protocol.build_arm("c0", "CL", 1, "X", model_latency_ns=500)
    assert "delay" not in plain.count_ops()
    rounds = int(protocol.CANDIDATES["c0"]["rounds"])
    assert delayed.count_ops()["delay"] == 2 * rounds * delayed.num_qubits
    with pytest.raises(ValueError):
        protocol.build_arm("c0", "CL", 1, "X", model_latency_ns=-1)


def test_pilot_circuits_and_refusals() -> None:
    """The pilot matrix has the frozen size, conditional count and refusals."""
    circuits = protocol.pilot_circuits()
    assert len(circuits) == len(protocol.PILOT_EVENTS) * 2 * 2
    conditional = protocol.build_pilot(4, "conditional", "Y")
    assert len(_if_blocks(conditional)) == 4
    assert not _if_blocks(protocol.build_pilot(4, "measure_only", "X"))
    for bad in ((3, "conditional", "X"), (4, "other", "X"), (4, "conditional", "Z")):
        with pytest.raises(ValueError):
            protocol.build_pilot(*bad)


def test_main_matrix_and_digest_are_frozen_and_order_sensitive() -> None:
    """The main matrix is unique by label and its digest is stable and order sensitive."""
    circuits = protocol.main_circuits()
    assert len(circuits) == len(protocol.MAIN_ARMS) * 2 + 2
    assert len({c.name for c in circuits}) == len(circuits)
    first = protocol.logical_digest(circuits)
    assert first == protocol.logical_digest(protocol.main_circuits())
    assert first != protocol.logical_digest(list(reversed(circuits)))


def test_count_helpers() -> None:
    """Register marginals, bit probabilities and the order parameter are exact."""
    counts = {"011 000001": 3, "110 000000": 1}
    assert protocol.register_marginal(counts, 0) == {"011": 3, "110": 1}
    assert protocol.readout_marginal(counts, 3) == {"011": 3, "110": 1}
    assert protocol.readout_marginal({"011000001": 4}, 3) == {"011": 4}
    with pytest.raises(ValueError):
        protocol.readout_marginal({"01": 4}, 3)
    probabilities = protocol.bit_probabilities({"001": 1, "011": 1}, 3)
    assert probabilities.tolist() == [1.0, 0.5, 0.0]
    with pytest.raises(ValueError):
        protocol.bit_probabilities({}, 3)
    with pytest.raises(ValueError):
        protocol.bit_probabilities({"01": 1}, 3)
    assert math.isclose(protocol.order_parameter(np.ones(3), np.zeros(3)), 1.0)
    assert protocol.monitor_rates({"000001": 2, "000000": 2}, 2) == [0.5, 0, 0, 0, 0, 0]


def test_readout_scale_and_refusal() -> None:
    """The readout contraction is exact and an uninformative readout is refused."""
    scale = protocol.readout_scale({"000": 90, "001": 10}, {"111": 80, "110": 20}, 3)
    assert np.allclose(scale, [0.7, 1.0, 1.0])
    with pytest.raises(ValueError):
        protocol.readout_scale({"001": 1}, {"110": 1}, 3)


def _simulate(
    circuits: Sequence[QuantumCircuit], shots: int, seed: int = 3
) -> dict[str, dict[str, int]]:
    """Simulate circuits ideally and return the ``ro`` counts by label.

    Parameters
    ----------
    circuits
        Circuits to simulate.
    shots
        Shots per circuit.
    seed
        Simulator seed.

    """
    simulator = AerSimulator(method="statevector")
    result = simulator.run(
        transpile(circuits, simulator, seed_transpiler=1), shots=shots, seed_simulator=seed
    ).result()
    return {
        c.name: protocol.register_marginal(result.get_counts(i), 0) for i, c in enumerate(circuits)
    }


def test_ideal_main_analysis_passes_primary_and_polarity() -> None:
    """Ideal counts pass the preregistered primary endpoint and polarity check."""
    counts = _simulate(protocol.main_circuits(), 2048)
    analysis = protocol.analyse_main(counts, draws=400)
    assert analysis["primary_pass"] is True
    assert analysis["polarity_pass"] is True
    primary = analysis["primary_c0_cl_minus_ol"]
    assert isinstance(primary, dict) and primary["point"] > 0.4


def test_main_analysis_fails_when_closed_loop_equals_open_loop() -> None:
    """The primary endpoint fails when the closed loop carries no extra information."""
    counts = _simulate(protocol.main_circuits(), 2048)
    for basis in protocol.BASES:
        counts[protocol.arm_label("c0", "CL", 1, basis)] = counts[
            protocol.arm_label("c0", "OL", 1, basis)
        ]
    analysis = protocol.analyse_main(counts, draws=400)
    assert analysis["primary_pass"] is False


def test_pilot_rule_goes_and_stops() -> None:
    """The pilot rule permits ideal coherence and stops lost or absent coherence."""
    ideal = _simulate(protocol.pilot_circuits(), 1024)
    assert protocol.analyse_pilot(ideal)["main_block_go"] is True
    stopped = dict(ideal)
    for basis in protocol.BASES:
        stopped[protocol.pilot_label(4, "conditional", basis)] = {"0": 512, "1": 512}
    decision = protocol.analyse_pilot(stopped)
    assert decision["main_block_go"] is False
    ratio = decision["retained_ratio_at_c0_events"]
    assert isinstance(ratio, float) and ratio < protocol.PILOT_COHERENCE_FLOOR
    dead = dict(stopped)
    for basis in protocol.BASES:
        dead[protocol.pilot_label(0, "conditional", basis)] = {"0": 512, "1": 512}
    assert protocol.analyse_pilot(dead)["retained_ratio_at_c0_events"] == 0.0


def _journal(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """Open a pilot journal for ``ibm_fez`` in a temporary directory.

    Parameters
    ----------
    tmp_path
        Temporary directory.

    """
    path = common.journal_path(tmp_path, "ibm", "ibm_fez", "pilot")
    journal = common.open_journal(
        path,
        experiment_id=protocol.EXPERIMENT_ID,
        provider="ibm",
        backend="ibm_fez",
        stage="pilot",
        logical_digest="a",
        isa_digest="b",
        labels=["x"],
        shots={"x": 1},
        layout={"x": [0]},
    )
    return path, journal


def test_journal_is_written_before_and_refuses_a_second_submission(tmp_path: Path) -> None:
    """A journal exists before submission and blocks a second submission of the stage.

    Parameters
    ----------
    tmp_path
        Temporary directory.

    """
    path, journal = _journal(tmp_path)
    assert json.loads(path.read_text())["status"] == "submitting"
    with pytest.raises(common.StageRefused):
        _journal(tmp_path)
    with pytest.raises(ValueError):
        common.record_job(path, journal, "  ")
    common.record_job(path, journal, "job-1")
    stored = json.loads(path.read_text())
    assert stored["status"] == "submitted" and stored["job_ids"] == ["job-1"]
    with pytest.raises(ValueError):
        common.journal_path(tmp_path, "ibm", "ibm_fez", "other")


def test_pilot_gate(tmp_path: Path) -> None:
    """The main stage needs a passing pilot of the same backend.

    Parameters
    ----------
    tmp_path
        Temporary directory.

    """
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps({"backend": "garnet", "main_block_go": True}))
    assert common.require_pilot_go(path, "garnet")["main_block_go"] is True
    with pytest.raises(common.StageRefused):
        common.require_pilot_go(path, "ibm_fez")
    path.write_text(json.dumps({"backend": "garnet", "main_block_go": False}))
    with pytest.raises(common.StageRefused):
        common.require_pilot_go(path, "garnet")


def test_count_coercion() -> None:
    """Counts must be non-negative integers that conserve the shot number."""
    assert common.coerce_counts({"0": 2, "1": 2}, 4) == {"0": 2, "1": 2}
    for bad in ({"0": -1}, {"0": True}, {"0": 1.5}):
        with pytest.raises(ValueError):
            common.coerce_counts(bad, 1)
    with pytest.raises(ValueError):
        common.coerce_counts({"0": 1}, 2)


def test_write_json_is_atomic_and_hashed(tmp_path: Path) -> None:
    """JSON is written without a leftover temporary file and its digest matches.

    Parameters
    ----------
    tmp_path
        Temporary directory.

    """
    path = tmp_path / "a" / "b.json"
    digest = common.write_json(path, {"k": 1})
    assert common.sha256_bytes(path.read_bytes()) == digest
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("module", [ibm, iqm])
def test_submitters_refuse_without_owner_go_before_any_provider_call(
    module: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both submitters refuse without the owner GO and write no journal.

    Parameters
    ----------
    module
        Submitter module under test.
    tmp_path
        Temporary directory.
    capsys
        Captured standard streams.

    """
    args = ["submit", "--stage", "pilot", "--out-dir", str(tmp_path)]
    if module is ibm:
        args += ["--credentials-vault", str(tmp_path / "absent.md")]
    assert module.main(args) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not any(tmp_path.rglob("*journal*"))


@pytest.mark.parametrize("module", [ibm, iqm])
def test_main_stage_requires_a_passing_pilot(module: ModuleType, tmp_path: Path) -> None:
    """Both submitters refuse the main stage without a matching passing pilot.

    Parameters
    ----------
    module
        Submitter module under test.
    tmp_path
        Temporary directory.

    """
    base = ["submit", "--stage", "main", "--out-dir", str(tmp_path), "--i-have-owner-go"]
    if module is ibm:
        base += ["--credentials-vault", str(tmp_path / "absent.md"), "--confirm-budget"]
    assert module.main(base) == 2
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"backend": "elsewhere", "main_block_go": True}))
    assert module.main([*base, "--pilot-analysis", str(pilot)]) == 2
    assert not any(tmp_path.rglob("*journal*"))


def test_stage_shots_and_budget() -> None:
    """Stage shot numbers and the QPU-seconds estimate follow the preregistration."""
    circuits, shots = ibm.stage_circuits("main")
    assert sum(shots.values()) == len(circuits) * protocol.MAIN_SHOTS
    assert ibm.estimate_seconds(10) == pytest.approx(33.0)
    pilot, pilot_shots = iqm.stage_circuits("pilot")
    assert set(pilot_shots.values()) == {protocol.PILOT_SHOTS} and len(pilot) == 20
    for module in (ibm, iqm):
        with pytest.raises(ValueError):
            module.stage_circuits("other")


def test_provider_helpers_on_a_vendor_reference_backend() -> None:
    """Transpile, two-qubit and layout helpers work on a vendor reference backend."""
    from qiskit_ibm_runtime.fake_provider import FakeFez

    backend = FakeFez()
    circuits = protocol.pilot_circuits()[:4] + [protocol.build_arm("c2", "CL", 1, "X")]
    for module in (ibm, iqm):
        isa = module._transpile(circuits, backend)
        two_qubit = module._two_qubit(isa)
        layouts = module._layouts(isa)
        assert set(two_qubit) == {c.name for c in circuits}
        assert two_qubit[circuits[0].name] == 0
        assert all(isinstance(q, int) for q in layouts[circuits[-1].name])


def _raw_record(tmp_path: Path, stage: str, circuits: Sequence[QuantumCircuit]) -> Path:
    """Write a raw-count record of ideally simulated circuits.

    Parameters
    ----------
    tmp_path
        Temporary directory.
    stage
        Stage name.
    circuits
        Circuits to simulate.

    """
    simulator = AerSimulator(method="statevector")
    shots = 1024
    result = simulator.run(
        transpile(circuits, simulator, seed_transpiler=1), shots=shots, seed_simulator=9
    ).result()
    counts = {}
    for index, circuit in enumerate(circuits):
        full = result.get_counts(index)
        registers = {"ro": protocol.register_marginal(full, 0)}
        if any(reg.name == "mb" for reg in circuit.cregs) and stage == "main":
            registers["mb"] = protocol.register_marginal(full, -1)
        counts[circuit.name] = registers
    path = tmp_path / f"ibm_fez_{stage}_raw_counts.json"
    common.write_json(
        path, {"stage": stage, "backend": "ibm_fez", "provider": "ibm", "counts": counts}
    )
    return path


@pytest.mark.parametrize("stage", ["pilot", "main"])
def test_analyse_command_applies_the_preregistered_rule(stage: str, tmp_path: Path) -> None:
    """The analyse command writes the preregistered pilot or main analysis.

    Parameters
    ----------
    stage
        Stage name.
    tmp_path
        Temporary directory.

    """
    circuits = protocol.pilot_circuits() if stage == "pilot" else protocol.main_circuits()
    raw = _raw_record(tmp_path, stage, circuits)
    assert ibm.main(["analyse", "--raw-counts", str(raw)]) == 0
    analysis = json.loads((tmp_path / f"ibm_fez_{stage}_analysis.json").read_text())
    assert analysis["backend"] == "ibm_fez"
    assert analysis["raw_counts_sha256"] == common.sha256_bytes(raw.read_bytes())
    if stage == "pilot":
        assert analysis["main_block_go"] is True
    else:
        assert analysis["primary_pass"] is True and analysis["polarity_pass"] is True
        rates = analysis["monitor_rates"][protocol.arm_label("c0", "CL", 1, "X")]
        frozen = protocol.COIN_PROBABILITIES["c0"]
        assert len(rates) == len(frozen)
        assert all(abs(r - f) < 0.08 for r, f in zip(rates, frozen, strict=True))


def test_open_loop_coins_reproduce_the_frozen_rates() -> None:
    """The open-loop coins fire at the frozen per-slot probabilities."""
    circuit = protocol.build_arm("c0", "OL", 1, "X")
    simulator = AerSimulator(method="statevector")
    full = (
        simulator.run(
            transpile(circuit, simulator, seed_transpiler=1), shots=4096, seed_simulator=4
        )
        .result()
        .get_counts()
    )
    coins = protocol.register_marginal(full, 1)
    rates = protocol.monitor_rates(coins, 2)
    assert all(
        abs(r - f) < 0.04 for r, f in zip(rates, protocol.COIN_PROBABILITIES["c0"], strict=True)
    )
