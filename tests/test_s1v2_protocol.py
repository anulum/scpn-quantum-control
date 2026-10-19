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
    assert [reg.name for reg in circuit.cregs] == ["mb", "cb", "ro"]
    assert circuit.num_qubits == protocol.N_SYSTEM * 3
    assert circuit.count_ops()["reset"] == protocol.N_SYSTEM * rounds * 2
    assert circuit.count_ops()["measure"] == protocol.N_SYSTEM * (2 * rounds + 1)


@pytest.mark.parametrize("candidate", ["c0", "c2"])
def test_closed_and_open_loop_use_identical_resources(candidate: str) -> None:
    """CL and OL have the same operations and differ only in the control register.

    Parameters
    ----------
    candidate
        Frozen candidate name.

    """
    closed = protocol.build_arm(candidate, "CL", 1, "X")
    opened = protocol.build_arm(candidate, "OL", 1, "X")
    none = protocol.build_arm(candidate, "NONE", 1, "X")
    assert dict(closed.count_ops()) == dict(opened.count_ops())
    assert closed.num_qubits == opened.num_qubits == none.num_qubits
    for name in ("measure", "reset"):
        assert closed.count_ops()[name] == none.count_ops()[name]


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
        transpile(circuits, simulator, seed_transpiler=1, num_processes=1),
        shots=shots,
        seed_simulator=seed,
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


def test_depolarised_pilot_with_zero_coherence_is_refused() -> None:
    """A fully depolarised pilot (true coherence 0) cannot pass through a noisy ratio."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(1.0, 1), ["ry"])
    simulator = AerSimulator(noise_model=noise)
    circuits = protocol.pilot_circuits()
    result = simulator.run(
        transpile(
            circuits,
            simulator,
            basis_gates=["ry", "rx", "h", "sdg", "measure", "if_else"],
            num_processes=1,
        ),
        shots=protocol.PILOT_SHOTS,
        seed_simulator=11,
    ).result()
    counts = {
        c.name: protocol.register_marginal(result.get_counts(i), 0) for i, c in enumerate(circuits)
    }
    decision = protocol.analyse_pilot(counts, draws=400)
    assert decision["main_block_go"] is False
    baseline = decision["baseline_lower_99"]
    assert isinstance(baseline, float) and baseline < protocol.PILOT_BASELINE_FLOOR


def _journal(
    tmp_path: Path, stage: str = "pilot", backend: str = "ibm_fez"
) -> tuple[Path, dict[str, object]]:
    """Open a journal in a temporary directory.

    Parameters
    ----------
    tmp_path
        Temporary directory.
    stage
        Stage name.
    backend
        Backend name.

    """
    path = common.journal_path(tmp_path, "ibm", backend, stage)
    circuits, shots = common.stage_circuits(stage)
    journal = common.open_journal(
        path,
        experiment_id=protocol.EXPERIMENT_ID,
        provider="ibm",
        backend=backend,
        stage=stage,
        logical_digest=protocol.logical_digest(circuits),
        isa_digest="isa",
        labels=[c.name for c in circuits],
        shots=shots,
        layout={},
        extra={"main_reference_layout": list(range(9))},
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
    assert not list(path.parent.glob(".*.tmp"))
    with pytest.raises(ValueError):
        common.journal_path(tmp_path, "ibm", "ibm_fez", "other")


_RACE_WORKER = """
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts import s1v2_submission_common as common
go = Path(sys.argv[3])
while not go.exists():
    time.sleep(0.001)
try:
    common.open_journal(
        common.journal_path(Path(sys.argv[2]), "ibm", "ibm_fez", "pilot"),
        experiment_id="race", provider="ibm", backend="ibm_fez", stage="pilot",
        logical_digest="d", isa_digest="i", labels=["x"], shots={"x": 1}, layout={},
    )
    print("granted")
except common.StageRefused:
    print("refused")
"""


@pytest.mark.parametrize("trial", range(4))
def test_concurrent_processes_get_exactly_one_reservation(trial: int, tmp_path: Path) -> None:
    """Eight separate interpreters racing for one stage yield one grant and seven refusals.

    Parameters
    ----------
    trial
        Repetition index.
    tmp_path
        Temporary directory.

    """
    import subprocess
    import sys
    import time

    repo = str(Path(__file__).resolve().parents[1])
    go = tmp_path / "go"
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", _RACE_WORKER, repo, str(tmp_path), str(go)],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(8)
    ]
    time.sleep(1.0)
    go.touch()
    outcomes = sorted(worker.communicate(timeout=60)[0].strip() for worker in workers)
    assert all(worker.returncode == 0 for worker in workers)
    assert outcomes == ["granted"] + ["refused"] * 7, (trial, outcomes)
    journal = common.journal_path(tmp_path, "ibm", "ibm_fez", "pilot")
    assert json.loads(journal.read_text())["status"] == "submitting"
    assert not list(journal.parent.glob(".*.tmp"))


def _pilot_chain(tmp_path: Path, *, dead: bool = False, variant: str = "") -> Path:
    """Build a real local pilot chain (journal, raw counts, analysis) from simulation.

    Parameters
    ----------
    tmp_path
        Temporary directory.
    dead
        Replace the zero-event coherence by an incoherent baseline.
    variant
        ``"extra_label"`` adds an unexpected label; ``"short_shots"`` drops shots
        from one label; both keep the chain hash-consistent.

    """
    path, journal = _journal(tmp_path)
    common.record_job(path, journal, "job-pilot")
    counts = _simulate(protocol.pilot_circuits(), protocol.PILOT_SHOTS)
    if dead:
        for basis in protocol.BASES:
            counts[protocol.pilot_label(0, "conditional", basis)] = {"0": 1024, "1": 1024}
    if variant == "extra_label":
        counts["unexpected"] = {"0": protocol.PILOT_SHOTS}
    if variant == "short_shots":
        label = protocol.pilot_label(8, "measure_only", "X")
        counts[label] = {key: value // 2 for key, value in counts[label].items()}
    raw = common.write_raw_counts(path, {label: {"ro": c} for label, c in counts.items()})
    return common.analyse_raw(raw)


def test_pilot_chain_passes_when_complete(tmp_path: Path) -> None:
    """A complete, hash-consistent passing pilot chain permits the main stage.

    Parameters
    ----------
    tmp_path
        Temporary directory.

    """
    analysis = _pilot_chain(tmp_path)
    journal = common.pilot_go(analysis, "ibm", "ibm_fez")
    assert journal["main_reference_layout"] == list(range(9))


@pytest.mark.parametrize(
    "tamper",
    [
        "provider",
        "backend_name",
        "experiment",
        "schema",
        "raw_schema",
        "journal_schema",
        "raw_schema_missing",
        "journal_schema_missing",
        "raw_edit",
        "raw_missing",
        "journal_edit",
        "dead",
        "extra_label",
        "short_shots",
    ],
)
def test_pilot_chain_refuses_every_broken_link(tamper: str, tmp_path: Path) -> None:
    """Wrong identity, stale hashes, missing files or a failing screen are refused.

    Parameters
    ----------
    tamper
        Which link of the chain is broken.
    tmp_path
        Temporary directory.

    """
    variant = tamper if tamper in ("extra_label", "short_shots") else ""
    analysis = _pilot_chain(tmp_path, dead=tamper == "dead", variant=variant)
    raw = analysis.with_name("ibm_fez_pilot_raw_counts.json")
    journal = analysis.with_name("ibm_fez_pilot_journal.json")
    provider, backend = "ibm", "ibm_fez"
    if tamper == "provider":
        provider = "iqm"
    elif tamper == "backend_name":
        backend = "ibm_marrakesh"
    elif tamper in ("experiment", "schema"):
        document = json.loads(analysis.read_text())
        document["experiment_id" if tamper == "experiment" else "schema"] = "other"
        common.write_json(analysis, document)
    elif tamper in (
        "raw_schema",
        "raw_schema_missing",
        "journal_schema",
        "journal_schema_missing",
    ):
        # Reseal the chain so that every hash link stays consistent and only the schema is wrong.
        if tamper.startswith("journal"):
            document = json.loads(journal.read_text())
            if tamper.endswith("missing"):
                document.pop("schema")
            else:
                document["schema"] = "other_v1"
            common.write_json(journal, document)
            raw_document = json.loads(raw.read_text())
            raw_document["journal_sha256"] = common.sha256_bytes(journal.read_bytes())
            common.write_json(raw, raw_document)
        else:
            raw_document = json.loads(raw.read_text())
            if tamper.endswith("missing"):
                raw_document.pop("schema")
            else:
                raw_document["schema"] = "other_v1"
            common.write_json(raw, raw_document)
        document = json.loads(analysis.read_text())
        document["raw_counts_sha256"] = common.sha256_bytes(raw.read_bytes())
        common.write_json(analysis, document)
    elif tamper == "raw_edit":
        document = json.loads(raw.read_text())
        document["job_ids"] = ["edited"]
        common.write_json(raw, document)
    elif tamper == "raw_missing":
        raw.unlink()
    elif tamper == "journal_edit":
        document = json.loads(journal.read_text())
        document["note"] = "edited after retrieval"
        common.write_json(journal, document)
    with pytest.raises(common.StageRefused, match="schema" if "schema" in tamper else None):
        common.pilot_go(analysis, provider, backend)


def test_resealed_chain_with_correct_schemas_still_passes(tmp_path: Path) -> None:
    """Resealing the hash links without a schema change keeps a valid chain valid.

    This is the positive control of the resealed wrong-schema cases above.

    Parameters
    ----------
    tmp_path
        Temporary directory.

    """
    analysis = _pilot_chain(tmp_path)
    raw = analysis.with_name("ibm_fez_pilot_raw_counts.json")
    journal = analysis.with_name("ibm_fez_pilot_journal.json")
    common.write_json(journal, json.loads(journal.read_text()))
    raw_document = json.loads(raw.read_text())
    raw_document["journal_sha256"] = common.sha256_bytes(journal.read_bytes())
    common.write_json(raw, raw_document)
    document = json.loads(analysis.read_text())
    document["raw_counts_sha256"] = common.sha256_bytes(raw.read_bytes())
    common.write_json(analysis, document)
    assert common.pilot_go(analysis, "ibm", "ibm_fez")["stage"] == "pilot"


def test_count_coercion() -> None:
    """Counts must be non-negative integers that conserve the shot number."""
    assert common.coerce_counts({"0": 2, "1": 2}, 4) == {"0": 2, "1": 2}
    for bad in ({"0": -1}, {"0": True}, {"0": 1.5}):
        with pytest.raises(ValueError):
            common.coerce_counts(bad, 1)
    with pytest.raises(ValueError):
        common.coerce_counts({"0": 1}, 2)


def test_write_json_is_durable_and_hashed(tmp_path: Path) -> None:
    """JSON is written without a leftover temporary file and its digest matches.

    Parameters
    ----------
    tmp_path
        Temporary directory.

    """
    path = tmp_path / "a" / "b.json"
    digest = common.write_json(path, {"k": 1})
    assert common.sha256_bytes(path.read_bytes()) == digest
    assert not list(path.parent.glob(".*.tmp"))


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
def test_main_stage_requires_a_passing_pilot_chain(module: ModuleType, tmp_path: Path) -> None:
    """Both submitters refuse the main stage without a valid pilot chain of that backend.

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
    stray = tmp_path / "elsewhere_pilot_analysis.json"
    stray.write_text(json.dumps({"main_block_go": True}))
    assert module.main([*base, "--pilot-analysis", str(stray)]) == 2
    assert not any(tmp_path.rglob("*journal*"))


@pytest.mark.parametrize("value", [float("nan"), 0.0, 10801.0, 2.5, -5.0])
def test_ibm_cap_refuses_invalid_values(value: float) -> None:
    """The per-job QPU cap must be a whole number of seconds within the provider limit.

    Parameters
    ----------
    value
        Candidate cap.

    """
    with pytest.raises(common.StageRefused):
        ibm.validated_cap(value)


def test_ibm_cap_is_held_by_the_installed_runtime_sampler() -> None:
    """A real SamplerV2 on a local vendor reference backend holds the cap; nothing is run."""
    from qiskit_ibm_runtime.fake_provider import FakeFez

    _, effective = ibm.capped_sampler(FakeFez(), ibm.validated_cap(300.0))
    assert effective == 300
    assert ibm.estimate_seconds(10) == pytest.approx(33.0)


def test_common_layout_places_every_stage_on_the_same_qubits() -> None:
    """All main circuits share one physical layout and the pilot runs on its s0 and m0."""
    from qiskit_ibm_runtime.fake_provider import FakeFez

    backend = FakeFez()
    layout = common.main_reference_layout(backend)
    assert len(layout) == 9 and len(set(layout)) == 9
    main = [protocol.build_arm("c0", arm, 1, "X") for arm in ("CL", "OL", "NONE")]
    calibration = common.transpile_stage(protocol.build_calibration(), backend, layout)
    assert {tuple(c.layout.initial_index_layout()[:3]) for c in calibration} == {tuple(layout[:3])}
    isa = common.transpile_stage(main, backend, common.stage_layout("main", layout))
    initial = {tuple(c.layout.initial_index_layout()[:9]) for c in isa}
    assert initial == {tuple(layout)}
    pilot = common.transpile_stage(
        protocol.pilot_circuits()[:2], backend, common.stage_layout("pilot", layout)
    )
    assert {tuple(c.layout.initial_index_layout()[:2]) for c in pilot} == {(layout[0], layout[3])}
    assert common.two_qubit_counts(pilot)[pilot[0].name] == 0
    assert set(common.final_layouts(isa)) == {c.name for c in main}
    with pytest.raises(ValueError):
        common.stage_layout("other", layout)
    with pytest.raises(ValueError):
        common.stage_circuits("other")
    assert common.missing_operations(backend) == []


def test_analyse_command_binds_the_main_analysis(tmp_path: Path) -> None:
    """The analyse command writes a hash-bound main analysis with monitor rates.

    Parameters
    ----------
    tmp_path
        Temporary directory.

    """
    path, journal = _journal(tmp_path, stage="main")
    common.record_job(path, journal, "job-main")
    simulator = AerSimulator(method="statevector")
    circuits = protocol.main_circuits()
    result = simulator.run(
        transpile(circuits, simulator, seed_transpiler=1, num_processes=1),
        shots=protocol.MAIN_SHOTS,
        seed_simulator=9,
    ).result()
    counts = {}
    for index, circuit in enumerate(circuits):
        full = result.get_counts(index)
        registers = {"ro": protocol.register_marginal(full, 0)}
        if any(reg.name == "mb" for reg in circuit.cregs):
            registers["mb"] = protocol.register_marginal(full, -1)
        counts[circuit.name] = registers
    raw = common.write_raw_counts(path, counts)
    assert ibm.main(["analyse", "--raw-counts", str(raw)]) == 0
    analysis = json.loads(raw.with_name("ibm_fez_main_analysis.json").read_text())
    assert analysis["raw_counts_sha256"] == common.sha256_bytes(raw.read_bytes())
    assert analysis["primary_pass"] is True and analysis["polarity_pass"] is True
    rates = analysis["monitor_rates"][protocol.arm_label("c0", "CL", 1, "X")]
    frozen = protocol.COIN_PROBABILITIES["c0"]
    assert all(abs(r - f) < 0.06 for r, f in zip(rates, frozen, strict=True))


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
    rates = protocol.monitor_rates(protocol.register_marginal(full, 1), 2)
    assert all(
        abs(r - f) < 0.04 for r, f in zip(rates, protocol.COIN_PROBABILITIES["c0"], strict=True)
    )
