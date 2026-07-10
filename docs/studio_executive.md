<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- Studio executive actions -->

# Studio Executive Actions

The QUANTUM studio is an *executive* tool, not only a federation publisher. The
[Studio Federation](studio_federation.md) surface is the *informative* layer the
SCPN-STUDIO hub ingests; the executive spine described here is the layer that
actually **runs** a verb, writes a standalone reproduction script, and seals an
auditable record.

## Lifecycle

```text
request -> plan -> (approval gate) -> execute -> generate-script -> seal
```

The verb spine (`scpn_quantum_control.studio.verbs`) is authoritative over
safety. A handler receives a `VerbContract` resolved from the declared `Verb`,
so it can never widen its own side effect, safety tier, backend set, or approval
requirement. A `live-hardware` verb (QPU submission through the provider HAL) or
a `certified` verb requires an explicit approval on the request; without it the
spine returns a fail-closed `gated` record and never executes — the deploy-onto-
endpoint safety contract.

## Running a differentiate action

The first plugin is the read-only `differentiate` handler. It differentiates a
bounded *rational* scalar program (named parameters plus `mul`/`add` operations
and numeric literals, no transcendentals so value and gradient are exact and
platform reproducible), executes it through the compiled `scpn_quantum_engine`
effect-IR replay, cross-checks the reverse-mode gradient against central finite
differences, and writes a standalone reproduction script.

```python
from scpn_quantum_control.studio import ExecutiveRequest, preview_action, run_action
from scpn_quantum_control.studio.executive_differentiate import default_registry

registry = default_registry()
request = ExecutiveRequest(
    verb="differentiate",
    action_id="demo-x2-plus-2y",
    parameters={
        "inputs": [["x", 3.0], ["y", 5.0]],
        "operations": [
            {"op": "mul", "inputs": ["x", "x"], "into": "x2"},
            {"op": "mul", "inputs": ["y", "2.0"], "into": "y2"},
            {"op": "add", "inputs": ["x2", "y2"], "into": "f"},
        ],
        "output": "f",
    },
)

plan = preview_action(request, registry=registry)
assert plan.requires_approval is False

record = run_action(request, registry=registry)
assert record.result.status == "succeeded"
record.result.outputs["value"]     # 19.0
record.result.outputs["gradient"]  # [6.0, 2.0]
record.result.outputs["verified"]  # True (reverse-mode agrees with finite differences)
print(record.script.source)        # a standalone, runnable reproduction script
```

`preview_action` returns the inspectable plan without executing. Every sealed
`ExecutiveRecord` carries a content digest over its request, plan, result, and
script, and its `produced_schemas` links the action back to the informative
`studio.differentiation-evidence.v1` family.

## Compiling a network

The read-only `compile` verb compiles an arbitrary bounded `K_nm`/`omega`
oscillator network into the studio's bit-exact XY compile unit. It builds the
`studio.xy-compile-recompute.v1` unit, verifies it against its own reference,
and writes a reproduction script.

```python
from scpn_quantum_control.studio.executive import ActionRegistry, ExecutiveRequest, run_action
from scpn_quantum_control.studio.executive_compile import CompileActionHandler

registry = ActionRegistry()
registry.register(CompileActionHandler())

request = ExecutiveRequest(
    verb="compile",
    action_id="compile-3node",
    parameters={
        "K_nm": [[0.0, 0.4, 0.1], [0.4, 0.0, 0.3], [0.1, 0.3, 0.0]],
        "omega": [-0.1, 0.05, 0.05],
        "time": 0.1,
        "trotter_steps": 1,
        "trotter_order": 1,
    },
)
record = run_action(request, registry=registry)
record.result.outputs["input_sha256"]  # the bit-exact compile digest
record.result.outputs["verified"]       # True (unit matches its own reference)
```

The claim boundary is the bit-exact XY compile decision path only: the input
digest is recompute-verifiable in a browser through the WASM kernel (see
[Studio Federation](studio_federation.md) → WS-1 recompute kernel). It is not a
physical `K_nm` claim, a continuous simulator value, or QPU execution.

## Simulating an evolution

The read-only `simulate` verb evolves a bounded `K_nm`/`omega` oscillator
network on a local dense-statevector simulator. It Trotter-evolves the XY spin
Hamiltonian from `t = 0` to `t_max` in `dt` steps and measures the Kuramoto
synchronisation order parameter `R(t)` along the trajectory, returning the
trajectory summary and a reproduction script.

```python
from scpn_quantum_control.studio.executive import (
    ActionRegistry,
    ExecutiveRequest,
    run_action,
)
from scpn_quantum_control.studio.executive_simulate import SimulateActionHandler

registry = ActionRegistry()
registry.register(SimulateActionHandler())

request = ExecutiveRequest(
    verb="simulate",
    action_id="simulate-3node",
    parameters={
        "K_nm": [[0.0, 0.4, 0.1], [0.4, 0.0, 0.3], [0.1, 0.3, 0.0]],
        "omega": [-0.1, 0.05, 0.05],
        "t_max": 0.2,
        "dt": 0.1,
        "trotter_per_step": 1,
        "trotter_order": 1,
    },
)
record = run_action(request, registry=registry)
record.result.outputs["order_parameter_final"]  # Kuramoto R at t_max
record.result.outputs["order_parameter_mean"]    # trajectory-mean R
record.result.outputs["n_points"]                # trajectory length
print(record.script.source)                       # a standalone reproduction script
```

The claim boundary is a *simulator estimate*: the reported order parameter is a
dense-statevector Trotter approximation at the stated step and Trotter
resolution, not a continuous-time exact solution, a physical `K_nm` claim, or QPU
execution. The reproduction script re-evolves the network and checks the sealed
`order_parameter_final`/`order_parameter_mean` summary to a numerical tolerance.
The action feeds the informative `studio.quantum-evolution.v1` family.

## Analysing a phase cloud

The read-only `analyse` verb runs the synchronisation witness over a bounded
phase cloud: harmonic Kuramoto order parameters, geodesic phase distances, and
exact Vietoris–Rips persistent homology (H0/H1 persistence, Betti curves,
persistent component count at a reference scale, dominant loop lifetime).

```python
from scpn_quantum_control.studio.executive import (
    ActionRegistry,
    ExecutiveRequest,
    run_action,
)
from scpn_quantum_control.studio.executive_analyse import AnalyseActionHandler

registry = ActionRegistry()
registry.register(AnalyseActionHandler())

request = ExecutiveRequest(
    verb="analyse",
    action_id="analyse-4node",
    parameters={
        "phases": [0.0, 0.05, -0.04, 0.02],
        "thresholds": [0.0, 0.5, 1.0, 2.0, 3.0],
        "reference_scale": 0.5,
        "expected_components": 1,
    },
)
record = run_action(request, registry=registry)
record.result.outputs["order_parameter"]              # first-harmonic Kuramoto R
record.result.outputs["persistent_component_count"]   # H0 components at the reference scale
record.result.outputs["dominant_h1_persistence"]       # dominant loop lifetime
record.result.outputs["witness_passed"]                # witness verdict
```

The claim boundary is a classical phase-configuration analysis: exact
persistent homology of the given finite phase cloud over the given filtration
thresholds — not a quantum-state measurement, a dynamical-evolution claim, or a
statement about any generating model. The action feeds the informative
`studio.sync-analysis.v1` family.

## Validating the claim ledger

The read-only `validate` verb checks the committed WS-3 reference-validation
registry against the committed differentiable claim ledger — every
certification must be unique, point at a ledger claim, and certify a promoted
claim — then measures the reference-validated coverage frontier.

```python
from scpn_quantum_control.studio.executive import (
    ActionRegistry,
    ExecutiveRequest,
    run_action,
)
from scpn_quantum_control.studio.executive_validate import ValidateActionHandler

registry = ActionRegistry()
registry.register(ValidateActionHandler())

request = ExecutiveRequest(verb="validate", action_id="validate-ledger", parameters={})
record = run_action(request, registry=registry)
record.result.outputs["validation_passed"]    # registry-consistency verdict
record.result.outputs["certificate_count"]    # WS-3 certifications on file
record.result.outputs["total_claims"]         # ledger claims measured
record.result.outputs["answer_rate"]          # reference-validated fraction
```

The claim boundary is registry-consistency and coverage measurement over the
committed artefacts only: it does not prove any physics claim itself, run a
simulation, or touch hardware. The action feeds the informative
`studio.physics-validation.v1` family.

## Replaying the committed hardware evidence

The read-only `replay` verb re-verifies the committed hardware result packs
from their raw artefacts and provenance: every declared artefact must exist
with its exact byte size and SHA-256 digest, and every declared provider job
identifier must appear inside the committed raw payloads. Any drift — a
missing artefact, a digest mismatch, an absent job identifier, or an unknown
pack id — seals a `failed` record instead of a weakened summary.

```python
from scpn_quantum_control.studio.executive import (
    ActionRegistry,
    ExecutiveRequest,
    run_action,
)
from scpn_quantum_control.studio.executive_replay import ReplayActionHandler

registry = ActionRegistry()
registry.register(ReplayActionHandler())

request = ExecutiveRequest(
    verb="replay",
    action_id="replay-all-packs",
    parameters={},  # or {"pack_ids": ["phase1_dla_parity_ibm_kingston_2026_04"]}
)
record = run_action(request, registry=registry)
record.result.outputs["replay_passed"]    # True — every artefact re-verified
record.result.outputs["pack_count"]       # committed packs re-verified
record.result.outputs["artifact_count"]   # artefacts digest-checked
```

The claim boundary is integrity re-verification of the committed artefacts
only: replay proves the raw evidence on disk is exactly what the manifest
promised — it does not prove any derived physics claim, contact a provider,
or produce new counts. The action feeds the informative
`studio.evidence-replay.v1` family.

## Benchmarking the native construction

The simulated `benchmark` verb times the dense XY-Hamiltonian construction for
a bounded `K_nm`/`omega` network on the requested backend — the native Rust
PyO3 kernel (default) or the pure-numpy reference — parity-checks the native
operator against the reference, and summarises the committed tier-benchmark
databank.

```python
from scpn_quantum_control.studio.executive import (
    ActionRegistry,
    ExecutiveRequest,
    run_action,
)
from scpn_quantum_control.studio.executive_benchmark import BenchmarkActionHandler

registry = ActionRegistry()
registry.register(BenchmarkActionHandler())

request = ExecutiveRequest(
    verb="benchmark",
    action_id="bench-3node",
    parameters={
        "K_nm": [[0.0, 0.4, 0.1], [0.4, 0.0, 0.3], [0.1, 0.3, 0.0]],
        "omega": [-0.1, 0.05, 0.05],
        "repeats": 5,
        "warmup": 1,
    },
)
record = run_action(request, registry=registry)
record.result.outputs["parity"]              # native operator matches the numpy reference
record.result.outputs["speedup_p50"]          # native-vs-reference P50 ratio (never asserted)
record.result.outputs["databank_row_count"]   # committed benchmark databank summary
record.result.outputs["production_claim_allowed"]  # always False
```

The claim boundary is deliberately narrow: the wall-clock numbers are
opportunistic local timing on a shared workstation — environment-dependent
regression evidence, never a published performance claim — so every sealed
record carries `production_claim_allowed: False` and its timing caveat
verbatim. Only the *deterministic* verdicts are reproducible: the reproduction
script re-asserts the operator shape, the native/reference parity, and the
committed databank row count, and re-prints fresh timings without asserting
them. A `rust` request fails closed when the native kernel is not importable.
The action feeds the informative `studio.native-speedup.v1` and
`studio.benchmark-databank.v1` families.

## Deploying to a QPU endpoint

The `execute` verb is the studio's only live-hardware action, and it is
approval-gated. The studio never submits a live QPU job itself — submission
needs provider credentials and costs real money. The `execute` handler plans an
approval-gated deployment and, on an approved request, builds a **no-submit**
deployment dossier and writes a standalone operator submission script.

```python
from scpn_quantum_control.studio.executive import (
    ActionRegistry,
    ExecutiveRequest,
    run_action,
)
from scpn_quantum_control.studio.executive_execute import ExecuteActionHandler

registry = ActionRegistry()
registry.register(ExecuteActionHandler())

request = ExecutiveRequest(
    verb="execute",
    action_id="deploy-brisbane",
    parameters={
        "provider": "ibm-quantum",
        "endpoint": "ibm_brisbane",
        "circuit_digest": "sha256:...",  # bit-exact link to a recompute-verifiable compile
        "circuit_ref": "data/studio/xy_compile_recompute_unit_20260708.json",
        "shots": 4096,
    },
    approved=True,  # without this the action is gated and never runs
)
record = run_action(request, registry=registry)
record.result.outputs["submitted"]      # False -- the studio never submits
record.result.outputs["result_status"]  # "unverifiable" until a provider attestation is attached
print(record.script.source)             # the operator submission script (guarded behind --confirm)
```

Without `approved=True` the record is `gated` and no script is written. The
generated script refuses to submit without `--confirm`, digests the returned
counts, and hands them to `build_qpu_result_pack_unit`; the operator attaches
their provider attestation to make the `studio.qpu-result-pack.v1` result
attestation-verifiable (see [Studio Federation](studio_federation.md)). The
studio never contacts a provider or produces counts.

## Running actions from the CLI

The `scpn-studio-run` entry point drives the executive spine from a shell (or
from the SCPN-STUDIO hub), one verb per invocation. It prints the sealed
`ExecutiveRecord` — or the inspectable plan under `--preview` — as JSON on
stdout, and writes the generated reproduction script into `--script-dir` on
success.

```bash
# compile a bounded network (inline parameters)
scpn-studio-run compile --action-id compile-3node \
  --params '{"K_nm": [[0.0, 0.4], [0.4, 0.0]], "omega": [-0.1, 0.1],
             "time": 0.1, "trotter_steps": 1, "trotter_order": 1}' \
  --script-dir out/

# inspect a simulate plan without executing it
scpn-studio-run simulate --action-id sim-3node --params-file request.json --preview

# an approval-gated deploy fails closed without --approve (exit code 3)
scpn-studio-run execute --action-id deploy --params-file deploy.json --approve
```

Exit codes are scriptable: `0` succeeded (or previewed), `1` the action failed,
`2` a request or parameter error, `3` the action was gated — a live-hardware or
certified verb invoked without `--approve` never executes. The default registry
carries every shipped handler (`analyse`, `benchmark`, `compile`,
`differentiate`, `execute`, `replay`, `simulate`, `validate`) and is also
available from Python as `scpn_quantum_control.studio.build_default_registry()`.

## Claim boundary

The differentiate action proves the exact reverse-mode value and gradient of a
bounded rational program, cross-checked against central finite differences — not
transcendental, linear-algebra, unbounded, provider, or hardware
differentiation. Gated verbs such as `execute` never run without an explicit
approval on the request.
