# Pipeline Runtime Contract

Every component that claims to participate in the SCPN stack must expose
a runnable pipeline path. Importable modules are not enough: each
pipeline needs a command, a typed contract, validation, and an auditable
output artifact.

## Three-repository runtime model

| Repository | Runtime role | Contract |
|------------|--------------|----------|
| SC-NeuroCore | Source-facing data bridge and neural/SCPN payload provider. | Provides `scpn_neurocore.bridge` loaders and labelled source artifacts. |
| Phase Orchestrator | Domain-to-oscillator compiler. | Emits validated QPU data artifacts from binding specs, domain packs, or calibrated streams. |
| SCPN Quantum Control | QPU compiler, executor, and measurement analyser. | Consumes QPU data artifacts and produces hardware/simulator result artifacts. |

The data flow is:

```text
source data -> SC-NeuroCore bridge -> Phase Orchestrator compiler
            -> QPU data artifact -> Quantum Control circuit/runtime
            -> counts/result artifact -> analysis/publication gate
```

Quantum Control must consume the final artifact. It must not invent
campaign matrices as a fallback path.

## Runnable pipeline definition

A pipeline is runnable only when all of the following are true:

1. It has a stable entry point: console command, script, or documented
   Python function.
2. It accepts explicit inputs: source path, replay id, artifact path,
   or configuration object.
3. It emits a typed artifact with schema version, source provenance,
   numeric hashes, and reproducibility metadata.
4. It has a smoke test that exercises the entry point without external
   paid services.
5. It has at least one integration test covering the handoff to the
   next repository.
6. It fails loudly when source data or optional hardware is unavailable.
7. Synthetic data requires explicit opt-in and is labelled in metadata.

## Purpose-tuned instances

Multiple instances may run at the same time when their state is isolated.
This is useful because the same stack can serve different purposes:

| Instance type | Example purpose | Isolation key |
|---------------|-----------------|---------------|
| Smoke instance | Fast contract check with fixture artifacts. | Temporary output directory and fixture replay id. |
| Replay instance | Re-run a recorded source/campaign exactly. | Replay id and immutable artifact hash. |
| Calibration instance | Tune extraction, coupling, or mitigation parameters. | Calibration id and parameter manifest. |
| Hardware instance | Submit QPU jobs under cost and queue controls. | Backend, job id, shot budget, and result directory. |
| Analysis instance | Recompute observables from saved counts. | Result artifact hash and analysis version. |

The opening is significant: we can run a connectome replay, a plasma
calibration, and a hardware retrieval loop concurrently without them
mutating each other's state. Each instance becomes an auditable
experiment rather than a terminal session.

## Minimum command surface

The target command surface is:

```text
sc-neurocore bridge export --source SOURCE --mode MODE --out artifact.json
spo qpu-artifact emit --domain-pack PATH --mode MODE --out artifact.json
python -m scpn_quantum_control ... --artifact artifact.json --out result.json
```

Until those exact commands exist, each repository must document the
current equivalent Python function or script and its limitations.

## Publication gate

Publication-grade runs require:

- `source_mode` in `recorded`, `replay`, `curated`, or `derived`
- a `source_timestamp` or `replay_id`
- stable SHA-256 hashes for `K_nm`, `omega`, and optional `theta0`
- no silent local random fallback
- no hard-coded observable targets
- preserved QPU job ids or simulator seeds
- result artifacts that state whether hardware, simulator, mitigation,
  and proxy observables were used

Smoke fixtures are valuable for pipeline health, but they are not
publication evidence.

## Current readiness snapshot

As of 2026-05-12:

- SC-NeuroCore exposes the canonical source-facing bridge namespace
  `scpn_neurocore.bridge`. The prior unseparated package spelling is no
  longer the active tracked package.
- `scpn_neurocore.bridge.load_live_stream` returns an artifact object
  with `K_nm` and `omega`, and bridge-only imports do not eagerly load
  datastream codec dependencies.
- Phase Orchestrator has a QPU data artifact emitter/validator; focused
  artifact tests pass locally.
- Quantum Control imports the canonical namespace in the frontier
  parameter generator and live-loop script. Focused generator tests
  consume SC-NeuroCore-style artifact objects.

The next integration target is a one-command dry run:

```text
SC-NeuroCore fixture/replay -> Phase Orchestrator artifact
-> Quantum Control simulator -> result artifact
```

That dry run should become the permanent inter-repository health check.
