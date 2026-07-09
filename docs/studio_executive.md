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

## Claim boundary

The differentiate action proves the exact reverse-mode value and gradient of a
bounded rational program, cross-checked against central finite differences — not
transcendental, linear-algebra, unbounded, provider, or hardware
differentiation. Gated verbs such as `execute` never run without an explicit
approval on the request.
