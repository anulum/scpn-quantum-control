# Deep-analysis research lanes

SCPN Quantum Control contains a substantial collection of analysis and gauge
modules. They do not all have the same maturity, evidence, or relationship to
differentiable control. The research-lane registry makes those differences
explicit without turning importability into a product or scientific claim.

The committed snapshot covers **74 modules**: 29 research lanes, 29 prototypes,
and 16 product candidates. “Product candidate” is a software-maturity label,
not productisation. Every row explicitly denies product, control, and
publication grants from registry membership alone.

## Read a row

Each row has four required classification fields:

| Field | Meaning |
|---|---|
| `module` | Exact import path under `scpn_quantum_control.analysis` or `scpn_quantum_control.gauge` |
| `maturity` | `research`, `prototype`, or `product_candidate` software maturity |
| `diff_hook` | No relationship, diagnostic use, an evidence-requiring candidate, a bounded existing composition, or an owner-gated deferred route |
| `claim_status` | `research_only`, `diagnostic_only`, `evidence_bounded`, or a fail-closed `refuse_only` interface |

Rows may also name a promotion target and a repository evidence pointer. An
empty evidence list is intentional for research-only and diagnostic-only code.
An `evidence_bounded` row cannot be constructed without an evidence pointer.

```python
from scpn_quantum_control.analysis import get_research_lane

lane = get_research_lane("scpn_quantum_control.analysis.qfi")
print(lane.maturity.value)          # prototype
print(lane.diff_hook.value)         # candidate_requires_evidence
print(lane.promotion_targets)       # ('geometric-control:planned',)
print(lane.registry_grants_control) # False
```

Unknown modules raise `KeyError`; the API never synthesises a permissive
default.

## Promotion routes

The registry records route status rather than flattening every relationship
into “promoted”:

| Route | Current meaning |
|---|---|
| Geometric QFI control | `planned`; QFI modules remain candidates requiring separate evidence |
| Topology-constrained control | `complete`; selected DLA assets are composed by a separate bounded package and evidence |
| Topological coherence observer | `deferred-owner-gate`; the route remains blocked without owner override |
| Adaptive FIM feedback | `complete`; the feedback and custody modules point to their separate committed evidence |

These links do not transfer the target's status to unrelated modules. In
particular, completion of the topology-control and adaptive-FIM routes does not
promote the full analysis catalogue.

## Inventory gate

`assert_research_lane_inventory()` discovers ordinary top-level Python modules
in the installed `analysis/` and `gauge/` packages. It compares that set with
the immutable reviewed rows. A new module without a row, or a row whose module
was removed, raises `RuntimeError` and names the exact drift.

```python
from scpn_quantum_control.analysis import assert_research_lane_inventory

report = assert_research_lane_inventory()
assert report.passed
assert not report.missing_modules
assert not report.orphaned_records
```

Only package `__init__.py` files and the registry implementation itself are
excluded. This keeps the gate focused on research lanes while avoiding a
self-registering governance row.

For CI or review, verify the committed deterministic artefacts:

```bash
python scripts/run_research_lane_registry.py --check
```

The command performs inventory validation before comparing bytes. It does not
load credentials, contact a provider, submit QPU work, or execute hardware.
The committed human-readable matrix is
`data/research_lane_registry/evidence.md`, paired with the machine-readable
JSON under the same directory.

## Claim boundary

Registry membership is catalogue metadata only. It does not grant
productisation, differentiability, control, hardware validity, advantage,
criticality, topology certification, consciousness interpretation, clinical
interpretation, or a publication claim.

For the six individually reviewed experimental theory hooks, also consult the
[Theory-Hook Promotion Matrix](theory_hook_promotion.md). For complete symbols,
exceptions, and serialization contracts, use the
[Research-lane Registry API](api/research_lane_registry.md).
