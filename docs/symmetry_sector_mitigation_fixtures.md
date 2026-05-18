<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- symmetry-sector mitigation fixtures -->

# Symmetry-Sector Mitigation Planner Fixtures

These no-QPU fixtures lock the planner contract before execution-path integration.

| Fixture | Status | Primitives | Blockers |
|---|---|---|---|
| `eligible_counts_guess` | `eligible` | parity_postselection, symmetry_expansion, guess_symmetry_decay | none |
| `blocked_missing_counts` | `blocked` | none | raw measurement counts are required before mitigation planning |
| `blocked_missing_guess_observables` | `blocked` | none | GUESS requires noise-scaled symmetry observables |
| `blocked_nonsymmetric_coupling` | `blocked` | none | coupling_matrix must be symmetric for XY parity-sector planning |

## Raw-count replay fixtures

| Fixture | Status | Applied primitives | Deferred primitives | Blockers |
|---|---|---|---|---|
| `replay_counts_postselection_expansion` | `applied` | parity_postselection, symmetry_expansion | guess_symmetry_decay | GUESS replay requires calibrated noise-scaled symmetry observable rows |
| `replay_blocked_missing_counts` | `blocked` | none | none | symmetry-sector plan is blocked: ('raw measurement counts are required before mitigation planning',) |

## Reproducibility gate

Regenerate and compare the fixtures with:

```bash
scpn-bench symmetry-sector-mitigation-gate
```

## Claim boundary

Planner fixtures prove deterministic eligibility/blocker outputs. Replay fixtures prove offline raw-count accounting. They do not mutate circuits, submit hardware jobs, or prove hardware improvement.
