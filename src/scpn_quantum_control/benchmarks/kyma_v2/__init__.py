# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v2 corrected-design composition probe
"""KYMA v2 compositional-generalisation probe (corrected design).

v2 answers the two defects diagnosed by the v1 NEGATIVE
(``KYMA_TOY_PROBE_PREREGISTRATION_7f6b_2026-07-18.md``, commit ``2f67de12``,
kept as the honest baseline):

1. **Coupling gating** — the control code gates the coupling ``K``, not just the
   frequency drive, so one substrate physically realises in-phase *and*
   anti-phase motifs at once (:mod:`.coupling`, :mod:`.teacher`).
2. **Non-separable, data-dependent readout** — a fixed ambient coupling makes the
   joint state non-factorisable and the label is a quantised *achieved* phase, so
   the answer depends on ``θ0`` and on the interaction of both relations; a
   param-matched MLP cannot compose it by learning each relation alone.

The design constants are fixed by a mechanism-only sanity check (:mod:`.design`,
teacher dynamics only) *before* any model is trained, and the pass/fail contract
is frozen in ``KYMA_V2_PROBE_PREREGISTRATION_7f6b_2026-07-21.md``.
"""

from __future__ import annotations

from .task import ProbeConfigV2, TrialBatchV2, build_trials

__all__ = ["ProbeConfigV2", "TrialBatchV2", "build_trials"]
