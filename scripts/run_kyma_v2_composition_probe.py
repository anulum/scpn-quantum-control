# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v2 composition probe runner
"""Run the KYMA v2 compositional-generalisation probe and write the artifact.

Two-stage, honesty-gated (v2 pre-registration
``KYMA_V2_PROBE_PREREGISTRATION_7f6b_2026-07-21.md``):

1. **Mechanism-only design** (:mod:`...kyma_v2.design`) fixes ``g_sync``,
   ``(dt, steps)`` and ``k_ambient`` from teacher dynamics alone — no model, no
   test-accuracy peeking — and records the realisability, non-separability rate
   and class histogram.
2. **Probe** (:mod:`...kyma_v2.probe`) trains the gated student and the
   parameter-matched MLP against the frozen constants and evaluates the frozen
   pass/fail contract over five seeds.

The design block and the pass/fail block are both written to the artifact so the
run is independently reproducible. 0 QPU — this is a classical Kuramoto probe.

Usage::

    python scripts/run_kyma_v2_composition_probe.py [--seeds 0 1 2 3 4] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from scpn_quantum_control.benchmarks.kyma_v2 import design, probe

_DEFAULT_OUT = Path("data/kyma_v2_composition_probe/kyma_v2_composition_probe.json")


def main() -> None:
    """Fix the design, run the probe, and write the combined artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(probe.DEFAULT_SEEDS))
    parser.add_argument("--design-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Stage 1 — mechanism-only design selection (teacher dynamics only).
    selection = design.select_config(seed=args.design_seed)
    config = selection.pop("config")

    # Stage 2 — probe under the frozen constants.
    result = probe.run_probe(tuple(args.seeds), config)

    artifact = {
        "probe": "kyma_v2_composition",
        "pre_registration": "KYMA_V2_PROBE_PREREGISTRATION_7f6b_2026-07-21.md",
        "baseline_v1": "KYMA_TOY_PROBE_PREREGISTRATION_7f6b_2026-07-18.md (commit 2f67de12)",
        "started_utc": started,
        "design": {"design_seed": args.design_seed, **selection, "config": asdict(config)},
        "result": result,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, default=float))
    print(
        f"verdict={result['verdict']} "
        f"student={result['student_accuracy']['mean']:.3f} "
        f"mlp={result['mlp_accuracy']['mean']:.3f} "
        f"chance={result['chance_accuracy']['mean']:.3f} "
        f"non_sep={selection['non_separability_rate']:.3f}"
    )
    print(f"artifact → {args.out}")


if __name__ == "__main__":
    main()
