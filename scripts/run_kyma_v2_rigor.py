# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v2.1 supplementary-rigor runner
"""Run the four KYMA v2.1 supplementary analyses and write the combined artifact.

Ablations (#1), stronger baselines (#2), MLP convergence (#3), and leave-one-out
(#4), against the v2 frozen design (`KYMA_V2_1_SUPPLEMENTARY_RIGOR_PREREGISTRATION_7f6b_2026-07-21.md`).
Each stage is written to the artifact as it completes, so a partial run on a
memory-constrained host is still usable. 0 QPU.

Usage::

    python scripts/run_kyma_v2_rigor.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scpn_quantum_control.benchmarks.kyma_v2 import design, rigor

_DEFAULT_OUT = Path("data/kyma_v2_composition_probe/kyma_v2_1_rigor.json")


def main() -> None:
    """Run each rigor stage, writing the artifact incrementally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    config = design.select_config(seed=0)["config"]
    artifact: dict[str, object] = {
        "probe": "kyma_v2_1_supplementary_rigor",
        "pre_registration": "KYMA_V2_1_SUPPLEMENTARY_RIGOR_PREREGISTRATION_7f6b_2026-07-21.md",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "design_config": {
            "g_sync": config.g_sync,
            "steps": config.steps,
            "k_bridge": config.k_bridge,
        },
    }

    def flush() -> None:
        args.out.write_text(json.dumps(artifact, indent=2, default=float))

    flush()

    stages = (
        ("ablations", lambda: rigor.run_ablations(config)),
        ("stronger_baselines", lambda: rigor.run_stronger_baselines(config)),
        ("mlp_convergence", lambda: rigor.mlp_convergence(config)),
        ("leave_one_out", lambda: rigor.run_leave_one_out()),
    )
    for name, fn in stages:
        t0 = time.perf_counter()
        artifact[name] = fn()
        artifact[f"{name}_seconds"] = round(time.perf_counter() - t0, 1)
        flush()
        print(f"[{name}] done in {artifact[f'{name}_seconds']}s", flush=True)

    print(f"artifact → {args.out}", flush=True)


if __name__ == "__main__":
    main()
