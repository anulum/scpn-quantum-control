# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA v3 symbolic composition probe runner
"""Run the KYMA v3 symbolic composition probe and write the artefact.

The contract, seeds, epochs and learning rates are frozen in
``docs/campaigns/kyma_v3_symbolic_composition_prereg_2026-09-29.md``; this runner
only executes them. The artefact records the host, the source commit supplied by
the caller and the declared nominal package power used for the energy proxy.
0 QPU — this is a classical oscillator-substrate probe.

Usage::

    python scripts/run_kyma_v3_probe.py --nominal-power-w WATTS --commit SHA [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from scpn_quantum_control.benchmarks.kyma_v3 import probe

_DEFAULT_OUT = Path("data/kyma_v3_symbolic_composition/kyma_v3_symbolic_composition.json")
_PREREGISTRATION = "docs/campaigns/kyma_v3_symbolic_composition_prereg_2026-09-29.md"


def main(argv: list[str] | None = None) -> int:
    """Run the frozen probe and write the artefact.

    Parameters
    ----------
    argv
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code (0).

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-power-w", type=float, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(probe.SEEDS))
    parser.add_argument("--epochs", type=int, default=probe.EPOCHS)
    args = parser.parse_args(argv)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result = probe.run_probe(tuple(args.seeds), args.epochs, args.nominal_power_w)
    artefact = {
        "probe": "kyma_v3_symbolic_composition",
        "pre_registration": _PREREGISTRATION,
        "source_commit": args.commit,
        "host": {
            "node": platform.node(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        },
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artefact, indent=2, default=float) + "\n", encoding="utf-8")
    verdict = result["verdict"]
    print(
        f"pass={verdict['pass']} substrate={verdict['substrate_mean']:.3f}±{verdict['substrate_sd']:.3f} "
        f"best_contract={verdict['best_contract_baseline']} margin={verdict['margin_over_best_contract']:+.3f} "
        f"chance={verdict['chance_floor']:.3f}"
    )
    print(f"artefact -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
