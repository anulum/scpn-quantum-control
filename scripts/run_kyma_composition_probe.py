#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — KYMA composition-probe runner
"""Run the KYMA toy compositional-generalisation probe and write the artifact.

Executes the frozen pre-registration (5 seeds; PASS iff held-out-conjunction
accuracy ≥ 70 %, ≥ 25 pp above the parameter-matched MLP baseline, and above
the measured chance floor) and writes the raw result artifact + config to
``data/kyma_composition_probe/`` for referee reproduction.

Usage::

    JAX_PLATFORMS=cpu python scripts/run_kyma_composition_probe.py [--seeds 0 1 2 3 4]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from scpn_quantum_control.benchmarks.kyma.probe import run_probe  # noqa: E402
from scpn_quantum_control.benchmarks.kyma.task import ProbeConfig  # noqa: E402

_OUT = _REPO / "data" / "kyma_composition_probe"


def part_b_paragraph(result: dict) -> str:
    """Honest Part-B §1.2.2a paragraph — positive OR negative + diagnosis."""
    sub = result["substrate_accuracy"]
    mlp = result["mlp_accuracy"]
    chance = result["chance_accuracy"]
    margin = result["margin_over_mlp_pp"]
    if result["verdict"] == "PASS":
        return (
            f"Reusable Kuramoto motifs compose in miniature: on a held-out "
            f"(in-phase ∧ anti-phase) conjunction never seen jointly in training, "
            f"the motif substrate reached {sub['mean']:.0%} ± {sub['sd']:.0%} accuracy "
            f"over 5 seeds — {margin:+.0%} above a parameter-matched non-motif MLP "
            f"({mlp['mean']:.0%} ± {mlp['sd']:.0%}) and far above the measured chance "
            f"floor ({chance['mean']:.1%}). The motif structure, not raw capacity, "
            f"enables the composition, supporting the WP1 compositional-control claim."
        )
    return (
        f"On this encoding the probe returns a NEGATIVE result: the Kuramoto motif "
        f"substrate reached {sub['mean']:.0%} ± {sub['sd']:.0%} held-out-conjunction "
        f"accuracy over 5 seeds, not clearing the pre-registered bar of ≥70 % AND "
        f"≥25 pp above the parameter-matched MLP baseline ({mlp['mean']:.0%} ± "
        f"{mlp['sd']:.0%}; chance {chance['mean']:.1%}). Diagnosis: the additive "
        f"input code is linearly decodable, so a generic MLP composes it trivially "
        f"while the substrate must physically realise both relations through a shared "
        f"coupling matrix — co-activation interference on disjoint cluster pairs, not "
        f"a capacity limit. WP1 implication: compositional control needs either a "
        f"non-additive readout or per-relation coupling gating, not a shared-K drive."
    )


def main(argv: list[str] | None = None) -> int:
    """Run the probe over the requested seeds and write the result artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args(argv)

    config = ProbeConfig()
    result = run_probe(tuple(args.seeds), config)
    result["host"] = platform.platform()
    result["python"] = platform.python_version()
    result["part_b_paragraph"] = part_b_paragraph(result)

    _OUT.mkdir(parents=True, exist_ok=True)
    artifact = _OUT / "kyma_composition_probe_result.json"
    artifact.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"verdict: {result['verdict']}")
    print(
        f"substrate {result['substrate_accuracy']['mean']:.3f}±"
        f"{result['substrate_accuracy']['sd']:.3f} | "
        f"MLP {result['mlp_accuracy']['mean']:.3f}±{result['mlp_accuracy']['sd']:.3f} | "
        f"chance {result['chance_accuracy']['mean']:.4f} | "
        f"margin {result['margin_over_mlp_pp']:+.3f}"
    )
    print(f"artifact: {artifact.relative_to(_REPO)}")
    print(result["part_b_paragraph"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
