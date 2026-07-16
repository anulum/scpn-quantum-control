#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — CHSH recomputation from the committed Bell artifact
"""Recompute the CHSH statistics from the committed Bell-test counts.

Reproduces, from ``results/ibm_hardware_2026-03-28/bell_test_4q.json`` alone,
the per-pair CHSH value ``S`` with its multinomial standard error and the
significance of the classical-bound violation. Pure counts arithmetic — no
quantum SDK is imported, so anybody with the repository can verify the
published numbers in milliseconds.

Background (dated amendment 2026-07-16): the repository previously advertised
the Bell violation as ``>8σ`` for both pairs. This script shows the correct
per-pair attribution — the ``S = 2.165`` pair (qubits 0–1) violates at
``7.54σ`` and the ``S = 2.188`` pair (qubits 2–3) at ``8.94σ``. Only the
higher pair clears 8σ. It also surfaces the anomalous second setting
(``pub_index`` 1), whose correlators (≈+0.29/+0.33) sit far below the other
three settings (≈0.80–0.86) yet still enter ``S`` with the standard minus
sign; see ``docs/results.md`` for the re-run plan.

Conventions encoded here, matching the executed experiment:

* Bitstrings are little-endian (Qiskit): the rightmost character is qubit 0.
* Each of the four ``pub_index`` entries is one CHSH analyser setting; the
  combination is ``S = E₀ − E₁ + E₂ + E₃`` (the minus sign sits on setting 1).
* ``E = P(same) − P(different)`` over the two bits of the pair.
* ``σ_S = sqrt(Σᵢ (1 − Eᵢ²)/Nᵢ)`` — first-order multinomial error
  propagation; significance is ``(S − 2)/σ_S``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ARTIFACT = REPO_ROOT / "results" / "ibm_hardware_2026-03-28" / "bell_test_4q.json"

#: The four analyser settings enter S with these signs (minus on setting 1).
SETTING_SIGNS: tuple[int, int, int, int] = (1, -1, 1, 1)

#: Measured qubit pairs, as little-endian qubit indices.
QUBIT_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (2, 3))

CLASSICAL_BOUND = 2.0


@dataclass(frozen=True)
class PairStatistics:
    """CHSH statistics for one measured qubit pair."""

    label: str
    settings_e: tuple[float, ...]
    s_value: float
    sigma: float

    @property
    def significance(self) -> float:
        """Violation of the classical bound in units of the standard error.

        Deterministic counts give ``σ = 0``; the significance is then ±∞ for
        any non-zero excess and 0 at the bound, instead of dividing by zero.
        """
        excess = self.s_value - CLASSICAL_BOUND
        if self.sigma == 0.0:
            if excess == 0.0:
                return 0.0
            return math.inf if excess > 0.0 else -math.inf
        return excess / self.sigma


def pair_correlator(counts: Mapping[str, int], pair: tuple[int, int]) -> tuple[float, int]:
    """Return ``(E, N)`` for one qubit pair from one setting's counts.

    ``E = P(same) − P(different)`` over the two little-endian bit positions.
    """
    same = 0
    different = 0
    for bitstring, count in counts.items():
        width_needed = max(pair) + 1
        if len(bitstring) < width_needed:
            raise ValueError(f"bitstring {bitstring!r} narrower than qubit pair {pair} requires")
        first = bitstring[-1 - pair[0]]
        second = bitstring[-1 - pair[1]]
        if first == second:
            same += count
        else:
            different += count
    total = same + different
    if total <= 0:
        raise ValueError(f"no shots recorded for qubit pair {pair}")
    return (same - different) / total, total


def chsh_for_pair(
    results: Sequence[Mapping[str, object]], pair: tuple[int, int]
) -> PairStatistics:
    """Compute S ± σ for one qubit pair across the four analyser settings."""
    if len(results) != len(SETTING_SIGNS):
        raise ValueError(f"expected {len(SETTING_SIGNS)} analyser settings, found {len(results)}")
    settings_e: list[float] = []
    variance = 0.0
    s_value = 0.0
    for sign, result in zip(SETTING_SIGNS, results, strict=True):
        counts = result.get("counts")
        if not isinstance(counts, Mapping):
            raise ValueError("setting entry lacks a 'counts' mapping")
        e_value, total = pair_correlator(counts, pair)
        settings_e.append(e_value)
        s_value += sign * e_value
        variance += (1.0 - e_value * e_value) / total
    return PairStatistics(
        label=f"q{pair[0]}q{pair[1]}",
        settings_e=tuple(settings_e),
        s_value=s_value,
        sigma=math.sqrt(variance),
    )


def recompute(artifact: Mapping[str, object]) -> tuple[PairStatistics, ...]:
    """Recompute CHSH statistics for every measured pair in the artifact."""
    results = artifact.get("results")
    if not isinstance(results, list):
        raise ValueError("artifact lacks a 'results' list")
    return tuple(chsh_for_pair(results, pair) for pair in QUBIT_PAIRS)


def render_report(pairs: Sequence[PairStatistics]) -> str:
    """Render the human-readable verification report."""
    lines = ["CHSH recomputation from committed counts (pure arithmetic):"]
    for stats in pairs:
        settings = ", ".join(f"{e:+.4f}" for e in stats.settings_e)
        lines.append(
            f"  pair {stats.label}: E per setting [{settings}] -> "
            f"S = {stats.s_value:.4f} +/- {stats.sigma:.4f} "
            f"({stats.significance:.2f} sigma above the classical bound)"
        )
    lines.append(
        "  note: setting index 1 (minus sign in S) is anomalous on this run "
        "(E well below the other three settings); see docs/results.md."
    )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ARTIFACT,
        help="Path to the committed Bell-test artifact JSON.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write the recomputed statistics as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    pairs = recompute(artifact)
    print(render_report(pairs))
    if args.json is not None:
        payload = {
            stats.label: {
                "settings_e": list(stats.settings_e),
                "s_value": stats.s_value,
                "sigma": stats.sigma,
                "significance": stats.significance,
            }
            for stats in pairs
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
