#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto handbook renderer
"""Render the public Kuramoto handbook from the live facade and benchmark artefact."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from scpn_quantum_control import kuramoto

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_BENCHMARK = REPO_ROOT / "docs" / "benchmarks" / "tiers" / "kuramoto_tiers.local.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "kuramoto_handbook.md"


class BenchmarkStats(TypedDict):
    """Latency statistics recorded for one benchmarked backend."""

    p50_us: float


class BackendRow(TypedDict):
    """One backend row from the tier-benchmark artefact."""

    backend: str
    status: str
    stats: BenchmarkStats | None


class BenchmarkResult(TypedDict):
    """One operation-size result from the tier-benchmark artefact."""

    operation: str
    size: int
    fastest_backend: str | None
    parity_max_abs_diff: float | None
    rows: list[BackendRow]


class BenchmarkArtifact(TypedDict):
    """Subset of the tier-benchmark artefact schema needed for the handbook."""

    environment: str
    generated_utc: str
    production_claim_allowed: bool
    provenance: Mapping[str, object]
    parameters: Mapping[str, object]
    results: list[BenchmarkResult]
    schema_version: str
    payload_sha256: NotRequired[str]


def load_benchmark_artifact(path: Path) -> BenchmarkArtifact:
    """Load the committed tier-benchmark artefact."""

    return cast(BenchmarkArtifact, json.loads(path.read_text(encoding="utf-8")))


def _symbol_list(symbols: Sequence[str]) -> str:
    """Format a stable comma-separated list of public symbols."""

    return ", ".join(f"`{symbol}`" for symbol in symbols)


def _rows_for_operation(results: Sequence[BenchmarkResult]) -> dict[str, list[BenchmarkResult]]:
    """Group benchmark rows by operation name."""

    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result["operation"], []).append(result)
    return grouped


def _p50(row: BackendRow | None) -> str:
    """Format a backend P50 value for the handbook table."""

    if row is None or row["stats"] is None:
        return "—"
    return f"{row['stats']['p50_us']:.3f}"


def _backend_rows(result: BenchmarkResult) -> dict[str, BackendRow]:
    """Index one result by backend name."""

    return {row["backend"]: row for row in result["rows"]}


def _string_sequence(value: object) -> Sequence[object]:
    """Return a JSON sequence value without treating strings as sequences."""

    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _representative_result(rows: Sequence[BenchmarkResult]) -> BenchmarkResult:
    """Choose a stable representative row for the operation summary table."""

    for target_size in (128, 32, 8):
        for row in rows:
            if row["size"] == target_size:
                return row
    return rows[0]


def _benchmark_table(artifact: BenchmarkArtifact) -> list[str]:
    """Render one row per benchmarked operation using a representative size."""

    grouped = _rows_for_operation(artifact["results"])
    lines = [
        "| Operation | Representative N | Rust p50 (µs) | Python p50 (µs) | Fastest backend | Parity max abs diff |",
        "|---|--:|--:|--:|---|--:|",
    ]
    for operation in sorted(grouped):
        result = _representative_result(grouped[operation])
        backends = _backend_rows(result)
        parity = (
            "—"
            if result["parity_max_abs_diff"] is None
            else f"{result['parity_max_abs_diff']:.2e}"
        )
        lines.append(
            f"| `{operation}` | {result['size']} | {_p50(backends.get('rust'))} "
            f"| {_p50(backends.get('python'))} | {result['fastest_backend'] or '—'} "
            f"| {parity} |"
        )
    return lines


def _capability_table(capabilities: Mapping[str, Sequence[str]]) -> list[str]:
    """Render the live Kuramoto facade capability groups."""

    lines = ["| Group | Count | Public symbols |", "|---|--:|---|"]
    for group, symbols in capabilities.items():
        lines.append(f"| `{group}` | {len(symbols)} | {_symbol_list(symbols)} |")
    return lines


def _benchmark_summary(artifact: BenchmarkArtifact) -> list[str]:
    """Render benchmark provenance and claim-boundary facts."""

    parameters = artifact["parameters"]
    provenance = artifact["provenance"]
    sizes = ", ".join(str(size) for size in _string_sequence(parameters.get("sizes", ())))
    tiers = ", ".join(str(tier) for tier in _string_sequence(parameters.get("tiers", ())))
    production = "yes" if artifact["production_claim_allowed"] else "no"
    return [
        f"- Schema: `{artifact['schema_version']}`.",
        f"- Environment: `{artifact['environment']}`.",
        f"- Generated UTC: `{artifact['generated_utc']}`.",
        f"- Commit: `{provenance.get('commit', 'unknown')}`.",
        f"- CPU model: `{provenance.get('cpu_model', 'unknown')}`.",
        f"- Python / NumPy: `{provenance.get('python', 'unknown')}` / `{provenance.get('numpy', 'unknown')}`.",
        f"- Rust engine: `{provenance.get('engine', 'unknown')}`.",
        f"- Tiers and sizes: `{tiers}` over N = `{sizes}`.",
        f"- Production performance claim allowed: `{production}`.",
        "- The side-by-side CI/local table is generated in "
        "[Multi-language Kuramoto tier benchmark](tier_benchmarks.md).",
    ]


def render_handbook(artifact: BenchmarkArtifact) -> str:
    """Render the complete Kuramoto handbook Markdown document."""

    capabilities = kuramoto.capabilities()
    capability_count = sum(len(symbols) for symbols in capabilities.values())
    benchmark_operations = len(_rows_for_operation(artifact["results"]))

    lines: list[str] = [
        "# SPDX-License-Identifier: AGPL-3.0-or-later",
        "# Commercial license available",
        "# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "# © Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "# ORCID: 0009-0009-3560-0851",
        "# Contact: www.anulum.li | protoscience@anulum.li",
        "# scpn-quantum-control — Kuramoto handbook",
        "",
        "<!-- Generated by tools/render_kuramoto_handbook.py — do not edit by hand. -->",
        "",
        "# Kuramoto Handbook",
        "",
        "This handbook is the reference page for the in-repository Kuramoto toolkit. It is",
        "generated from the live `scpn_quantum_control.kuramoto` facade and the committed",
        "multi-tier benchmark artefact, so the public API inventory, model families, and",
        "performance evidence stay aligned with the code.",
        "",
        f"The current facade exposes {capability_count} public symbols in {len(capabilities)}",
        f"capability groups. The committed tier benchmark covers {benchmark_operations}",
        "registered compute primitives across the Rust and Python tiers on the local",
        "workstation, with Julia and CI columns rendered when their artefacts exist.",
        "",
        '!!! info "This toolkit now ships as the standalone `oscillatools` distribution"',
        "    The coupled-phase-oscillator (Kuramoto) toolkit documented here has been",
        "    extracted into the standalone `oscillatools` distribution on a NumPy +",
        "    SciPy floor. Inside `scpn-quantum-control` the",
        "    `scpn_quantum_control.kuramoto` facade is now a re-export shim over",
        "    `oscillatools`, so this page mirrors the standalone API. New work should",
        "    depend on `oscillatools` directly; the canonical handbook ships with that",
        "    distribution.",
        "",
        "## Mathematical Core",
        "",
        "The base phase dynamics are",
        "",
        "$$",
        "\\dot\\theta_i = \\omega_i + \\sum_j K_{ij}\\sin(\\theta_j - \\theta_i).",
        "$$",
        "",
        "`omega_i` is the natural frequency of oscillator `i`, `K_ij` is the",
        "coupling from oscillator `j` to `i`, and the complex order parameter is",
        "",
        "$$",
        "Z = R e^{i\\psi} = \\frac{1}{N}\\sum_{j=1}^N e^{i\\theta_j}.",
        "$$",
        "",
        "The toolkit keeps this object explicit across every extension. Mean-field,",
        "networked, Sakaguchi, Daido, simplex, hyperedge, heterogeneous, delayed,",
        "noisy, inertial, plastic, and controlled variants all reduce to a named",
        "force field plus the Jacobian or adjoint needed by the analysis and control",
        "layers.",
        "",
        "## Model Families",
        "",
        "| Family | Equation or contract | Main API surface |",
        "|---|---|---|",
        "| Mean-field Kuramoto | `K r sin(psi - theta_i)` and its dense Jacobian. | `mean_field_force`, `mean_field_jacobian` |",
        "| Networked Kuramoto | `sum_j K_ij sin(theta_j - theta_i)` for dense coupling matrices. | `networked_kuramoto_force`, `networked_kuramoto_jacobian` |",
        "| Sakaguchi-Kuramoto | Frustrated coupling `sin(theta_j - theta_i - alpha)`. | `sakaguchi_force`, `sakaguchi_mean_field_force` |",
        "| Daido harmonics | Higher harmonic order parameter `r_m exp(i psi_m)`. | `daido_order_parameter`, `daido_mean_field_force` |",
        "| Triadic/simplex/hyperedge | Higher-order terms over simplex order or explicit hyperedge lists. | `simplex_mean_field_force`, `hyperedge_force`, `heterogeneous_force` |",
        "| Delayed synchronisation | Method-of-steps dynamics with branch-stability equations. | `integrate_delayed_kuramoto`, `synchronised_frequency_roots` |",
        "| Noisy mean-field | Euler-Maruyama trajectories and Fokker-Planck onset equations. | `integrate_noisy_kuramoto`, `noisy_critical_coupling` |",
        "| Inertial oscillators | Second-order phase-space dynamics with a mechanical-energy diagnostic. | `integrate_inertial`, `inertial_energy` |",
        "| Plastic/adaptive coupling | Co-evolving phase and coupling state with Hebbian equilibrium. | `integrate_adaptive_kuramoto`, `hebbian_coupling_equilibrium` |",
        "",
        "## Integrators and Adjoints",
        "",
        "The public facade exposes fixed-step Euler and RK4 trajectories, adaptive",
        "Dormand-Prince trajectories, delayed/inertial/noisy/adaptive variants, and",
        "reverse-mode adjoints for the differentiable Euler, RK4, and DOPRI paths.",
        "The adjoints return gradients with respect to the initial phases, natural",
        "frequencies, and coupling parameters where the corresponding mathematical",
        "contract is implemented. Unsupported routes fail closed rather than",
        "fabricating gradients.",
        "",
        "## Observables and Diagnostics",
        "",
        "- Global and local order parameters expose values, phases, gradients, and",
        "  Hessians where defined.",
        "- Interaction energy exposes value, gradient, and Hessian for optimisation",
        "  and terminal-control objectives.",
        "- Frequency-locking diagnostics report effective frequencies, synchronisation",
        "  index, locked fraction, and gradient of the terminal index.",
        "- Chimera and metastability diagnostics measure community order, spatial",
        "  incoherence, and temporal variability.",
        "- Phase-information diagnostics compute circular entropy, normalised entropy,",
        "  pairwise mutual information, and the full mutual-information matrix.",
        "- Coherence and clustering diagnostics expose phase-locking matrices, spectral",
        "  structure, leading coherence eigenvectors, and partial-synchronisation",
        "  partitions.",
        "",
        "## Analysis and Control",
        "",
        "- Stability analysis computes the locked-state Jacobian spectrum, removes the",
        "  Goldstone mode, and classifies synchronisation stability.",
        "- Critical-coupling routines implement Lorentzian and Gaussian closed forms",
        "  and self-consistent order-parameter branches.",
        "- Ott-Antonsen reduction integrates the two-dimensional reduced order",
        "  parameter and terminal sensitivities.",
        "- Lyapunov, continuation, hysteresis, and basin-estimation routines expose",
        "  transition and multistability structure.",
        "- Control routines optimise terminal coherence, phase-target objectives,",
        "  interaction-energy objectives, pinning gains, synchronising couplings, and",
        "  trajectory-matching system-identification losses.",
        "",
        "## Public Capability Index",
        "",
        *_capability_table(capabilities),
        "",
        "## Performance Evidence",
        "",
        *_benchmark_summary(artifact),
        "",
        "The table below uses one representative row per benchmarked operation. It",
        "does not replace the full raw artefact, which keeps every size, percentile,",
        "tier availability row, and parity record.",
        "",
        *_benchmark_table(artifact),
        "",
        "## First-Path Usage",
        "",
        "```python",
        "import numpy as np",
        "",
        "from scpn_quantum_control import kuramoto",
        "",
        "theta0 = np.array([0.0, 0.7, 1.6, 2.9], dtype=np.float64)",
        "omega = np.array([0.1, -0.2, 0.15, 0.05], dtype=np.float64)",
        "coupling = np.array(",
        "    [",
        "        [0.0, 0.6, 0.0, 0.2],",
        "        [0.6, 0.0, 0.5, 0.0],",
        "        [0.0, 0.5, 0.0, 0.4],",
        "        [0.2, 0.0, 0.4, 0.0],",
        "    ],",
        "    dtype=np.float64,",
        ")",
        "",
        "trajectory = kuramoto.kuramoto_rk4_trajectory(theta0, omega, coupling, 0.01, 64)",
        "diagnostics = kuramoto.frequency_order_diagnostics(trajectory, dt=0.01)",
        "value, gradient = kuramoto.synchronisation_value_and_grad(theta0, omega, coupling)",
        "```",
        "",
        "Use `kuramoto.capabilities()` to inspect the grouped API and",
        '`kuramoto.describe("analysis")` to list one group programmatically.',
        "",
        "## Worked Workflow",
        "",
        "Run the deterministic six-oscillator companion workflow when you need",
        "one executable Phase 5 path covering RK4 integration, frequency-locking",
        "diagnostics, stability spectrum, coherence clustering, Gaussian critical",
        "coupling, and projected synchronising-coupling design:",
        "",
        "```bash",
        "python examples/29_kuramoto_handbook_workflow.py",
        "```",
        "",
        "The matching notebook is",
        "`notebooks/48_kuramoto_handbook_workflow.ipynb`. Both surfaces use the",
        "public `scpn_quantum_control.kuramoto` facade and emit or preserve",
        "diagnostics that can be compared with this handbook's capability and",
        "benchmark tables.",
        "",
        "## Claim Boundaries",
        "",
        "- The local benchmark artefact is functional and reproducibility evidence, not",
        "  a production latency claim.",
        "- CI numbers are hosted-runner evidence. They are useful for drift detection",
        "  and side-by-side comparison, not a universal hardware claim.",
        "- Quantum hardware execution, broad quantum-advantage claims, and measured",
        "  physical `K_nm` magnitudes remain governed by the evidence ledgers and",
        "  preregistered campaign gates.",
        "- The coupled-phase-oscillator toolkit has been extracted into the standalone",
        "  `oscillatools` distribution (CEO/IP-approved 2026-07-04). Inside this repository",
        "  `scpn_quantum_control.kuramoto` is a re-export shim over `oscillatools`, so the",
        "  canonical source and documentation now ship with that distribution. See",
        "  [Kuramoto Standalone Package Decision](kuramoto_standalone_package_decision.md).",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the handbook and write it to disk."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_LOCAL_BENCHMARK,
        help="local tier-benchmark artefact JSON",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown output path")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    artifact = load_benchmark_artifact(args.benchmark)
    document = render_handbook(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"[render] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
