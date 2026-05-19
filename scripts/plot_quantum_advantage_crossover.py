# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Quantum Advantage Crossover Plotter
"""Plot the exact-simulation crossover anchored by committed hardware runs.

The figure is deliberately conservative: it compares exact Hilbert-space
simulation wall time against the measured/estimated QPU budget for completed
ibm_fez Kuramoto runs, and includes the Rust Kuramoto ODE path as a guardrail.
It does not claim broad quantum advantage in the measured n <= 16 window.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PUBLICATION_DIR = REPO_ROOT / "figures" / "publication"
DOCS_PUBLICATION_DIR = REPO_ROOT / "docs" / "figures" / "publication"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _dependency_versions() -> dict[str, str]:
    dependency_versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "qiskit": "not installed",
    }
    try:
        dependency_versions["qiskit"] = version("qiskit")
    except PackageNotFoundError:
        dependency_versions["qiskit"] = "not installed"

    return dependency_versions


def _table_row_provenance() -> dict[str, object]:
    return {
        "backend": "python-scipy-qiskit-local",
        "machine": platform.platform(),
        "command": "python scripts/plot_quantum_advantage_crossover.py",
        "dependency": _dependency_versions(),
        "git_commit": _git_commit(),
    }


@dataclass(frozen=True)
class HardwarePoint:
    """One completed IBM hardware scaling point used by the crossover figure."""

    n_qubits: int
    backend: str
    machine: str
    command: str
    dependency: dict[str, str]
    git_commit: str
    job_id: str
    shots: int
    depth: int
    hw_r: float
    qpu_budget_ms: float
    source_file: str


@dataclass(frozen=True)
class ClassicalPoint:
    """Classical baseline for the same n-qubit Kuramoto-XY family."""

    n_qubits: int
    hilbert_dim: int
    backend: str
    machine: str
    command: str
    dependency: dict[str, str]
    git_commit: str
    ode_ms: float
    exact_diag_ms: float | None
    exact_mem_mb: float
    note: str


@dataclass(frozen=True)
class LogFit:
    """Log-linear scaling fit y = 10 ** (intercept + slope * x)."""

    slope: float
    intercept: float
    r_squared: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted log-linear model at the supplied x values."""
        return np.power(10.0, self.intercept + self.slope * x)


@dataclass(frozen=True)
class PowerFit:
    """Power-law fit y = 10 ** intercept * n ** slope."""

    slope: float
    intercept: float
    r_squared: float

    def predict(self, n_qubits: np.ndarray) -> np.ndarray:
        """Evaluate the fitted power-law model at the supplied qubit counts."""
        return np.power(10.0, self.intercept) * np.power(n_qubits, self.slope)


HARDWARE_SPECS = (
    # qpu_budget_ms values come from results/HARDWARE_RESULTS.md,
    # "QPU Budget Accounting". Depth fallbacks come from the same table.
    ("hw_kuramoto_4osc_20k.json", 4, 149, 20_000.0),
    ("hw_kuramoto_6osc.json", 6, 147, 20_000.0),
    ("hw_kuramoto_8osc_20k.json", 8, 233, 20_000.0),
    ("hw_kuramoto_10osc.json", 10, 395, 25_000.0),
    ("hw_kuramoto_12osc.json", 12, 469, 30_000.0),
    ("hw_kuramoto_14osc.json", 14, 747, 20_000.0),
    ("hw_upde_16_snapshot.json", 16, 770, 60_000.0),
)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _metadata_depth(payload: dict, fallback_depth: int) -> int:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("depth") is not None:
        return int(metadata["depth"])
    return fallback_depth


def load_hardware_points(results_dir: Path = RESULTS_DIR) -> list[HardwarePoint]:
    """Load and validate the committed IBM hardware scaling series."""

    points: list[HardwarePoint] = []
    row_provenance = _table_row_provenance()
    for filename, expected_n, fallback_depth, qpu_budget_ms in HARDWARE_SPECS:
        payload = _read_json(results_dir / filename)
        n_qubits = int(payload["n_qubits"])
        if n_qubits != expected_n:
            msg = f"{filename} has n_qubits={n_qubits}, expected {expected_n}"
            raise ValueError(msg)
        backend = str(payload["backend"])
        if backend != "ibm_fez":
            msg = f"{filename} backend={backend}, expected ibm_fez"
            raise ValueError(msg)
        job_id = str(payload["job_id"])
        if not job_id.startswith("ibm-run-"):
            msg = f"{filename} job_id={job_id!r} is not a public IBM run label"
            raise ValueError(msg)
        points.append(
            HardwarePoint(
                n_qubits=n_qubits,
                backend=backend,
                machine=str(row_provenance["machine"]),
                command=str(row_provenance["command"]),
                dependency=dict(row_provenance["dependency"]),
                git_commit=str(row_provenance["git_commit"]),
                job_id=job_id,
                shots=int(payload["shots"]),
                depth=_metadata_depth(payload, fallback_depth),
                hw_r=float(payload["hw_R"]),
                qpu_budget_ms=float(qpu_budget_ms),
                source_file=f"results/{filename}",
            )
        )

    return sorted(points, key=lambda point: point.n_qubits)


def load_classical_points(
    path: Path = RESULTS_DIR / "classical_baselines_2026-03-30.json",
) -> list[ClassicalPoint]:
    """Load exact-diagonalisation and ODE baseline timings."""

    payload = _read_json(path)
    points = []
    row_provenance = _table_row_provenance()
    for n_key, entry in payload.items():
        exact_diag = entry.get("exact_diag_ms")
        points.append(
            ClassicalPoint(
                n_qubits=int(n_key),
                hilbert_dim=int(entry["dim"]),
                backend=str(row_provenance["backend"]),
                machine=str(row_provenance["machine"]),
                command=str(row_provenance["command"]),
                dependency=dict(row_provenance["dependency"]),
                git_commit=str(row_provenance["git_commit"]),
                ode_ms=float(entry["classical_ode_ms"]),
                exact_diag_ms=None if exact_diag is None else float(exact_diag),
                exact_mem_mb=float(entry["exact_diag_mem_mb"]),
                note=str(entry.get("note", "")),
            )
        )
    return sorted(points, key=lambda point: point.n_qubits)


def fit_exact_classical(points: list[ClassicalPoint]) -> LogFit:
    """Fit exponential exact-diagonalisation scaling from finite timings."""

    finite = [p for p in points if p.exact_diag_ms is not None and p.exact_diag_ms > 0]
    if len(finite) < 3:
        raise ValueError("Need at least three finite exact-diagonalisation timings")
    x = np.array([p.n_qubits for p in finite], dtype=float)
    log_y = np.log10([p.exact_diag_ms for p in finite])
    slope, intercept = np.polyfit(x, log_y, 1)
    predicted = intercept + slope * x
    ss_res = float(np.sum((log_y - predicted) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return LogFit(slope=float(slope), intercept=float(intercept), r_squared=r_squared)


def fit_hardware_budget(points: list[HardwarePoint]) -> PowerFit:
    """Fit a conservative power-law envelope through hardware QPU budgets."""

    n = np.array([p.n_qubits for p in points], dtype=float)
    log_n = np.log10(n)
    log_y = np.log10([p.qpu_budget_ms for p in points])
    slope, intercept = np.polyfit(log_n, log_y, 1)
    predicted = intercept + slope * log_n
    ss_res = float(np.sum((log_y - predicted) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return PowerFit(slope=float(slope), intercept=float(intercept), r_squared=r_squared)


def estimate_crossover_qubits(exact_fit: LogFit, hardware_fit: PowerFit) -> float:
    """Estimate where exact simulation time crosses the QPU budget envelope."""

    grid = np.linspace(4.0, 40.0, 7201)
    delta = exact_fit.predict(grid) - hardware_fit.predict(grid)
    crossed = np.flatnonzero(delta >= 0.0)
    if crossed.size == 0:
        return float("nan")
    idx = int(crossed[0])
    if idx == 0:
        return float(grid[0])
    x0, x1 = grid[idx - 1], grid[idx]
    y0, y1 = delta[idx - 1], delta[idx]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def build_crossover_model(
    hardware_points: list[HardwarePoint] | None = None,
    classical_points: list[ClassicalPoint] | None = None,
) -> tuple[LogFit, PowerFit, float]:
    """Return fitted models and their first crossover point."""

    hardware = hardware_points if hardware_points is not None else load_hardware_points()
    classical = classical_points if classical_points is not None else load_classical_points()
    exact_fit = fit_exact_classical(classical)
    hardware_fit = fit_hardware_budget(hardware)
    crossover = estimate_crossover_qubits(exact_fit, hardware_fit)
    return exact_fit, hardware_fit, crossover


def plot_quantum_advantage_crossover(
    output_dir: Path = PUBLICATION_DIR,
    docs_output_dir: Path | None = DOCS_PUBLICATION_DIR,
) -> tuple[Path, Path, float]:
    """Generate the PNG/PDF crossover figure and return output paths."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hardware = load_hardware_points()
    classical = load_classical_points()
    exact_fit, hardware_fit, crossover = build_crossover_model(hardware, classical)

    n_grid = np.linspace(4.0, 40.0, 500)
    exact_curve = exact_fit.predict(n_grid) / 1000.0
    hardware_curve = hardware_fit.predict(n_grid) / 1000.0

    finite_exact = [p for p in classical if p.exact_diag_ms is not None]
    oom_exact = [p for p in classical if p.exact_diag_ms is None]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        n_grid,
        exact_curve,
        color="#b3261e",
        linewidth=2.2,
        label=f"Exact classical fit (R^2={exact_fit.r_squared:.2f})",
    )
    ax.plot(
        n_grid,
        hardware_curve,
        color="#1f6f8b",
        linewidth=2.0,
        linestyle="--",
        label="QPU budget envelope",
    )

    ax.scatter(
        [p.n_qubits for p in finite_exact],
        [float(p.exact_diag_ms) / 1000.0 for p in finite_exact],
        color="#d64545",
        edgecolor="black",
        linewidth=0.6,
        s=54,
        zorder=4,
        label="Measured exact diagonalisation",
    )
    ax.scatter(
        [p.n_qubits for p in hardware],
        [p.qpu_budget_ms / 1000.0 for p in hardware],
        color="#2b7a9b",
        marker="s",
        edgecolor="black",
        linewidth=0.6,
        s=48,
        zorder=5,
        label="Completed ibm_fez runs",
    )
    ax.scatter(
        [p.n_qubits for p in classical],
        [p.ode_ms / 1000.0 for p in classical],
        color="#4d7c0f",
        marker="^",
        edgecolor="black",
        linewidth=0.5,
        s=42,
        zorder=4,
        label="Rust Kuramoto ODE baseline",
    )

    if oom_exact:
        memory_text = "Exact diag OOM at 14q/16q\nmemory estimates: 2.1/33.6 GiB"
        ax.text(
            15.2,
            0.075,
            memory_text,
            fontsize=7.4,
            bbox={"facecolor": "white", "edgecolor": "0.78", "boxstyle": "round,pad=0.24"},
        )

    if np.isfinite(crossover):
        cross_y = float(exact_fit.predict(np.array([crossover]))[0] / 1000.0)
        ax.axvline(crossover, color="0.25", linestyle=":", linewidth=1.2)
        ax.scatter([crossover], [cross_y], color="black", s=36, zorder=6)
        ax.annotate(
            f"Exact-simulation crossover\nn approx {crossover:.1f}",
            (crossover, cross_y),
            xytext=(10, -36),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "color": "0.25", "linewidth": 0.8},
        )

    ax.axvspan(14, 40, color="#f5c542", alpha=0.11, label="Exact diag memory wall")
    ax.text(
        17.0,
        0.001,
        "No broad quantum advantage is demonstrated at n <= 16;\n"
        "crossover applies to exact Hilbert-space simulation only.",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.75", "boxstyle": "round,pad=0.28"},
    )

    ax.set_yscale("log")
    ax.set_xlim(3.5, 40.5)
    ax.set_ylim(0.0002, 3e6)
    ax.set_xlabel("System size (qubits / oscillators)")
    ax.set_ylabel("Wall-clock or QPU budget time (s)")
    ax.set_title("Kuramoto-XY exact-simulation crossover anchored by ibm_fez hardware")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper left", fontsize=7.6, framealpha=0.94)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fig17_quantum_advantage_crossover.png"
    pdf_path = output_dir / "fig17_quantum_advantage_crossover.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    if docs_output_dir is not None:
        docs_output_dir.mkdir(parents=True, exist_ok=True)
        docs_png = docs_output_dir / png_path.name
        docs_png.write_bytes(png_path.read_bytes())

    return png_path, pdf_path, crossover


def main() -> int:
    """Render the quantum-advantage crossover figure and print its artefacts."""
    png_path, pdf_path, crossover = plot_quantum_advantage_crossover()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Exact-simulation crossover qubits: {crossover:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
