#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — generate s1 feedback paper assets script
# scpn-quantum-control -- S1 monitored-feedback paper assets
"""Generate publication figures for the S1 monitored-feedback paper."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "s1_feedback_loop"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures" / "s1_feedback_control"
DATE_STAMP = "2026-05-20"

OBSERVABLES = ("IXX", "IYY", "XXI", "YYI", "XYI", "YXI", "ZZI")
RUN_ROWS = (
    ("S1b original", "s1b_xy_observable_analysis_{backend}_{stamp_s1b}.json", None),
    ("S1c shallow", "s1c_xy_observable_analysis_{backend}_{stamp_s1c}.json", None),
    (
        "S1d current",
        "s1d_xy_observable_analysis_{backend}_{stamp_s1d}.json",
        "current_shallow_positive",
    ),
    (
        "S1d flipped",
        "s1d_xy_observable_analysis_{backend}_{stamp_s1d}.json",
        "polarity_flipped",
    ),
    ("S1d weak", "s1d_xy_observable_analysis_{backend}_{stamp_s1d}.json", "weak_positive"),
    (
        "S1e current",
        "s1e_xy_observable_analysis_{backend}_{stamp_s1e}.json",
        "current_shallow_positive",
    ),
    (
        "S1e flipped",
        "s1e_xy_observable_analysis_{backend}_{stamp_s1e}.json",
        "polarity_flipped",
    ),
    ("S1e weak", "s1e_xy_observable_analysis_{backend}_{stamp_s1e}.json", "weak_positive"),
    ("S1f quadrature", "s1f_xy_observable_analysis_{backend}_{stamp_s1f}.json", None),
)
BACKEND_RUNS = (
    {
        "label": "Kingston",
        "backend": "ibm_kingston",
        "stamp_s1b": "20260520T130238Z",
        "stamp_s1c": "20260520T132455Z",
        "stamp_s1d": "20260520T134614Z",
        "stamp_s1e": "20260520T140919Z",
        "stamp_s1f": "20260520T142213Z",
    },
    {
        "label": "Fez",
        "backend": "ibm_fez",
        "stamp_s1b": "20260520T145259Z",
        "stamp_s1c": "20260520T145406Z",
        "stamp_s1d": "20260520T145513Z",
        "stamp_s1e": "20260520T145901Z",
        "stamp_s1f": "20260520T150249Z",
    },
    {
        "label": "Marrakesh",
        "backend": "ibm_marrakesh",
        "stamp_s1b": "20260520T150538Z",
        "stamp_s1c": "20260520T150716Z",
        "stamp_s1d": "20260520T150821Z",
        "stamp_s1e": "20260520T151122Z",
        "stamp_s1f": "20260520T151450Z",
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _observable_deltas(payload: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(row["basis"]): float(row["feedback_minus_control"]) for row in payload["observables"]
    }


def _variant_deltas(payload: Mapping[str, Any], variant: str) -> dict[str, float]:
    for row in payload["policy_variants"]:
        if row["policy_variant"] == variant:
            return _observable_deltas(row)
    raise ValueError(f"policy variant not present: {variant}")


def build_matrix(
    data_dir: Path, backend_run: Mapping[str, str]
) -> tuple[list[str], list[str], np.ndarray]:
    """Build feedback-minus-control observable matrix for S1b--S1f."""

    labels: list[str] = []
    matrix: list[list[float]] = []
    cache: dict[str, dict[str, Any]] = {}

    for row_label, filename_template, variant in RUN_ROWS:
        filename = filename_template.format(**backend_run)
        payload = cache.setdefault(filename, _read_json(data_dir / filename))
        deltas = _variant_deltas(payload, variant) if variant else _observable_deltas(payload)
        labels.append(row_label)
        matrix.append([deltas.get(observable, np.nan) for observable in OBSERVABLES])

    return labels, list(OBSERVABLES), np.array(matrix, dtype=float)


def write_single_backend_heatmap(
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    values: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    """Write PNG and PDF heatmaps for the S1 direct-observable extensions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    masked = np.ma.masked_invalid(values)
    max_abs = float(np.nanmax(np.abs(values)))

    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    image = ax.imshow(masked, cmap=cmap, vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_xlabel("Measured Pauli observable")
    ax.set_ylabel("S1 extension or policy")
    ax.set_title("Feedback minus matched-control Pauli expectation")

    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if np.isnan(value):
                ax.text(col_index, row_index, "n/a", ha="center", va="center", fontsize=7)
            else:
                colour = "white" if abs(value) > 0.55 * max_abs else "black"
                ax.text(
                    col_index,
                    row_index,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=7,
                )

    cbar = fig.colorbar(image, ax=ax, shrink=0.88)
    cbar.set_label(
        r"$\Delta_P=\langle P\rangle_{\mathrm{feedback}}-\langle P\rangle_{\mathrm{control}}$"
    )

    png = output_dir / f"s1_feedback_delta_heatmap_{DATE_STAMP}.png"
    pdf = output_dir / f"s1_feedback_delta_heatmap_{DATE_STAMP}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def write_multibackend_heatmap(
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    backend_values: Sequence[tuple[str, np.ndarray]],
    output_dir: Path,
) -> list[Path]:
    """Write a three-backend heatmap panel for the S1 extension stack."""

    output_dir.mkdir(parents=True, exist_ok=True)
    max_abs = max(float(np.nanmax(np.abs(values))) for _, values in backend_values)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")

    fig, axes = plt.subplots(
        1,
        len(backend_values),
        figsize=(14.5, 6.1),
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for ax, (backend_label, values) in zip(np.atleast_1d(axes), backend_values, strict=True):
        masked = np.ma.masked_invalid(values)
        image = ax.imshow(masked, cmap=cmap, vmin=-max_abs, vmax=max_abs, aspect="auto")
        ax.set_title(backend_label)
        ax.set_xticks(np.arange(len(col_labels)), labels=col_labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
        ax.set_xlabel("Observable")

        for row_index in range(values.shape[0]):
            for col_index in range(values.shape[1]):
                value = values[row_index, col_index]
                if np.isnan(value):
                    text = "n/a"
                    colour = "black"
                else:
                    text = f"{value:+.2f}"
                    colour = "white" if abs(value) > 0.55 * max_abs else "black"
                ax.text(
                    col_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=6.5,
                )

    np.atleast_1d(axes)[0].set_ylabel("S1 extension or policy")
    if image is None:
        raise ValueError("no backend values supplied")
    cbar = fig.colorbar(image, ax=np.atleast_1d(axes), shrink=0.82)
    cbar.set_label(
        r"$\Delta_P=\langle P\rangle_{\mathrm{feedback}}-\langle P\rangle_{\mathrm{control}}$"
    )

    png = output_dir / f"s1_feedback_multibackend_delta_heatmap_{DATE_STAMP}.png"
    pdf = output_dir / f"s1_feedback_multibackend_delta_heatmap_{DATE_STAMP}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> int:
    """Run the command-line entry point."""
    args = parse_args()
    rows, columns, matrix = build_matrix(args.data_dir, BACKEND_RUNS[0])
    outputs = write_single_backend_heatmap(rows, columns, matrix, args.figure_dir)
    backend_values = []
    for backend_run in BACKEND_RUNS:
        rows, columns, matrix = build_matrix(args.data_dir, backend_run)
        backend_values.append((backend_run["label"], matrix))
    outputs.extend(write_multibackend_heatmap(rows, columns, backend_values, args.figure_dir))
    for output in outputs:
        print(output.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
