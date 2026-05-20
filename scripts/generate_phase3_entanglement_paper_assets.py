#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Phase 3 entanglement/tomography paper assets
"""Generate paper tables and figures for Phase 3 entanglement/tomography."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = (
    REPO_ROOT
    / "data"
    / "phase3_entanglement_tomography"
    / "entanglement_tomography_rows_2026-05-20.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "phase3_entanglement_tomography"
DEFAULT_FIGURE_DIR = REPO_ROOT / "figures" / "phase3"
TODAY = "2026-05-20"


def _float(value: str) -> float:
    return float(value)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _group(
    rows: Iterable[Mapping[str, str]], keys: Sequence[str]
) -> dict[tuple[str, ...], list[Mapping[str, str]]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    return grouped


def build_label_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Aggregate deviation metrics by source circuit label."""

    output: list[dict[str, Any]] = []
    for key, group_rows in sorted(_group(rows, ["family", "label"]).items()):
        signed = [_float(row["deviation_from_exact"]) for row in group_rows]
        absdev = [_float(row["absolute_deviation"]) for row in group_rows]
        stderr = [_float(row["stderr_expectation"]) for row in group_rows]
        output.append(
            {
                "family": key[0],
                "label": key[1],
                "n_observables": len(group_rows),
                "mean_signed_deviation": mean(signed),
                "mean_absolute_deviation": mean(absdev),
                "max_absolute_deviation": max(absdev),
                "mean_stderr": mean(stderr),
            }
        )
    return output


def build_basis_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Aggregate deviation metrics by reduced-Pauli basis setting."""

    output: list[dict[str, Any]] = []
    for key, group_rows in sorted(_group(rows, ["basis_setting"]).items()):
        signed = [_float(row["deviation_from_exact"]) for row in group_rows]
        absdev = [_float(row["absolute_deviation"]) for row in group_rows]
        output.append(
            {
                "basis_setting": key[0],
                "n_observables": len(group_rows),
                "mean_signed_deviation": mean(signed),
                "mean_absolute_deviation": mean(absdev),
                "max_absolute_deviation": max(absdev),
            }
        )
    return output


def build_top_deviations(
    rows: Sequence[Mapping[str, str]], *, limit: int = 12
) -> list[dict[str, Any]]:
    """Return the largest absolute deviations from exact references."""

    selected = sorted(rows, key=lambda row: _float(row["absolute_deviation"]), reverse=True)[
        :limit
    ]
    return [
        {
            "family": row["family"],
            "label": row["label"],
            "initial": row["initial"],
            "depth": row["depth"],
            "lambda_fim": row["lambda_fim"],
            "basis_setting": row["basis_setting"],
            "mean_expectation": _float(row["mean_expectation"]),
            "stderr_expectation": _float(row["stderr_expectation"]),
            "exact_expectation": _float(row["exact_expectation"]),
            "deviation_from_exact": _float(row["deviation_from_exact"]),
            "absolute_deviation": _float(row["absolute_deviation"]),
        }
        for row in selected
    ]


def write_markdown_table(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a compact Markdown table for direct manuscript review."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_heatmap(rows: Sequence[Mapping[str, str]], figure_dir: Path) -> tuple[Path, Path]:
    """Plot signed measured-minus-exact deviations by label and basis setting."""

    labels = [
        "dla_even_shallow",
        "dla_odd_shallow",
        "dla_even_signal",
        "dla_odd_signal",
        "fim_lambda0_reference",
        "fim_lambda4_feedback",
    ]
    bases = ["IIXX", "IIYY", "IIZZ", "IXXI", "IYYI", "IZZI", "XXII", "YYII", "ZZII"]
    lookup = {
        (row["label"], row["basis_setting"]): _float(row["deviation_from_exact"]) for row in rows
    }
    matrix = [[lookup[(label, basis)] for basis in bases] for label in labels]

    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "phase3_entanglement_deviation_heatmap_2026-05-20.png"
    pdf_path = figure_dir / "phase3_entanglement_deviation_heatmap_2026-05-20.pdf"

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
    vmax = max(abs(value) for line in matrix for value in line)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(image, ax=ax, shrink=0.84)
    cbar.set_label("Measured - exact Pauli expectation")
    ax.set_xticks(range(len(bases)))
    ax.set_xticklabels(bases, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Phase 3 reduced-Pauli deviations on ibm_marrakesh")
    for y, line in enumerate(matrix):
        for x, value in enumerate(line):
            color = "white" if abs(value) > 0.35 * vmax else "black"
            ax.text(x, y, f"{value:+.2f}", ha="center", va="center", fontsize=7, color=color)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def write_manifest(
    path: Path,
    *,
    source_rows: Path,
    outputs: Sequence[Path],
) -> None:
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- scpn-quantum-control -- Phase 3 entanglement/tomography paper assets -->",
        "",
        "# Phase 3 Entanglement/Tomography Paper Assets",
        "",
        f"Date: {TODAY}",
        "",
        f"- Source rows: `{source_rows.relative_to(REPO_ROOT)}`",
        "",
        "## Outputs",
        "",
    ]
    for output in outputs:
        lines.append(f"- `{output.relative_to(REPO_ROOT)}` SHA256 `{_sha256(output)}`")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "These assets summarise measured reduced-Pauli deviations for the fixed",
            "`ibm_marrakesh` run only. They do not support scalable tomography,",
            "quantum advantage, or backend-general entanglement-dynamics claims.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _read_rows(args.rows)
    label_summary = build_label_summary(rows)
    basis_summary = build_basis_summary(rows)
    top_deviations = build_top_deviations(rows)

    output_dir = args.output_dir
    label_csv = output_dir / f"entanglement_tomography_label_summary_{TODAY}.csv"
    basis_csv = output_dir / f"entanglement_tomography_basis_summary_{TODAY}.csv"
    top_csv = output_dir / f"entanglement_tomography_top_deviations_{TODAY}.csv"
    label_md = output_dir / f"entanglement_tomography_label_summary_{TODAY}.md"
    top_md = output_dir / f"entanglement_tomography_top_deviations_{TODAY}.md"

    _write_csv(label_csv, label_summary)
    _write_csv(basis_csv, basis_summary)
    _write_csv(top_csv, top_deviations)
    write_markdown_table(label_md, "Phase 3 Entanglement Label Summary", label_summary)
    write_markdown_table(top_md, "Phase 3 Entanglement Largest Deviations", top_deviations)
    heatmap_png, heatmap_pdf = plot_heatmap(rows, args.figure_dir)

    manifest = output_dir / f"entanglement_tomography_paper_assets_{TODAY}.md"
    outputs = [label_csv, basis_csv, top_csv, label_md, top_md, heatmap_png, heatmap_pdf]
    write_manifest(manifest, source_rows=args.rows, outputs=outputs)
    for output in [*outputs, manifest]:
        print(output.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
