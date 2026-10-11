#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM Garnet manuscript figures
"""Draw the figures of the IQM Garnet manuscript from committed analysis artefacts.

Every plotted value is read from an analysis artefact under ``data/``; nothing
is recomputed from raw counts. The only derived quantities are binomial
standard errors of the single-window Garnet points and the quadrature sum of
the two sector standard errors of the IBM reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = (
    REPO_ROOT / "paper/submissions/submission_007_iqm_garnet_preregistered_campaigns/figures"
)
GARNET = "data/iqm_paper_replication"
SOURCES = {
    "layout_first": "data/iqm_layout_transfer/iqm_layout_transfer_primary_analysis_2026-07-21.json",
    "layout_per_size": (
        "data/iqm_layout_transfer_per_size/live_2026-07-26/live_analysis_2026-07-26.json"
    ),
    "heron": "data/phase2_dla_parity/phase2_reduced_ag_summary_2026-05-05.json",
    "garnet_joint": f"{GARNET}/iqm_dla_depth_profile_primary_analysis_2026-07-22.json",
    "windows": (
        f"{GARNET}/iqm_dla_window_variability_analysis_through_observation_window_10_2026-09-04.json"
    ),
    "window_epochs": (
        f"{GARNET}/iqm_dla_window_variability_calibration_epoch_sensitivity"
        "_through_observation_window_10_2026-09-04.json"
    ),
    "epochs": f"{GARNET}/iqm_dla_depth_profile_powered_analysis_2026-10-11.json",
}
FIGURE_NAMES = (
    "garnet_layout_transfer_by_chain_length",
    "heron_garnet_parity_leakage_by_depth",
    "garnet_parity_asymmetry_ten_windows",
    "garnet_depth_contrast_twelve_epochs",
)
SOURCE_RECORD = "figure_sources.json"
SINGLE_WINDOW_SHOTS = 4096
HERON_MAX_DEPTH = 20
BLUE, RED, GREY, INK = "#1f4e79", "#b5483f", "#8a8a8a", "#222222"
BARS: dict[str, Any] = {"ms": 3.2, "lw": 0.9, "capsize": 1.8, "capthick": 0.8}
DELTA = r"$\Delta=\mathrm{leak}_{\mathrm{even}}-\mathrm{leak}_{\mathrm{odd}}$"


def configure(*, latex_text: bool) -> None:
    """Set the plotting style shared by all figures.

    Parameters
    ----------
    latex_text : bool
        Render all text with LaTeX so that it matches the manuscript typeface.
        Requires a LaTeX installation; the default mathtext rendering does not.

    """
    plt.rcParams.update(
        {
            "text.usetex": latex_text,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )
    if latex_text:
        plt.rcParams["text.latex.preamble"] = r"\usepackage[T1]{fontenc}\usepackage{amsmath}"


def load_sources(repo_root: Path) -> dict[str, Any]:
    """Read every analysis artefact the figures are drawn from.

    Parameters
    ----------
    repo_root : Path
        Root of the repository that holds the committed artefacts.

    Returns
    -------
    dict[str, Any]
        Parsed artefacts keyed as in ``SOURCES``.

    """
    return {
        key: json.loads((repo_root / relative).read_text(encoding="utf-8"))
        for key, relative in SOURCES.items()
    }


def source_record(repo_root: Path) -> dict[str, dict[str, str]]:
    """Return path and SHA-256 of every artefact the figures are drawn from.

    Parameters
    ----------
    repo_root : Path
        Root of the repository that holds the committed artefacts.

    Returns
    -------
    dict[str, dict[str, str]]
        For each key of ``SOURCES`` its repository path and content digest.

    """
    return {
        key: {
            "path": relative,
            "sha256": hashlib.sha256((repo_root / relative).read_bytes()).hexdigest(),
        }
        for key, relative in SOURCES.items()
    }


def binomial_standard_error(proportion: float, shots: int) -> float:
    """Return the binomial standard error of an observed proportion.

    Parameters
    ----------
    proportion : float
        Observed fraction of leaked shots.
    shots : int
        Number of shots the fraction was measured on.

    Returns
    -------
    float
        Square root of ``p (1 - p) / shots``.

    """
    return math.sqrt(proportion * (1.0 - proportion) / shots)


def _panel_letter(axis: Axes, letter: str) -> None:
    """Label one panel in its upper left corner."""
    axis.text(0.0, 1.03, f"({letter})", transform=axis.transAxes, va="bottom", ha="left")


def layout_transfer_figure(first: dict[str, Any], per_size: dict[str, Any]) -> Figure:
    """Draw the three placements and the powered per-size difference.

    Parameters
    ----------
    first : dict[str, Any]
        Primary analysis of the first layout-transfer experiment.
    per_size : dict[str, Any]
        Analysis of the powered per-size follow-up.

    Returns
    -------
    Figure
        Two panels: error per placement, and automatic minus optimised error.

    """
    sizes = ("8", "12", "16")
    labels = [f"$n={size}$" for size in sizes]
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    arms = (
        ("optimised", "optimised", BLUE, "o"),
        ("default", "automatic", INK, "s"),
        ("naive", "topological baseline", GREY, "^"),
    )
    for index, (arm, label, colour, marker) in enumerate(arms):
        left.plot(
            [position + (index - 1) * 0.18 for position in range(3)],
            [first["point_errors"][size][arm] for size in sizes],
            marker,
            color=colour,
            ms=4.2,
            label=label,
        )
    left.set_xticks(range(3))
    left.set_xticklabels(labels)
    left.set_ylim(0, 0.46)
    left.set_ylabel(r"error $|R_{\mathrm{hw}}-0.5|$")
    left.legend(frameon=False, loc="upper left", handletextpad=0.2)
    _panel_letter(left, "a")
    points = [per_size["per_size"][size]["primary_default_minus_optimised"] for size in sizes]
    right.errorbar(
        range(3),
        [point["point"] for point in points],
        yerr=[
            [point["point"] - point["bootstrap_ci95"][0] for point in points],
            [point["bootstrap_ci95"][1] - point["point"] for point in points],
        ],
        fmt="o",
        color=INK,
        **BARS,
    )
    right.axhline(0, color=GREY, lw=0.6)
    right.set_xticks(range(3))
    right.set_xticklabels(labels)
    right.set_xlim(-0.5, 2.5)
    right.set_ylabel("automatic $-$ optimised error")
    right.annotate(
        "optimiser better", (2.45, 0.004), ha="right", va="bottom", fontsize=7, color=GREY
    )
    right.annotate("optimiser worse", (2.45, -0.004), ha="right", va="top", fontsize=7, color=GREY)
    _panel_letter(right, "b")
    figure.tight_layout(w_pad=2.0)
    return figure


def device_comparison_figure(
    heron: dict[str, Any], garnet_joint: dict[str, Any], epochs: dict[str, Any]
) -> Figure:
    """Draw parity leakage on IBM Heron r2 and on IQM Garnet by depth.

    Parameters
    ----------
    heron : dict[str, Any]
        Per-depth summary of the ``ibm_kingston`` Phase 2 reference.
    garnet_joint : dict[str, Any]
        Depth-profile analysis holding the joint July profile of Garnet.
    epochs : dict[str, Any]
        Twelve-epoch analysis with the per-depth random-effects means.

    Returns
    -------
    Figure
        Two panels: sector leakage, and the even-minus-odd difference.

    """
    reference = [row for row in heron["depth_summaries"] if row["depth"] <= HERON_MAX_DEPTH]
    profile = garnet_joint["joint_profile_with_cross_window_caveat"]["profile"]
    heron_depths = [row["depth"] for row in reference]
    garnet_depths = [row["depth"] for row in profile]
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for sector, style, face in (("even", "-", None), ("odd", "--", "white")):
        left.errorbar(
            heron_depths,
            [row[f"leakage_{sector}"] for row in reference],
            yerr=[row[f"sem_{sector}"] for row in reference],
            fmt=f"o{style}",
            color=BLUE,
            mfc=face or BLUE,
            label=f"Heron r2, {sector}",
            **BARS,
        )
    for sector, style, face in (("even", "-", None), ("odd", "--", "white")):
        values = [row[f"leak_{sector}"] for row in profile]
        left.errorbar(
            garnet_depths,
            values,
            yerr=[binomial_standard_error(value, SINGLE_WINDOW_SHOTS) for value in values],
            fmt=f"s{style}",
            color=RED,
            mfc=face or RED,
            label=f"Garnet, {sector}",
            **BARS,
        )
    left.axhline(0.5, color=GREY, lw=0.6, ls=":")
    left.set_ylim(0, 0.54)
    left.set_xticks(range(2, 21, 2))
    left.set_xlabel("Trotter depth")
    left.set_ylabel("parity leakage")
    left.legend(frameon=False, ncol=2, loc="lower right", columnspacing=1.0, handletextpad=0.3)
    _panel_letter(left, "a")
    right.errorbar(
        heron_depths,
        [row["leakage_even"] - row["leakage_odd"] for row in reference],
        yerr=[math.hypot(row["sem_even"], row["sem_odd"]) for row in reference],
        fmt="o-",
        color=BLUE,
        label="Heron r2",
        **BARS,
    )
    right.errorbar(
        garnet_depths,
        [row["leak_even"] - row["leak_odd"] for row in profile],
        yerr=[
            math.hypot(
                binomial_standard_error(row["leak_even"], SINGLE_WINDOW_SHOTS),
                binomial_standard_error(row["leak_odd"], SINGLE_WINDOW_SHOTS),
            )
            for row in profile
        ],
        fmt="s",
        color=RED,
        mfc="white",
        label="Garnet, single windows (July)",
        **BARS,
    )
    means = epochs["secondary_per_depth_random_effects"]
    right.errorbar(
        [8.35, 12.35],
        [means["8"]["mean"], means["12"]["mean"]],
        yerr=[means["8"]["standard_error"], means["12"]["standard_error"]],
        fmt="D",
        color=RED,
        label="Garnet, mean of twelve epochs",
        **BARS,
    )
    right.axhline(0, color=GREY, lw=0.6)
    right.set_xticks(range(2, 21, 2))
    right.set_xlabel("Trotter depth")
    right.set_ylabel(DELTA)
    right.legend(frameon=False, loc="upper right", handletextpad=0.3)
    _panel_letter(right, "b")
    figure.tight_layout(w_pad=2.0)
    return figure


def window_variability_figure(windows: dict[str, Any], window_epochs: dict[str, Any]) -> Figure:
    """Draw the even-minus-odd difference in the ten execution windows.

    Parameters
    ----------
    windows : dict[str, Any]
        Window-level analysis of the variability campaign.
    window_epochs : dict[str, Any]
        Epoch-sensitivity analysis mapping windows to calibration epochs.

    Returns
    -------
    Figure
        One panel per depth; shading alternates with the calibration epoch.

    """
    mapping = window_epochs["epoch_level_sensitivity"]["window_to_epoch"]
    epoch_of = {row["window"]: row["epoch"] for row in mapping}
    figure, axes = plt.subplots(1, 4, figsize=(7.0, 2.05), sharey=True)
    for axis, depth in zip(axes, ("4", "8", "10", "12"), strict=True):
        for epoch in sorted(set(epoch_of.values())):
            members = [window for window, value in epoch_of.items() if value == epoch]
            if epoch % 2 == 0:
                axis.axvspan(min(members) - 0.5, max(members) + 0.5, color="0.93", lw=0)
        for row in windows["per_window"]:
            cell = row["depths"][depth]
            axis.errorbar(
                row["window"],
                cell["difference_even_minus_odd"],
                yerr=cell["shot_noise_se"],
                fmt="o",
                color=INK,
                **BARS,
            )
        axis.axhline(0, color=GREY, lw=0.6)
        axis.set_title(f"depth {depth}", fontsize=8.5, pad=3)
        axis.set_xticks([1, 4, 7, 10])
        axis.set_xlim(0.4, 10.6)
        axis.set_xlabel("window")
    axes[0].set_ylabel(DELTA)
    figure.tight_layout(w_pad=0.6)
    return figure


def depth_contrast_figure(epochs: dict[str, Any]) -> Figure:
    """Draw the twelve-epoch asymmetries and their preregistered contrast.

    Parameters
    ----------
    epochs : dict[str, Any]
        Frozen primary analysis of the twelve-epoch campaign.

    Returns
    -------
    Figure
        Two panels: per-depth difference, and contrast with its pooled mean.

    """
    rows = epochs["per_epoch"]
    primary = epochs["primary_depth_ordering"]
    index = [row["epoch"] for row in rows]
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for depth, colour, face, offset, marker in (
        ("8", BLUE, BLUE, -0.13, "o"),
        ("12", RED, "white", 0.13, "s"),
    ):
        left.errorbar(
            [epoch + offset for epoch in index],
            [row["depths"][depth]["difference_even_minus_odd"] for row in rows],
            yerr=[row["depths"][depth]["shot_noise_se"] for row in rows],
            fmt=marker,
            color=colour,
            mfc=face,
            label=f"depth {depth}",
            **BARS,
        )
    left.axhline(0, color=GREY, lw=0.6)
    left.set_xticks(index)
    left.set_xlabel("calibration epoch")
    left.set_ylabel(DELTA)
    left.set_ylim(-0.036, 0.068)
    left.legend(frameon=False, loc="upper left", ncol=2, handletextpad=0.2, columnspacing=1.0)
    _panel_letter(left, "a")
    mean, error = primary["mean"], primary["standard_error"]
    right.axhspan(mean - error, mean + error, color=BLUE, alpha=0.14, lw=0)
    right.axhline(mean, color=BLUE, lw=1.0)
    right.errorbar(
        index,
        [row["contrast_delta8_minus_delta12"] for row in rows],
        yerr=[row["contrast_shot_noise_se"] for row in rows],
        fmt="o",
        color=INK,
        **BARS,
    )
    right.axhline(0, color=GREY, lw=0.6)
    right.set_xticks(index)
    right.set_xlabel("calibration epoch")
    right.set_ylabel(r"contrast $\Delta_{8}-\Delta_{12}$")
    _panel_letter(right, "b")
    figure.tight_layout(w_pad=2.0)
    return figure


def write_figures(repo_root: Path, output_dir: Path, *, latex_text: bool = False) -> list[Path]:
    """Draw all figures as PNG and PDF without tool metadata and record their sources.

    Parameters
    ----------
    repo_root : Path
        Root of the repository that holds the committed artefacts.
    output_dir : Path
        Directory that receives the figure files; created when missing.
    latex_text : bool
        Render text with LaTeX instead of mathtext.

    Returns
    -------
    list[Path]
        Written files: PNG then PDF for each figure in manuscript order, then
        the record of source artefacts and their digests.

    """
    configure(latex_text=latex_text)
    data = load_sources(repo_root)
    figures = (
        layout_transfer_figure(data["layout_first"], data["layout_per_size"]),
        device_comparison_figure(data["heron"], data["garnet_joint"], data["epochs"]),
        window_variability_figure(data["windows"], data["window_epochs"]),
        depth_contrast_figure(data["epochs"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, figure in zip(FIGURE_NAMES, figures, strict=True):
        png, pdf = output_dir / f"{name}.png", output_dir / f"{name}.pdf"
        figure.savefig(
            png, bbox_inches="tight", pad_inches=0.02, dpi=300, metadata={"Software": None}
        )
        figure.savefig(
            pdf,
            bbox_inches="tight",
            pad_inches=0.02,
            metadata={"Creator": None, "Producer": None, "CreationDate": None},
        )
        plt.close(figure)
        written.extend((png, pdf))
    record = output_dir / SOURCE_RECORD
    record.write_text(json.dumps(source_record(repo_root), indent=2) + "\n", encoding="utf-8")
    written.append(record)
    return written


def main(argv: list[str] | None = None) -> int:
    """Write the manuscript figures from the command line.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; ``None`` reads the process arguments.

    Returns
    -------
    int
        Zero after all figures are written.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--latex-text", action="store_true")
    args = parser.parse_args(argv)
    for path in write_figures(args.repo_root, args.output_dir, latex_text=args.latex_text):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
