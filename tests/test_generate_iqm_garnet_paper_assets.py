# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM Garnet manuscript figure tests
"""Tests for the figures of the IQM Garnet manuscript."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module() -> ModuleType:
    """Import the generator script as a module."""
    script = REPO_ROOT / "scripts" / "generate_iqm_garnet_paper_assets.py"
    spec = importlib.util.spec_from_file_location("generate_iqm_garnet_paper_assets", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the IQM Garnet figure generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_source_artefact_is_committed_and_parsed() -> None:
    """Each figure source exists in the repository and holds the plotted fields."""
    module = _load_module()
    data = module.load_sources(REPO_ROOT)
    assert set(data) == set(module.SOURCES)
    assert len(data["epochs"]["per_epoch"]) == 12
    assert len(data["windows"]["per_window"]) == 10
    assert set(data["layout_first"]["point_errors"]) == {"8", "12", "16"}
    assert data["heron"]["backend"] == "ibm_kingston"


def test_binomial_standard_error_matches_the_closed_form() -> None:
    """The single-window error bar is the binomial standard error."""
    module = _load_module()
    assert module.binomial_standard_error(0.5, 4096) == pytest.approx(0.5 / 64)
    assert module.binomial_standard_error(0.25, 100) == pytest.approx(math.sqrt(0.001875))
    assert module.binomial_standard_error(0.0, 10) == 0.0


def test_depth_contrast_figure_plots_the_artefact_values() -> None:
    """The contrast panel carries the twelve committed contrasts and their mean."""
    module = _load_module()
    module.configure(latex_text=False)
    epochs = module.load_sources(REPO_ROOT)["epochs"]
    figure = module.depth_contrast_figure(epochs)
    right = figure.axes[1]
    plotted = [float(value) for value in right.lines[1].get_ydata()]
    assert plotted == [row["contrast_delta8_minus_delta12"] for row in epochs["per_epoch"]]
    assert float(right.lines[0].get_ydata()[0]) == epochs["primary_depth_ordering"]["mean"]


def test_device_comparison_keeps_the_reference_within_twenty_steps() -> None:
    """The IBM reference is drawn up to depth 20 and Garnet at its five depths."""
    module = _load_module()
    module.configure(latex_text=False)
    data = module.load_sources(REPO_ROOT)
    figure = module.device_comparison_figure(data["heron"], data["garnet_joint"], data["epochs"])
    left = figure.axes[0]
    depths = [[float(value) for value in line.get_xdata()] for line in left.lines]
    assert [2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0] in depths
    assert [4.0, 6.0, 8.0, 10.0, 12.0] in depths


def test_write_figures_creates_both_formats_without_tool_metadata(tmp_path: Path) -> None:
    """All four figures are written as PNG and PDF and name no producing tool."""
    module = _load_module()
    written = module.write_figures(REPO_ROOT, tmp_path / "figures")
    assert [path.name for path in written] == [
        f"{name}.{suffix}" for name in module.FIGURE_NAMES for suffix in ("png", "pdf")
    ] + [module.SOURCE_RECORD]
    for path in written[:-1]:
        payload = path.read_bytes()
        assert len(payload) > 1000
        assert b"atplotlib" not in payload


def test_source_record_binds_each_figure_input_to_its_committed_bytes(tmp_path: Path) -> None:
    """The written record names every source artefact with its actual digest."""
    module = _load_module()
    record_path = module.write_figures(REPO_ROOT, tmp_path)[-1]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert set(record) == set(module.SOURCES)
    for key, relative in module.SOURCES.items():
        digest = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        assert record[key] == {"path": relative, "sha256": digest}


def test_configure_selects_latex_rendering_on_request() -> None:
    """The LaTeX switch sets the text renderer and its preamble."""
    module = _load_module()
    module.configure(latex_text=True)
    assert module.plt.rcParams["text.usetex"] is True
    assert "fontenc" in module.plt.rcParams["text.latex.preamble"]
    module.configure(latex_text=False)
    assert module.plt.rcParams["text.usetex"] is False


def test_command_line_writes_to_the_requested_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command line entry point writes the files and prints their paths."""
    module = _load_module()
    assert module.main(["--output-dir", str(tmp_path)]) == 0
    printed = capsys.readouterr().out.split()
    assert len(printed) == 9
    assert all(Path(line).is_file() for line in printed)


def test_script_entry_point_exits_with_the_command_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running the file as a script writes the figures and exits with status zero."""
    script = REPO_ROOT / "scripts" / "generate_iqm_garnet_paper_assets.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(script), run_name="__main__")
    assert raised.value.code == 0
    assert (tmp_path / "figure_sources.json").is_file()
