# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Public Api
"""Verify public API exports are stable and importable."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_top_level_version() -> None:
    """Package exposes __version__ matching pyproject.toml."""
    import importlib.metadata

    import scpn_quantum_control

    assert hasattr(scpn_quantum_control, "__version__")
    try:
        expected = importlib.metadata.version("scpn-quantum-control")
        assert scpn_quantum_control.__version__ == expected
    except importlib.metadata.PackageNotFoundError:
        # Fallback for development environments where package is not installed
        assert isinstance(scpn_quantum_control.__version__, str)
        assert len(scpn_quantum_control.__version__.split(".")) >= 3


def test_top_level_version_is_not_hardcoded_carrier() -> None:
    """Runtime version is resolved from package metadata, not duplicated by hand."""
    init_source = (
        Path(__file__).resolve().parents[1] / "src" / "scpn_quantum_control" / "__init__.py"
    ).read_text(encoding="utf-8")

    assert '__version__ = "0.9.' not in init_source
    assert 'version("scpn-quantum-control")' in init_source


def _check_exports(submod: str) -> int:
    mod = __import__(f"scpn_quantum_control.{submod}", fromlist=["__all__"])
    for name in mod.__all__:
        obj = getattr(mod, name)
        assert callable(obj) or isinstance(
            obj, (type, np.ndarray, dict, frozenset, list, tuple, str, int, float)
        ), f"{submod}.{name} has unexpected type {type(obj)}"
    return len(mod.__all__)


def test_bridge_exports() -> None:
    """bridge.__all__ exports are importable and typed."""
    assert _check_exports("bridge") > 0


def test_phase_exports() -> None:
    """phase.__all__ exports are importable and typed."""
    assert _check_exports("phase") > 0


def test_control_exports() -> None:
    """control.__all__ exports are importable and typed."""
    assert _check_exports("control") > 0


def test_qsnn_exports() -> None:
    """qsnn.__all__ exports are importable and typed."""
    assert _check_exports("qsnn") > 0


def test_mitigation_exports() -> None:
    """mitigation.__all__ exports are importable and typed."""
    assert _check_exports("mitigation") > 0


def test_hardware_exports() -> None:
    """hardware.__all__ exports are importable and typed."""
    assert _check_exports("hardware") > 0


def test_benchmark_exports() -> None:
    """benchmarks.__all__ exports are importable and typed."""
    assert _check_exports("benchmarks") > 0


def test_no_private_in_all() -> None:
    """No __all__ contains underscore-prefixed names."""
    for submod_name in [
        "bridge",
        "phase",
        "control",
        "qsnn",
        "mitigation",
        "hardware",
        "qec",
        "benchmarks",
    ]:
        mod = __import__(f"scpn_quantum_control.{submod_name}", fromlist=["__all__"])
        if hasattr(mod, "__all__"):
            private = [n for n in mod.__all__ if n.startswith("_")]
            assert not private, f"{submod_name}.__all__ has private names: {private}"


# ---------------------------------------------------------------------------
# Top-level __all__ completeness
# ---------------------------------------------------------------------------


def test_top_level_all_nonempty() -> None:
    """Keep the package's declared public export list populated."""
    import scpn_quantum_control

    assert len(scpn_quantum_control.__all__) > 50


def test_top_level_all_no_duplicates() -> None:
    """Keep each public export declared exactly once."""
    import scpn_quantum_control

    names = scpn_quantum_control.__all__
    assert len(names) == len(set(names))


def test_qec_exports() -> None:
    """qec.__all__ exports are importable and typed."""
    assert _check_exports("qec") > 0


def test_all_submodules_importable() -> None:
    """Every known subpackage imports without error."""
    for submod in [
        "bridge",
        "phase",
        "control",
        "qsnn",
        "mitigation",
        "hardware",
        "qec",
        "analysis",
        "benchmarks",
        "identity",
        "crypto",
    ]:
        mod = __import__(f"scpn_quantum_control.{submod}")
        assert mod is not None


@pytest.mark.parametrize(
    "first_module",
    [
        "differentiable_dashboard",
        "differentiable_api",
        "differentiable_benchmark_report",
        "phase",
        "program_ad_adjoint",
    ],
)
def test_cold_differentiable_import_order_preserves_public_execution(
    tmp_path: Path, first_module: str
) -> None:
    """Cold imports preserve public aliases and execute a source-visible objective.

    Parameters
    ----------
    tmp_path
        Owned directory containing the actual imported numerical objective.
    first_module
        Differentiable owner or adjacent package loaded before public API access.

    """
    (tmp_path / "cold_objective.py").write_text(
        "from scpn_quantum_control import TraceADArray\n"
        "def objective(values: TraceADArray) -> object:\n"
        "    return values[0] ** 2\n",
        encoding="utf-8",
    )
    script = """
import importlib
import sys
importlib.import_module('scpn_quantum_control.' + sys.argv[1])
import scpn_quantum_control as control
api = importlib.import_module('scpn_quantum_control.differentiable_api')
assert control.differentiable_dashboard_status is api.differentiable_dashboard_status
assert control.differentiable_benchmark_report is api.differentiable_benchmark_report
status = control.differentiable_dashboard_status()
assert any(row.surface == 'program_ad_ir' for row in status.rows)
sys.path.insert(0, sys.argv[2])
from cold_objective import objective
result = control.whole_program_value_and_grad(objective, [2.0], trace=False)
assert result.value == 4.0
assert result.gradient.tolist() == [4.0]
from scpn_quantum_control.differentiable import program_adjoint_replay_gradient
assert program_adjoint_replay_gradient(result).tolist() == [4.0]
"""
    result = subprocess.run(
        [sys.executable, "-c", script, first_module, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
