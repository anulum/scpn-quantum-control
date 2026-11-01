# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — full-program journeys across the admitted whole-program AD routes
"""One user program goes through every admitted route with the same identity and numbers.

The program composes an elementwise polynomial, a supported linear solve and a
scalar objective. Oracles: the analytic chain rule written out here, the Python
replay record of the same execution, and the bounded Rust effect-IR replay of
the same serialised program. A composition that fails inside library code is
refused with the stage located in the user's source; an exception the user's
own code raises propagates unchanged.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import (
    WholeProgramStageRefusal,
    whole_program_value_and_grad,
)
from scpn_quantum_control.whole_program_ad_api import (
    WholeProgramReplayRecord,
    capture_whole_program_replay,
    replay_whole_program,
)

_A = np.array([[3.0, 1.0], [1.0, 2.0]])
_C = np.array([0.5, -1.0])
_POINT = np.array([0.7, -0.4])


def _journey(p: Any) -> Any:
    """Polynomial, then a linear solve, then a scalar objective."""
    b = p * p + 2.0 * p + _C
    x = np.linalg.solve(_A, b)
    return np.sum(x * x) + 3.0 * np.sum(x)


def _mixed(p: Any) -> Any:
    """Build an array from traced scalars, which the solve cannot take."""
    b = np.array([p[0] * p[0] + p[1], p[0] * p[1] - 1.0])
    x = np.linalg.solve(_A, b)
    return x[0] * x[0] + 3.0 * x[1]


def _analytic(p: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    """Return the value and the chain-rule gradient ``(db/dp)^T A^-T (2x + 3)``."""
    b = p * p + 2.0 * p + _C
    x = np.linalg.solve(_A, b)
    outer = np.linalg.solve(_A.T, 2.0 * x + 3.0)
    return float(np.sum(x * x) + 3.0 * np.sum(x)), np.diag(2.0 * p + 2.0).T @ outer


def test_whole_program_backend_journeys_01_composed_program_matches_the_chain_rule() -> None:
    """The whole-program value and gradient equal the hand-written chain rule."""
    for point in (_POINT, np.array([-1.3, 0.25]), np.array([2.0, 1.5])):
        result = whole_program_value_and_grad(_journey, point)
        value, gradient = _analytic(point)
        assert result.value == pytest.approx(value, rel=1e-13, abs=1e-13)
        np.testing.assert_allclose(result.gradient, gradient, rtol=1e-12, atol=1e-12)
        assert result.program_ir is not None and result.method == "whole_program_ad"


def test_whole_program_backend_journeys_02_frozen_program_replays_across_admitted_routes() -> None:
    """Python replay and Rust effect-IR replay reproduce the saved hash and numbers."""
    result = whole_program_value_and_grad(_journey, _POINT)
    record = capture_whole_program_replay(result)
    assert result.program_ir is not None
    serialised = result.program_ir.serialization
    assert record.program_ir_sha256 == hashlib.sha256(serialised.encode()).hexdigest()
    # The record survives a JSON round trip unchanged, and the Python route replays it.
    restored = WholeProgramReplayRecord.from_json(record.to_json())
    assert restored == record
    report = replay_whole_program(restored, _journey)
    assert report.record.program_ir_sha256 == record.program_ir_sha256
    # The Rust route consumes the same serialised program.
    from scpn_quantum_control.program_ad_rust_bridge import (
        value_and_grad_program_ad_effect_ir_with_rust,
    )

    rust = value_and_grad_program_ad_effect_ir_with_rust(serialised, _POINT)
    assert rust.supported and rust.blocked_reasons == ()
    assert rust.supported_effect_count == rust.effect_count == len(result.program_ir.effects)
    assert rust.value == pytest.approx(record.value, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(rust.gradient, record.gradient, rtol=1e-12, atol=1e-12)
    assert (
        "rust" in result.polyglot_targets and "effect-IR replay" in result.polyglot_targets["rust"]
    )


def test_whole_program_backend_journeys_03_unsupported_mixed_composition_is_located() -> None:
    """An array built from traced scalars refuses at the solve line, naming the operation."""
    with pytest.raises(WholeProgramStageRefusal) as refused:
        whole_program_value_and_grad(_mixed, _POINT)
    stage = refused.value
    assert stage.filename == __file__
    assert stage.statement == "x = np.linalg.solve(_A, b)"
    assert stage.operation.endswith(":solve")
    assert "no value or derivative is issued" in str(stage)
    assert isinstance(stage.__cause__, TypeError)
    # The same composition is located identically on a second attempt.
    with pytest.raises(WholeProgramStageRefusal) as again:
        whole_program_value_and_grad(_mixed, _POINT)
    assert (again.value.line, again.value.operation) == (stage.line, stage.operation)
    # The admitted program still runs afterwards: nothing was left behind by the refusal.
    assert whole_program_value_and_grad(_journey, _POINT).value == pytest.approx(
        _analytic(_POINT)[0]
    )


def test_package_refusals_are_not_relocated() -> None:
    """A refusal raised inside the package keeps its own type and message."""

    def scalar_output(p: Any) -> Any:
        """Return a vector instead of a scalar."""
        return p * 2.0

    with pytest.raises(ValueError, match="must return a whole-program AD scalar") as raised:
        whole_program_value_and_grad(scalar_output, _POINT)
    assert not isinstance(raised.value, WholeProgramStageRefusal)


def test_user_raised_exceptions_propagate_unchanged() -> None:
    """An exception the user's own code raises keeps its type; only library failures are located."""

    def stop() -> None:
        """Raise as user code may do deliberately."""
        raise RuntimeError("stopped by the user", 7)

    def stop_midway(p: Any) -> Any:
        """Stop after a traced product through the user's own helper."""
        product = p[0] * p[1]
        stop()
        return product

    with pytest.raises(RuntimeError, match="stopped by the user") as raised:
        whole_program_value_and_grad(stop_midway, _POINT)
    assert type(raised.value) is RuntimeError and raised.value.args[1] == 7
    # The admitted program still runs afterwards.
    assert whole_program_value_and_grad(_journey, _POINT).value == pytest.approx(
        _analytic(_POINT)[0]
    )
    record = json.loads(
        capture_whole_program_replay(whole_program_value_and_grad(_journey, _POINT)).to_json()
    )
    assert record["record"]["program_ir_sha256"] and record["digest"]


_USER_SOURCE = """
import numpy as np


def stop():
    raise RuntimeError("stopped by the user", 7)


def objective(p):
    product = p[0] * p[1]
    stop()
    return product
"""


def _write_distribution(site: Path, top: str, source: str) -> None:
    """Write ``top`` as an importable distribution with its own metadata under ``site``."""
    package = site / top
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(source, encoding="utf-8")
    info = site / f"{top}-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {top}\nVersion: 1.0\n", encoding="utf-8"
    )
    (info / "top_level.txt").write_text(f"{top}\n", encoding="utf-8")


@pytest.fixture
def import_from(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[Path, str], Any]]:
    """Import a module from a directory and remove it from ``sys.modules`` afterwards."""
    imported: list[str] = []

    def load(directory: Path, name: str) -> Any:
        """Prepend ``directory`` to the import path and import ``name`` from it."""
        monkeypatch.syspath_prepend(str(directory))
        imported.append(name)
        return importlib.import_module(name)

    yield load
    for name in imported:
        sys.modules.pop(name, None)


def test_an_installed_user_distribution_keeps_its_own_exceptions(
    tmp_path: Path, import_from: Callable[[Path, str], Any]
) -> None:
    """User code installed as a distribution with metadata is still the user's code.

    The objective and the helper that raises live in the same top-level package,
    so the helper's RuntimeError reaches the caller unchanged although the
    package is an importable distribution. This test module's own package name,
    ``tests``, is also declared by an installed distribution; its same-package
    helpers above raise unchanged for the same reason.
    """
    _write_distribution(tmp_path / "dist", "wp_installed_user", _USER_SOURCE)
    module = import_from(tmp_path / "dist", "wp_installed_user")
    with pytest.raises(RuntimeError, match="stopped by the user") as raised:
        whole_program_value_and_grad(module.objective, _POINT)
    assert type(raised.value) is RuntimeError
    assert raised.value.args == ("stopped by the user", 7)
    assert raised.value.__cause__ is None


_CHILD_PROBE = """
import json, sys
import numpy as np
from scpn_quantum_control.differentiable import (
    WholeProgramStageRefusal, whole_program_value_and_grad,
)
import wp_site_user, wp_sibling_user, wp_mixed_user

out = {"numpy": np.__file__, "prefix": sys.prefix}
for name, objective in (
    ("site", wp_site_user.objective),
    ("sibling", wp_sibling_user.objective),
    ("mixed", wp_mixed_user.mixed),
):
    try:
        whole_program_value_and_grad(objective, np.array([0.7, -0.4]))
        out[name] = ["no error"]
    except Exception as error:
        cause = type(error.__cause__).__name__ if error.__cause__ else None
        out[name] = [type(error).__name__, list(map(str, error.args[:2])), cause]
print(json.dumps(out))
"""

_MIXED_SOURCE = """
import numpy as np

_A = np.array([[3.0, 1.0], [1.0, 2.0]])


def mixed(p):
    b = np.array([p[0] * p[0] + p[1], p[0] * p[1] - 1.0])
    x = np.linalg.solve(_A, b)
    return x[0] * x[0] + 3.0 * x[1]
"""


def test_ownership_holds_in_a_fresh_environment_with_numpy_outside_its_roots(
    tmp_path: Path,
) -> None:
    """Real installation layouts of a fresh virtual environment keep user and library apart.

    A new environment (no packages installed) imports this package and NumPy
    through ``PYTHONPATH``, so NumPy lies outside every root of that
    environment. User code sits in the environment's real site-packages and in
    a sibling directory that shares the text prefix of its standard-library
    directory. User exceptions keep their type; NumPy failing on traced values
    is still located.
    """
    import subprocess
    import sysconfig
    import venv

    environment = tmp_path / "env"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    python = environment / "bin" / "python"
    paths = json.loads(
        subprocess.run(
            [
                str(python),
                "-c",
                "import json, sysconfig; print(json.dumps(sysconfig.get_paths()))",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    site = Path(paths["purelib"])
    sibling = Path(paths["platstdlib"] + "-sibling")
    sibling.mkdir(parents=True)
    (site / "wp_site_user.py").write_text(_USER_SOURCE, encoding="utf-8")
    (sibling / "wp_sibling_user.py").write_text(_USER_SOURCE, encoding="utf-8")
    (site / "wp_mixed_user.py").write_text(_MIXED_SOURCE, encoding="utf-8")
    repository = Path(__file__).resolve().parents[1]
    import_path = [
        str(sibling),
        str(repository / "src"),
        str(repository / "oscillatools" / "src"),
        sysconfig.get_paths()["purelib"],
    ]
    result = subprocess.run(
        [str(python), "-c", _CHILD_PROBE],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": ":".join(import_path), "PATH": str(environment / "bin")},
        timeout=300,
    )
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    numpy_file = Path(observed["numpy"]).resolve()
    assert not numpy_file.is_relative_to(Path(observed["prefix"]).resolve())
    assert observed["site"] == ["RuntimeError", ["stopped by the user", "7"], None]
    assert observed["sibling"] == ["RuntimeError", ["stopped by the user", "7"], None]
    assert observed["mixed"][0] == "WholeProgramStageRefusal"
    assert observed["mixed"][2] is not None
