# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Lazy package runtime boundaries
"""Exercise genuine package resolution failures and import publication order."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TypedDict, cast

import pytest

pytestmark = pytest.mark.framework_imports

ROOT = Path(__file__).resolve().parents[1]


class Binding(TypedDict):
    """Original object binding used to exercise a real package export."""

    name: str
    origin_module: str
    origin_attribute: str | None


class Package(TypedDict):
    """Package name and its observed public bindings."""

    module: str
    bindings: list[Binding]


class Snapshot(TypedDict):
    """Package collection in the runtime identity observation."""

    packages: list[Package]


OBSERVATION = cast(
    Snapshot,
    json.loads(
        (ROOT / "tests/fixtures/public_export_identity_snapshot.json").read_text(encoding="utf-8")
    ),
)
# Empty namespaces and the concurrently maintained Studio facade are
# exercised by the exhaustive public compatibility suite instead.
DEFERRED_PACKAGES = [
    package
    for package in OBSERVATION["packages"]
    if package["bindings"] and package["module"] != "scpn_quantum_control.studio"
]


@pytest.mark.parametrize("wire", [b"[]", b"null", b'"string"'])
def test_lazy_contract_decoder_refuses_non_object_wire(wire: bytes) -> None:
    """Resolve the real decoder and reject non-object input before schema dispatch.

    Parameters
    ----------
    wire
        Valid JSON whose top-level value cannot represent a contract record.

    """
    from scpn_quantum_control.experimental.llm_qpu.contracts import decode_contract

    with pytest.raises(ValueError, match="contract wire must be an object"):
        decode_contract(wire)


@pytest.mark.parametrize(
    "package", DEFERRED_PACKAGES, ids=[p["module"] for p in DEFERRED_PACKAGES]
)
@pytest.mark.parametrize("failure", ["module", "attribute"])
def test_failed_resolution_is_uncached_and_a_corrected_target_can_retry(
    package: Package, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real importer against absent modules and absent attributes.

    Parameters
    ----------
    package
        Actual package and its independently observed bindings.
    failure
        Missing module or missing attribute to request through the real resolver.
    monkeypatch
        Scoped target correction and package-cache restoration.

    """
    module = importlib.import_module(package["module"])
    table = cast(dict[str, tuple[str, str | None]], module._PUBLIC_EXPORTS)
    binding = next(b for b in package["bindings"] if b["name"] in table)
    name = binding["name"]
    original = table[name]
    origin = importlib.import_module(original[0])
    expected = origin if original[1] is None else getattr(origin, original[1])
    broken = (
        ("scpn_quantum_control.absent_module_for_import_boundary_test", original[1])
        if failure == "module"
        else (original[0], "absent_attribute_for_import_boundary_test")
    )
    exception = ModuleNotFoundError if failure == "module" else AttributeError
    with monkeypatch.context() as patch:
        patch.delattr(module, name, raising=False)
        patch.setitem(table, name, broken)
        with pytest.raises(exception, match="absent_"):
            getattr(module, name)
        assert name not in vars(module)
        patch.setitem(table, name, original)
        assert getattr(module, name) is expected
        assert vars(module)[name] is expected


def test_real_child_imports_preserve_same_named_object_exports() -> None:
    """Import actual colliding leaves first, then verify public object identity."""
    source = textwrap.dedent(
        """
        import importlib
        import json
        from concurrent.futures import ThreadPoolExecutor
        from types import ModuleType

        collisions = [
            ('scpn_quantum_control', 'differentiable_api', 'scpn_quantum_control.differentiable_api'),
            ('scpn_quantum_control', 'differentiable_benchmark_report', 'scpn_quantum_control.differentiable_benchmark_report'),
            ('scpn_quantum_control.analysis', 'bkt_analysis', 'scpn_quantum_control.analysis.bkt_analysis'),
            ('scpn_quantum_control.analysis', 'dla_truncated_tn', 'scpn_quantum_control.analysis.dla_truncated_tn'),
            ('scpn_quantum_control.analysis', 'tcbo_weighted_complex', 'scpn_quantum_control.analysis.tcbo_weighted_complex'),
            ('scpn_quantum_control.applications', 'eeg_benchmark', 'scpn_quantum_control.applications.eeg_benchmark'),
            ('scpn_quantum_control.applications', 'fmo_benchmark', 'scpn_quantum_control.applications.fmo_benchmark'),
            ('scpn_quantum_control.applications', 'iter_benchmark', 'scpn_quantum_control.applications.iter_benchmark'),
            ('scpn_quantum_control.fep', 'variational_free_energy', 'scpn_quantum_control.fep.variational_free_energy'),
            ('scpn_quantum_control.gauge', 'cft_analysis', 'scpn_quantum_control.gauge.cft_analysis'),
            ('scpn_quantum_control.hardware', 'kuramoto_layout_cost', 'scpn_quantum_control.hardware.kuramoto_layout_cost'),
            ('scpn_quantum_control.identity', 'coherence_budget', 'scpn_quantum_control.identity.coherence_budget'),
            ('scpn_quantum_control.phase', 'adapt_vqe', 'scpn_quantum_control.phase.adapt_vqe'),
            ('scpn_quantum_control.phase', 'gradient_tape', 'scpn_quantum_control.phase.gradient_tape'),
            ('scpn_quantum_control.ssgf', 'quantum_outer_cycle', 'scpn_quantum_control.ssgf.quantum_outer_cycle'),
        ]
        for package_name, name, child_name in reversed(collisions):
            child = importlib.import_module(child_name)
            package = importlib.import_module(package_name)
            origin_name, attribute = package._PUBLIC_EXPORTS[name]
            expected = getattr(importlib.import_module(origin_name), attribute)
            assert isinstance(child, ModuleType)
            assert getattr(package, name) is expected, (package_name, name)
            assert vars(package)[name] is expected
            assert importlib.import_module(child_name) is child
            assert package.__spec__.name == package_name
            sentinel = object()
            setattr(package, name, sentinel)
            assert getattr(package, name) is sentinel
            setattr(package, name, expected)
            setattr(package, name, importlib.import_module('json'))
            assert getattr(package, name) is importlib.import_module('json')
            setattr(package, name, child)
            assert getattr(package, name) is expected

        package = importlib.import_module('scpn_quantum_control.compiler')
        with ThreadPoolExecutor(max_workers=4) as pool:
            exports = list(pool.map(lambda _: package.MLIRCompileConfig, range(12)))
        assert all(value is exports[0] for value in exports)
        assert exports[0] is importlib.import_module('scpn_quantum_control.compiler.mlir').MLIRCompileConfig
        print(json.dumps({'collisions': len(collisions), 'concurrent_accesses': len(exports)}))
        """
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "oscillatools/src"), environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"collisions": 15, "concurrent_accesses": 12}
