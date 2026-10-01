# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — workspace public lazy export boundaries
"""Preserve the independently frozen workspace API through actual public imports."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast

import pytest


class Binding(TypedDict):
    """One independently observed public object and its original defining module."""

    name: str
    origin_module: str
    origin_attribute: str | None


class Package(TypedDict):
    """One frozen package's ordered public names and origin identities."""

    module: str
    bindings: list[Binding]


class Snapshot(TypedDict):
    """Independent public-surface observation supplied by the architecture owner."""

    packages: list[Package]


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = cast(
    Snapshot,
    json.loads((ROOT / "tests/data/studio_workspace/public_export_identity.json").read_text()),
)
BINDINGS = next(
    package["bindings"]
    for package in SNAPSHOT["packages"]
    if package["module"] == "scpn_quantum_control.studio_workspace"
)


@pytest.mark.parametrize("binding", BINDINGS, ids=[row["name"] for row in BINDINGS])
def test_workspace_export_retains_real_origin_and_cached_identity(binding: Binding) -> None:
    """Resolve each declared export through the real package and original owner.

    Parameters
    ----------
    binding
        Independently frozen object identity, rather than a generated lazy-table oracle.

    """
    package = importlib.import_module("scpn_quantum_control.studio_workspace")
    origin = importlib.import_module(binding["origin_module"])
    attribute = binding["origin_attribute"]
    assert attribute is not None
    expected = getattr(origin, attribute)
    actual = getattr(package, binding["name"])
    assert actual is expected
    assert getattr(package, binding["name"]) is expected
    assert vars(package)[binding["name"]] is expected


def test_workspace_discovery_and_unknown_name_refusal_preserve_public_order() -> None:
    """Keep all thirty real names discoverable without accepting an undeclared export."""
    package = importlib.import_module("scpn_quantum_control.studio_workspace")
    assert len(BINDINGS) == 30
    assert package.__all__ == [binding["name"] for binding in BINDINGS]
    assert set(package.__all__).issubset(dir(package))
    assert dir(package) == sorted(dir(package))
    with pytest.raises(AttributeError, match="has no attribute 'undeclared_workspace_export'"):
        _ = package.undeclared_workspace_export
    assert "undeclared_workspace_export" not in vars(package)


def test_cold_workspace_import_defers_all_owners_and_star_import_retains_identity() -> None:
    """Import the actual package in a fresh interpreter before exercising its complete API."""
    source = """
import importlib
import json
import sys
from pathlib import Path

package = importlib.import_module("scpn_quantum_control.studio_workspace")
owners = {"canonical", "contracts", "graph", "json_transport", "settings", "settings_portability"}
assert all("scpn_quantum_control.studio_workspace." + owner not in sys.modules for owner in owners)
snapshot = json.loads(Path(sys.argv[1]).read_text())
bindings = next(p["bindings"] for p in snapshot["packages"] if p["module"] == package.__name__)
assert len(bindings) == 30
assert package.__all__ == [binding["name"] for binding in bindings]
assert all(binding["name"] in dir(package) for binding in bindings)
scope = {}
exec("from scpn_quantum_control.studio_workspace import *", scope)
assert list(name for name in scope if name != "__builtins__") == package.__all__
for binding in bindings:
    expected = getattr(importlib.import_module(binding["origin_module"]), binding["origin_attribute"])
    assert scope[binding["name"]] is expected
    assert getattr(package, binding["name"]) is expected
print(json.dumps({"cold_deferred_owners": len(owners), "public_bindings": len(bindings)}))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            source,
            str(ROOT / "tests/data/studio_workspace/public_export_identity.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"cold_deferred_owners": 6, "public_bindings": 30}
