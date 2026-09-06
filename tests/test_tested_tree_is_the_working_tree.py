# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tested-tree provenance gate
"""Prove the suite is exercising the working tree and not an installed copy.

A non-editable install places a full copy of a package under ``site-packages``.
Because ``pythonpath = ["."]`` puts the repository root on the path rather than
``src``, that copy wins every import and the suite silently validates whatever
the tree looked like on the day someone last ran ``pip install .``. Nothing
fails, coverage stays high, and every green result describes a snapshot instead
of the code under edit.

Continuous integration installs both packages with ``pip install --no-deps -e``,
so the divergence is local-only and invisible from a workflow log. That asymmetry
is the point: this gate makes a stale local environment fail loudly and name its
own remedy, instead of quietly granting a passing suite to code nobody ran.

Both distributions in this repository are covered. A gate that watched only the
package whose staleness happened to be noticed would leave the sibling free to
drift, and a stale sibling produces the same false green through a re-export.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# package import name -> directory holding it, relative to the repository root
SOURCE_PACKAGES: dict[str, str] = {
    "scpn_quantum_control": "src",
    "oscillatools": "oscillatools/src",
}

REMEDY = (
    "python -m pip install --no-deps -e . && python -m pip install --no-deps -e oscillatools/"
    "  (matches the CI install steps)"
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _expected_location(package: str) -> Path:
    return _repository_root() / SOURCE_PACKAGES[package] / package


@pytest.mark.parametrize("package", sorted(SOURCE_PACKAGES))
def test_package_resolves_to_the_source_tree(package: str) -> None:
    """Each package must be imported from the tree under version control."""
    module = importlib.import_module(package)
    assert module.__file__ is not None
    resolved = Path(module.__file__).resolve().parent
    expected = _expected_location(package)
    assert resolved == expected, (
        f"the suite is importing {package} from {resolved} instead of {expected}; "
        f"the environment holds a stale copy of the package. Remedy: {REMEDY}"
    )


@pytest.mark.parametrize("package", sorted(SOURCE_PACKAGES))
def test_no_installed_copy_shadows_the_source_tree(package: str) -> None:
    """No second copy of a package may sit on the import path."""
    module = importlib.import_module(package)
    expected = _expected_location(package)
    duplicates = [
        Path(entry).resolve() for entry in module.__path__ if Path(entry).resolve() != expected
    ]
    assert not duplicates, (
        f"additional locations for {package} are on the import path: {duplicates}; "
        f"an installed copy can shadow edits to the source tree. Remedy: {REMEDY}"
    )
