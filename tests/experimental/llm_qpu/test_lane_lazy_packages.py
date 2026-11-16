# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU lazy package namespaces
"""The lane's lazy packages list their exports, resolve them to the originals and refuse others."""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest

from scpn_quantum_control.experimental.llm_qpu import contracts, runtime
from scpn_quantum_control.experimental.llm_qpu.runtime import attempt_journal


def _fresh_runtime() -> ModuleType:
    """Execute the runtime package initialiser into a new, unregistered namespace.

    ``importlib.reload`` reuses the module's dictionary, so names that other tests
    already resolved would stay in it. A module built from the spec starts empty.

    Returns
    -------
    ModuleType
        Runtime package object with no export resolved yet.

    """
    spec = importlib.util.find_spec(runtime.__name__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lazy_lane_packages_list_every_export_before_it_is_resolved() -> None:
    """``dir()`` shows each declared name, including those not yet imported."""
    fresh = _fresh_runtime()
    assert not set(fresh.__all__) & set(vars(fresh))
    for package in (contracts, fresh):
        listed = dir(package)
        assert listed == sorted(listed)
        assert set(package.__all__) <= set(listed)
    assert "decode_contract" in dir(contracts)


def test_lazy_lane_packages_refuse_an_undeclared_name() -> None:
    """An undeclared name raises the error of a missing module attribute."""
    undeclared = "missing_export"
    for package in (contracts, runtime):
        with pytest.raises(
            AttributeError,
            match=f"^module '{package.__name__}' has no attribute '{undeclared}'$",
        ):
            getattr(package, undeclared)


def test_runtime_package_resolves_the_original_object_and_caches_it() -> None:
    """A resolved export is the owning module's object and is then read from the namespace."""
    fresh = _fresh_runtime()
    assert "JobPlan" not in vars(fresh)
    assert fresh.JobPlan is attempt_journal.JobPlan
    assert vars(fresh)["JobPlan"] is attempt_journal.JobPlan
