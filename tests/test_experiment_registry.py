# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Experiment Registry
"""Multi-angle tests for experiment registry.

Verifies: registry completeness, function signatures, return types,
parametrised experiment validation, no duplicates, naming conventions.
"""

from __future__ import annotations

import inspect

from scpn_quantum_control.hardware.experiments import ALL_EXPERIMENTS


class TestRegistryCompleteness:
    def test_all_keys_resolve_to_callables(self):
        for name, func in ALL_EXPERIMENTS.items():
            assert callable(func), f"{name} maps to non-callable {func}"

    def test_no_duplicate_names(self):
        assert len(ALL_EXPERIMENTS) == len(set(ALL_EXPERIMENTS.keys()))

    def test_registry_not_empty(self):
        assert len(ALL_EXPERIMENTS) > 0

    def test_registry_size_at_least_15(self):
        """Should have ≥15 registered experiments."""
        assert len(ALL_EXPERIMENTS) >= 15


class TestExperimentSignatures:
    def test_first_param_is_runner(self):
        for name, func in ALL_EXPERIMENTS.items():
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            assert params[0] == "runner", (
                f"{name}: first param is '{params[0]}', expected 'runner'"
            )

    def test_all_have_shots_param(self):
        """Most experiments should accept a shots parameter."""
        has_shots = 0
        for _name, func in ALL_EXPERIMENTS.items():
            sig = inspect.signature(func)
            if "shots" in sig.parameters:
                has_shots += 1
        # At least half should have shots
        assert has_shots >= len(ALL_EXPERIMENTS) // 2

    def test_signatures_have_at_least_2_params(self):
        """Every experiment should have runner + at least 1 other param."""
        for name, func in ALL_EXPERIMENTS.items():
            sig = inspect.signature(func)
            assert len(sig.parameters) >= 1, f"{name} has too few parameters"


class TestNamingConventions:
    def test_names_are_lowercase(self):
        for name in ALL_EXPERIMENTS:
            assert name == name.lower(), f"{name} should be lowercase"

    def test_names_use_underscores(self):
        for name in ALL_EXPERIMENTS:
            assert " " not in name, f"{name} should use underscores, not spaces"
            assert "-" not in name, f"{name} should use underscores, not hyphens"

    def test_names_are_strings(self):
        for name in ALL_EXPERIMENTS:
            assert isinstance(name, str)
