"""Verify experiment registry completeness and internal consistency."""

from __future__ import annotations

import inspect

from scpn_quantum_control.hardware.experiments import ALL_EXPERIMENTS


def test_all_registry_keys_resolve():
    """Every key in ALL_EXPERIMENTS maps to a callable."""
    for name, func in ALL_EXPERIMENTS.items():
        assert callable(func), f"{name} maps to non-callable {func}"


def test_experiment_signatures_have_runner():
    """Every experiment function takes 'runner' as first positional arg."""
    for name, func in ALL_EXPERIMENTS.items():
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        assert params[0] == "runner", f"{name}: first param is '{params[0]}', expected 'runner'"


def test_no_duplicate_experiment_names():
    """Registry keys are unique (dict enforces this, but verify count)."""
    assert len(ALL_EXPERIMENTS) == len(set(ALL_EXPERIMENTS.keys()))
