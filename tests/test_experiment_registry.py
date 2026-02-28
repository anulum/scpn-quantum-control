"""Verify experiment registry completeness and internal consistency."""

from __future__ import annotations

import inspect

from scpn_quantum_control.hardware import experiments as exp_mod
from scpn_quantum_control.hardware.experiments import ALL_EXPERIMENTS


def test_all_experiment_functions_registered():
    """Every *_experiment function in experiments.py is in ALL_EXPERIMENTS."""
    func_objects = {
        obj
        for name, obj in inspect.getmembers(exp_mod, inspect.isfunction)
        if name.endswith("_experiment") and not name.startswith("_")
    }
    registered = set(ALL_EXPERIMENTS.values())
    missing = func_objects - registered
    missing_names = {f.__name__ for f in missing}
    assert not missing_names, f"Unregistered experiment functions: {missing_names}"


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
