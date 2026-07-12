#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — verify hardware result packs script
# scpn-quantum-control -- hardware result-pack verifier script
"""Command-line wrapper for hardware result-pack verification.

The verifier logic (:mod:`scpn_quantum_control.hardware_result_packs`) depends
only on the standard library, but importing it through the package root would
execute :mod:`scpn_quantum_control`'s ``__init__``, which eagerly imports the
full quantum stack (Qiskit, oscillatools, and the rest). To keep this integrity
tool runnable with minimal dependencies — the standard library and the JSON/tar
paths it verifies — the module is located with :func:`importlib.util.find_spec`
(which locates the package without executing its ``__init__``) and loaded
directly from its file. The verifier therefore runs in a bare or partial
environment where the accelerator stack is absent.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path


def load_verifier_main() -> Callable[[], int]:
    """Load the stdlib-only pack verifier without importing the package root.

    Returns
    -------
    Callable[[], int]
        The ``main`` entry point of
        :mod:`scpn_quantum_control.hardware_result_packs`, loaded directly from
        its file so no package ``__init__`` side effects (Qiskit, oscillatools)
        are triggered.

    Raises
    ------
    ModuleNotFoundError
        If the ``scpn_quantum_control`` package or the verifier module file
        cannot be located.
    """
    package_spec = importlib.util.find_spec("scpn_quantum_control")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise ModuleNotFoundError("scpn_quantum_control package could not be located")
    module_path = Path(package_spec.submodule_search_locations[0]) / "hardware_result_packs.py"
    module_spec = importlib.util.spec_from_file_location(
        "scpn_quantum_control._hardware_result_packs_standalone", module_path
    )
    if module_spec is None or module_spec.loader is None:
        raise ModuleNotFoundError(f"cannot load the pack verifier from {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    verifier_main: Callable[[], int] = module.main
    return verifier_main


if __name__ == "__main__":
    raise SystemExit(load_verifier_main()())
