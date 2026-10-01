# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Spiking Neural Network
"""Expose quantum spiking neural-network primitives and training surfaces.

The facade groups the LIF neuron, bounded synapse and STDP rule, dense quantum
layer, local trainer result/diagnostic contracts, and neuromorphic bridge
configuration, state, result, and explicit claim-boundary surfaces.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .qlayer import QuantumDenseLayer
    from .qlif import QuantumLIFNeuron
    from .qstdp import QuantumSTDP
    from .qsynapse import QuantumSynapse
    from .quantum_neuromorphic_bridge import (
        CLAIM_BOUNDARY,
        DynamicCouplingConfig,
        NeuromorphicStepResult,
        QuantumLIFConfig,
        QuantumNeuromorphicBridge,
        TraceSTDPConfig,
        TraceSTDPState,
    )
    from .training import (
        QSNNParameterShiftDescentRun,
        QSNNTrainer,
        QSNNTrainingDiagnostics,
        QSNNTrainingRun,
    )

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "QuantumDenseLayer": ("scpn_quantum_control.qsnn.qlayer", "QuantumDenseLayer"),
    "QuantumLIFNeuron": ("scpn_quantum_control.qsnn.qlif", "QuantumLIFNeuron"),
    "QuantumSTDP": ("scpn_quantum_control.qsnn.qstdp", "QuantumSTDP"),
    "QuantumSynapse": ("scpn_quantum_control.qsnn.qsynapse", "QuantumSynapse"),
    "CLAIM_BOUNDARY": ("scpn_quantum_control.qsnn.quantum_neuromorphic_bridge", "CLAIM_BOUNDARY"),
    "DynamicCouplingConfig": (
        "scpn_quantum_control.qsnn.quantum_neuromorphic_bridge",
        "DynamicCouplingConfig",
    ),
    "NeuromorphicStepResult": (
        "scpn_quantum_control.qsnn.quantum_neuromorphic_bridge",
        "NeuromorphicStepResult",
    ),
    "QuantumLIFConfig": (
        "scpn_quantum_control.qsnn.quantum_neuromorphic_bridge",
        "QuantumLIFConfig",
    ),
    "QuantumNeuromorphicBridge": (
        "scpn_quantum_control.qsnn.quantum_neuromorphic_bridge",
        "QuantumNeuromorphicBridge",
    ),
    "TraceSTDPConfig": (
        "scpn_quantum_control.qsnn.quantum_neuromorphic_bridge",
        "TraceSTDPConfig",
    ),
    "TraceSTDPState": ("scpn_quantum_control.qsnn.quantum_neuromorphic_bridge", "TraceSTDPState"),
    "QSNNParameterShiftDescentRun": (
        "scpn_quantum_control.qsnn.training",
        "QSNNParameterShiftDescentRun",
    ),
    "QSNNTrainer": ("scpn_quantum_control.qsnn.training", "QSNNTrainer"),
    "QSNNTrainingDiagnostics": ("scpn_quantum_control.qsnn.training", "QSNNTrainingDiagnostics"),
    "QSNNTrainingRun": ("scpn_quantum_control.qsnn.training", "QSNNTrainingRun"),
}


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS))


__all__ = [
    "CLAIM_BOUNDARY",
    "DynamicCouplingConfig",
    "NeuromorphicStepResult",
    "QuantumDenseLayer",
    "QuantumLIFConfig",
    "QuantumLIFNeuron",
    "QuantumNeuromorphicBridge",
    "QuantumSTDP",
    "QuantumSynapse",
    "QSNNTrainer",
    "QSNNParameterShiftDescentRun",
    "QSNNTrainingDiagnostics",
    "QSNNTrainingRun",
    "TraceSTDPConfig",
    "TraceSTDPState",
]
