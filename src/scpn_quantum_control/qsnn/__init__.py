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
