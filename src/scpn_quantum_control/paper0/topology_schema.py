# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 topology schema
"""Source-anchored Paper 0 topology boundary.

This module deliberately records provenance and structural coupling classes,
not a numeric ``K_nm`` matrix. It is the boundary object needed before S19
resource-signature scans or provider replication plans may be reconnected to
Paper 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Paper0TopologyValidationError(ValueError):
    """Raised when a Paper 0 topology schema violates provenance policy."""


@dataclass(frozen=True, slots=True)
class Paper0TopologyLayer:
    """One Paper 0 layer or meta-layer source record."""

    layer_id: int
    name: str
    source_ledger_ids: tuple[str, ...]
    oscillator_population: str
    is_meta_layer: bool = False


@dataclass(frozen=True, slots=True)
class Paper0IntraLayerTopology:
    """Intra-layer coupling family without committing to a numeric matrix."""

    layer_id: int
    coupling_symbol: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    topology_status: str
    validation_role: str


@dataclass(frozen=True, slots=True)
class Paper0InterLayerEdge:
    """Directed source-anchored relation between layers."""

    source_layer: int
    target_layer: int
    channel: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    description: str


@dataclass(frozen=True, slots=True)
class Paper0FieldPort:
    """Global field coupling port for UPDE field terms."""

    key: str
    couples_to_layers: tuple[int, ...]
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    variables: tuple[str, ...]
    description: str


@dataclass(frozen=True, slots=True)
class Paper0AdaptiveParameterSet:
    """Adaptive coupling and quasicritical controller parameter family."""

    key: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    variables: tuple[str, ...]
    guarded_controls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Paper0SyntheticControl:
    """Synthetic topology used only as a validation control."""

    key: str
    topology: str
    is_control: bool
    provenance_status: str
    purpose: str


@dataclass(frozen=True, slots=True)
class Paper0TopologySchema:
    """Complete source-boundary schema for Paper 0 topology work."""

    layers: tuple[Paper0TopologyLayer, ...]
    intra_layer_topologies: tuple[Paper0IntraLayerTopology, ...]
    inter_layer_edges: tuple[Paper0InterLayerEdge, ...]
    field_ports: dict[str, Paper0FieldPort]
    adaptive_parameters: dict[str, Paper0AdaptiveParameterSet]
    synthetic_controls: tuple[Paper0SyntheticControl, ...]
    numeric_coupling_matrix: None
    provenance_status: str
    provider_ready: bool


LAYER_SOURCE_LEDGER_IDS: dict[int, tuple[str, ...]] = {
    1: ("P0R00031",),
    2: ("P0R00032",),
    3: ("P0R00033",),
    4: ("P0R00034",),
    5: ("P0R00036",),
    6: ("P0R00037",),
    7: ("P0R00038",),
    8: ("P0R00039",),
    9: ("P0R00041",),
    10: ("P0R00042",),
    11: ("P0R00043",),
    12: ("P0R00044",),
    13: ("P0R00046",),
    14: ("P0R00047",),
    15: ("P0R00048",),
    16: ("P0R00049", "P0R00050", "P0R00574"),
}

LAYER_NAMES: dict[int, str] = {
    1: "Quantum Biological",
    2: "Neurochemical-Neurological",
    3: "Genomic-Epigenomic-Morphogenetic",
    4: "Cellular-Tissue Synchronisation",
    5: "Organismal-Psychoemotional Feedback",
    6: "Planetary-Biospheric",
    7: "Geometrical-Symbolic",
    8: "Cosmic Phase-Locking",
    9: "Memory Imprint-Existential Holograph",
    10: "Projective Field Boundary Control",
    11: "Noospheric-Cultural-Informational",
    12: "Ecological-Gaian Synchrony",
    13: "Source-Field / Meta-Universal",
    14: "Transdimensional Resonance",
    15: "Consilium / Oversoul Integrator",
    16: "Cybernetic Closure",
}

UPDE_INTRA_LAYER_EQUATIONS = ("EQ0003", "EQ0032", "EQ0037", "EQ0039", "EQ0129")
UPDE_INTRA_LAYER_LEDGERS = ("P0R00520", "P0R02507", "P0R02530", "P0R02622", "P0R06120")
INTER_LAYER_EQUATIONS = ("EQ0033", "EQ0040")
INTER_LAYER_LEDGERS = ("P0R02510", "P0R02630", "P0R02559")
FIELD_EQUATIONS = ("EQ0034", "EQ0041", "EQ0043")
FIELD_LEDGERS = ("P0R02512", "P0R02634", "P0R02644")
ADAPTIVE_EQUATIONS = ("EQ0045",)
ADAPTIVE_LEDGERS = ("P0R02910",)


def build_paper0_topology_schema() -> Paper0TopologySchema:
    """Build and validate the conservative Paper 0 topology source boundary."""
    layers = tuple(
        Paper0TopologyLayer(
            layer_id=layer_id,
            name=LAYER_NAMES[layer_id],
            source_ledger_ids=LAYER_SOURCE_LEDGER_IDS[layer_id],
            oscillator_population="finite Paper 0 layer population; cardinality unresolved",
            is_meta_layer=layer_id == 16,
        )
        for layer_id in range(1, 17)
    )
    schema = Paper0TopologySchema(
        layers=layers,
        intra_layer_topologies=_build_intra_layer_topologies(),
        inter_layer_edges=_build_inter_layer_edges(),
        field_ports={
            "global_psi_field": Paper0FieldPort(
                key="global_psi_field",
                couples_to_layers=tuple(range(1, 16)),
                source_equation_ids=FIELD_EQUATIONS,
                source_ledger_ids=FIELD_LEDGERS,
                variables=("zeta_L", "Psi_Global", "Theta_Psi", "theta_i_L"),
                description="UPDE global field-coupling port C_Field.",
            )
        },
        adaptive_parameters={
            "quasicritical_controller": Paper0AdaptiveParameterSet(
                key="quasicritical_controller",
                source_equation_ids=ADAPTIVE_EQUATIONS,
                source_ledger_ids=ADAPTIVE_LEDGERS,
                variables=(
                    "K_ij_L",
                    "gamma_L",
                    "R_L",
                    "R_L_star",
                    "lambda_L",
                    "xi_ij_L",
                    "eta_L",
                    "alpha_L",
                    "sigma_L",
                ),
                guarded_controls=("bounded_update", "zero_gain_null", "wrong_sign_feedback"),
            )
        },
        synthetic_controls=_build_synthetic_controls(),
        numeric_coupling_matrix=None,
        provenance_status="paper0_source_anchored_no_numeric_matrix",
        provider_ready=False,
    )
    validate_paper0_topology_schema(schema)
    return schema


def validate_paper0_topology_schema(schema: Paper0TopologySchema) -> None:
    """Validate the source-boundary invariants for the Paper 0 topology schema."""
    layer_ids = tuple(layer.layer_id for layer in schema.layers)
    if layer_ids != tuple(range(1, 17)):
        raise Paper0TopologyValidationError("Paper 0 topology must contain layers 1 through 16")
    if not schema.layers[-1].is_meta_layer:
        raise Paper0TopologyValidationError("Layer 16 must be marked as the meta-layer")
    if schema.numeric_coupling_matrix is not None:
        raise Paper0TopologyValidationError(
            "numeric coupling matrix is not allowed in source boundary"
        )
    if schema.provider_ready:
        raise Paper0TopologyValidationError("source boundary must not be provider ready")
    for layer in schema.layers:
        if not layer.source_ledger_ids:
            raise Paper0TopologyValidationError(f"layer {layer.layer_id} missing source anchor")
    for topology in schema.intra_layer_topologies:
        if not topology.source_equation_ids or not topology.source_ledger_ids:
            raise Paper0TopologyValidationError(
                f"layer {topology.layer_id} intra-layer topology missing source anchor"
            )
    for edge in schema.inter_layer_edges:
        _validate_layer_ref(edge.source_layer, layer_ids)
        _validate_layer_ref(edge.target_layer, layer_ids)
        if not edge.source_equation_ids and not edge.source_ledger_ids:
            raise Paper0TopologyValidationError(f"{edge.channel} edge missing source anchor")
    for port in schema.field_ports.values():
        if not port.source_equation_ids or not port.source_ledger_ids:
            raise Paper0TopologyValidationError(f"field port {port.key} missing source anchor")
    for parameters in schema.adaptive_parameters.values():
        if not parameters.source_equation_ids or not parameters.source_ledger_ids:
            raise Paper0TopologyValidationError(
                f"adaptive parameter set {parameters.key} missing source anchor"
            )
    for control in schema.synthetic_controls:
        if (
            not control.is_control
            or control.provenance_status != "synthetic_control_not_paper0_topology"
        ):
            raise Paper0TopologyValidationError(
                f"synthetic control {control.key} must be labelled as a non-Paper-0 control"
            )


def schema_to_s19_source_boundary(schema: Paper0TopologySchema) -> dict[str, Any]:
    """Export a non-provider source boundary for S19 resource-signature scans."""
    validate_paper0_topology_schema(schema)
    return {
        "schema_key": "paper0.topology.source_boundary.v1",
        "provenance_status": schema.provenance_status,
        "provider_ready": schema.provider_ready,
        "hardware_status": "source_boundary_only_no_provider_submission",
        "numeric_coupling_matrix": schema.numeric_coupling_matrix,
        "layer_count": len(schema.layers),
        "meta_layer": 16,
        "layers": [
            {
                "layer_id": layer.layer_id,
                "name": layer.name,
                "source_ledger_ids": list(layer.source_ledger_ids),
                "oscillator_population": layer.oscillator_population,
                "is_meta_layer": layer.is_meta_layer,
            }
            for layer in schema.layers
        ],
        "inter_layer_edges": [
            {
                "source_layer": edge.source_layer,
                "target_layer": edge.target_layer,
                "channel": edge.channel,
                "source_equation_ids": list(edge.source_equation_ids),
                "source_ledger_ids": list(edge.source_ledger_ids),
                "description": edge.description,
            }
            for edge in schema.inter_layer_edges
        ],
        "field_ports": {
            key: {
                "couples_to_layers": list(port.couples_to_layers),
                "source_equation_ids": list(port.source_equation_ids),
                "source_ledger_ids": list(port.source_ledger_ids),
                "variables": list(port.variables),
                "description": port.description,
            }
            for key, port in schema.field_ports.items()
        },
        "adaptive_parameters": {
            key: {
                "source_equation_ids": list(parameters.source_equation_ids),
                "source_ledger_ids": list(parameters.source_ledger_ids),
                "variables": list(parameters.variables),
                "guarded_controls": list(parameters.guarded_controls),
            }
            for key, parameters in schema.adaptive_parameters.items()
        },
        "synthetic_controls": [control.key for control in schema.synthetic_controls],
        "source_equation_ids": sorted(_collect_equation_ids(schema)),
        "source_ledger_ids": sorted(_collect_ledger_ids(schema)),
        "policy": (
            "This source boundary does not export a numeric K_nm matrix. "
            "Synthetic chain, ring, and complete graphs remain controls only."
        ),
    }


def _build_intra_layer_topologies() -> tuple[Paper0IntraLayerTopology, ...]:
    return tuple(
        Paper0IntraLayerTopology(
            layer_id=layer_id,
            coupling_symbol="K_ij^L",
            source_equation_ids=UPDE_INTRA_LAYER_EQUATIONS,
            source_ledger_ids=UPDE_INTRA_LAYER_LEDGERS,
            topology_status="source_symbolic_unresolved_cardinality",
            validation_role="Paper 0 symbolic intra-layer coupling boundary",
        )
        for layer_id in range(1, 16)
    )


def _build_inter_layer_edges() -> tuple[Paper0InterLayerEdge, ...]:
    downward = tuple(
        Paper0InterLayerEdge(
            source_layer=layer,
            target_layer=layer - 1,
            channel="downward_generation",
            source_equation_ids=INTER_LAYER_EQUATIONS,
            source_ledger_ids=INTER_LAYER_LEDGERS + ("P0R00574", "P0R00575"),
            description="top-down generative projection in the Paper 0 hierarchy",
        )
        for layer in range(15, 1, -1)
    )
    upward = tuple(
        Paper0InterLayerEdge(
            source_layer=layer,
            target_layer=layer + 1,
            channel="upward_inference",
            source_equation_ids=INTER_LAYER_EQUATIONS,
            source_ledger_ids=INTER_LAYER_LEDGERS + ("P0R00574", "P0R00575"),
            description="bottom-up inference or prediction-error flow in the Paper 0 hierarchy",
        )
        for layer in range(1, 15)
    )
    closure = (
        Paper0InterLayerEdge(
            source_layer=1,
            target_layer=16,
            channel="recursive_closure",
            source_equation_ids=(),
            source_ledger_ids=("P0R00574", "P0R00575", "P0R02559"),
            description="bottom-up feedback enters the layer-16 recursive supervisor",
        ),
        Paper0InterLayerEdge(
            source_layer=16,
            target_layer=15,
            channel="recursive_closure",
            source_equation_ids=(),
            source_ledger_ids=("P0R00574", "P0R00575", "P0R02559"),
            description="layer-16 supervisory recursion returns control to layer 15",
        ),
    )
    return downward + upward + closure


def _build_synthetic_controls() -> tuple[Paper0SyntheticControl, ...]:
    return (
        Paper0SyntheticControl(
            key="control.chain",
            topology="open_chain",
            is_control=True,
            provenance_status="synthetic_control_not_paper0_topology",
            purpose="finite-size boundary control for source-anchored topology experiments",
        ),
        Paper0SyntheticControl(
            key="control.ring",
            topology="periodic_ring",
            is_control=True,
            provenance_status="synthetic_control_not_paper0_topology",
            purpose="periodic-boundary control for source-anchored topology experiments",
        ),
        Paper0SyntheticControl(
            key="control.complete",
            topology="complete_graph",
            is_control=True,
            provenance_status="synthetic_control_not_paper0_topology",
            purpose="all-to-all coupling control for source-anchored topology experiments",
        ),
    )


def _validate_layer_ref(layer_id: int, layer_ids: tuple[int, ...]) -> None:
    if layer_id not in layer_ids:
        raise Paper0TopologyValidationError(f"unknown layer reference: {layer_id}")


def _collect_equation_ids(schema: Paper0TopologySchema) -> set[str]:
    equation_ids: set[str] = set()
    for topology in schema.intra_layer_topologies:
        equation_ids.update(topology.source_equation_ids)
    for edge in schema.inter_layer_edges:
        equation_ids.update(edge.source_equation_ids)
    for port in schema.field_ports.values():
        equation_ids.update(port.source_equation_ids)
    for parameters in schema.adaptive_parameters.values():
        equation_ids.update(parameters.source_equation_ids)
    return equation_ids


def _collect_ledger_ids(schema: Paper0TopologySchema) -> set[str]:
    ledger_ids: set[str] = set()
    for layer in schema.layers:
        ledger_ids.update(layer.source_ledger_ids)
    for topology in schema.intra_layer_topologies:
        ledger_ids.update(topology.source_ledger_ids)
    for edge in schema.inter_layer_edges:
        ledger_ids.update(edge.source_ledger_ids)
    for port in schema.field_ports.values():
        ledger_ids.update(port.source_ledger_ids)
    for parameters in schema.adaptive_parameters.values():
        ledger_ids.update(parameters.source_ledger_ids)
    return ledger_ids


__all__ = [
    "Paper0AdaptiveParameterSet",
    "Paper0FieldPort",
    "Paper0InterLayerEdge",
    "Paper0IntraLayerTopology",
    "Paper0SyntheticControl",
    "Paper0TopologyLayer",
    "Paper0TopologySchema",
    "Paper0TopologyValidationError",
    "build_paper0_topology_schema",
    "schema_to_s19_source_boundary",
    "validate_paper0_topology_schema",
]
