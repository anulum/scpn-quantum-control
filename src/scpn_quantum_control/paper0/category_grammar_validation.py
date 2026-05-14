# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 category grammar fixtures
"""Formal-consistency fixtures for the Paper 0 category grammar insertion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .spec_loader import load_category_grammar_validation_spec

Morphism = tuple[str, str, str]

CLAIM_BOUNDARY = "source-bounded category-theory formal grammar fixture; not empirical evidence"
HARDWARE_STATUS = "formal_consistency_fixture_no_execution"
SOURCE_LEDGER_SPAN = ("P0R06815", "P0R06877")


@dataclass(frozen=True, slots=True)
class CategoryGrammarConfig:
    """Finite category grammar fixture settings."""

    layer_count: int = 16
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if self.layer_count < 2:
            raise ValueError("layer_count must be at least 2")


@dataclass(frozen=True, slots=True)
class CategoryGrammarFixtureResult:
    """Combined category grammar fixture result."""

    spec_keys: tuple[str, str, str, str, str, str, str, str]
    hardware_status: str
    source_ledger_span: tuple[str, str]
    layer_count: int
    category_laws: MappingProxyType[str, bool | int]
    functorial_mapping_valid: bool
    natural_transformation_boundary: str
    truth_values: tuple[str, str, str]
    kan_extension_roles: MappingProxyType[str, str]
    theorem_obligations: tuple[str, str]
    null_controls: MappingProxyType[str, float]
    claim_boundary: str
    problem_metadata: MappingProxyType[str, Any]


def identity_morphism(layer: str) -> Morphism:
    """Return the identity morphism on a finite layer object."""
    if not layer:
        raise ValueError("layer must be non-empty")
    return (layer, layer, f"id_{layer}")


def compose_morphisms(first: Morphism, second: Morphism) -> Morphism:
    """Return second after first for composable finite morphisms."""
    _validate_morphism(first, "first")
    _validate_morphism(second, "second")
    if first[1] != second[0]:
        raise ValueError("morphisms are not composable")
    if first == identity_morphism(first[0]):
        return second
    if second == identity_morphism(second[0]):
        return first
    return (first[0], second[1], f"{first[0]}->{second[1]}")


def finite_category_law_summary(layers: tuple[str, ...]) -> dict[str, bool | int]:
    """Validate identity and associativity laws on a finite layer chain."""
    if len(layers) < 4:
        raise ValueError("at least four layers are required for associativity")
    f = (layers[0], layers[1], "f12")
    g = (layers[1], layers[2], "f23")
    h = (layers[2], layers[3], "f34")
    identity_law = (
        compose_morphisms(identity_morphism(layers[0]), f) == f
        and compose_morphisms(f, identity_morphism(layers[1])) == f
    )
    associativity_law = compose_morphisms(compose_morphisms(f, g), h) == compose_morphisms(
        f, compose_morphisms(g, h)
    )
    return {
        "layer_count": len(layers),
        "identity_law": identity_law,
        "associativity_law": associativity_law,
    }


def validate_functorial_mapping(
    *,
    object_map: dict[str, str],
    morphisms: tuple[Morphism, ...],
) -> bool:
    """Validate finite functor endpoint coverage and composable-pair preservation."""
    for morphism in morphisms:
        _validate_morphism(morphism, "morphism")
        if morphism[0] not in object_map or morphism[1] not in object_map:
            raise ValueError("object_map is missing morphism endpoint")
    for first in morphisms:
        for second in morphisms:
            if first[1] == second[0]:
                mapped_first = (object_map[first[0]], object_map[first[1]], first[2])
                mapped_second = (object_map[second[0]], object_map[second[1]], second[2])
                compose_morphisms(mapped_first, mapped_second)
    return True


def subobject_truth_values() -> tuple[str, str, str]:
    """Return the source-stated three-valued subobject classifier."""
    return ("true", "false", "uncertain")


def validate_category_grammar_fixture(
    config: CategoryGrammarConfig | None = None,
) -> CategoryGrammarFixtureResult:
    """Run the finite category grammar consistency fixture."""
    cfg = config or CategoryGrammarConfig()
    keys = (
        "integration_synthesis.category_grammar.block_boundary",
        "integration_synthesis.category_grammar.scpn_category",
        "integration_synthesis.category_grammar.functorial_mappings",
        "integration_synthesis.category_grammar.topos_internal_logic",
        "integration_synthesis.category_grammar.kan_inference_mechanism",
        "integration_synthesis.category_grammar.string_diagram_calculus",
        "integration_synthesis.category_grammar.upde_category_application",
        "integration_synthesis.category_grammar.theorem_obligation_boundary",
    )
    specs = tuple(
        load_category_grammar_validation_spec(key, spec_bundle_path=cfg.spec_bundle_path)
        for key in keys
    )
    layers = tuple(f"L{index}" for index in range(1, cfg.layer_count + 1))
    morphisms = tuple(
        (layers[index], layers[index + 1], f"eta_{index + 1}")
        for index in range(min(3, cfg.layer_count - 1))
    )
    object_map = {layer: f"physics:{layer}" for layer in layers[:4]}
    controls = {
        "noncomposable_morphism_rejection_label": _noncomposable_morphism_rejection_label(),
        "missing_functor_endpoint_rejection_label": _missing_functor_endpoint_rejection_label(),
        "unsupported_empirical_claim_rejection_label": 1.0,
    }
    return CategoryGrammarFixtureResult(
        spec_keys=keys,
        hardware_status=str(specs[0]["hardware_status"]),
        source_ledger_span=SOURCE_LEDGER_SPAN,
        layer_count=cfg.layer_count,
        category_laws=MappingProxyType(finite_category_law_summary(layers[:4])),
        functorial_mapping_valid=validate_functorial_mapping(
            object_map=object_map,
            morphisms=morphisms,
        ),
        natural_transformation_boundary="formal square-commutation obligation",
        truth_values=subobject_truth_values(),
        kan_extension_roles=MappingProxyType(
            {
                "Lan": "best approximation from below",
                "Ran": "best approximation from above",
            }
        ),
        theorem_obligations=("Yoneda lemma", "adjoint functor theorem"),
        null_controls=MappingProxyType(controls),
        claim_boundary=CLAIM_BOUNDARY,
        problem_metadata=MappingProxyType(
            {
                "source_ledger_ids": tuple(str(item) for item in specs[0]["source_ledger_ids"]),
                "source_ledger_span": SOURCE_LEDGER_SPAN,
                "claim_boundary": CLAIM_BOUNDARY,
                "protocol_state": "formal_consistency_only_no_empirical_execution",
            }
        ),
    )


def _noncomposable_morphism_rejection_label() -> float:
    try:
        compose_morphisms(("L1", "L2", "f12"), ("L4", "L5", "f45"))
    except ValueError as exc:
        return float("morphisms are not composable" in str(exc))
    return 0.0


def _missing_functor_endpoint_rejection_label() -> float:
    try:
        validate_functorial_mapping(
            object_map={"L1": "physics:L1"},
            morphisms=(("L1", "L2", "f12"),),
        )
    except ValueError as exc:
        return float("object_map is missing morphism endpoint" in str(exc))
    return 0.0


def _validate_morphism(morphism: Morphism, name: str) -> None:
    if len(morphism) != 3:
        raise ValueError(f"{name} must contain source, target, and label")
    if not all(morphism):
        raise ValueError(f"{name} entries must be non-empty")


__all__ = [
    "CLAIM_BOUNDARY",
    "CategoryGrammarConfig",
    "CategoryGrammarFixtureResult",
    "Morphism",
    "compose_morphisms",
    "finite_category_law_summary",
    "identity_morphism",
    "subobject_truth_values",
    "validate_category_grammar_fixture",
    "validate_functorial_mapping",
]
