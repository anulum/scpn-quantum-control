# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD primitive registry contracts
"""Primitive registry contracts and Program AD registry-dispatch coverage.

This module owns the typed primitive identity, transform, contract, and
registry-dispatch coverage surfaces used by differentiable programming. The
larger differentiable facade imports these symbols for backwards compatibility
while primitive-family registration code continues to populate the shared
default registry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

VectorObjective = Callable[[NDArray[np.float64]], ArrayLike]
CustomJVPRule = Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]
CustomVJPRule = Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]
PrimitiveBatchingRule = Callable[
    [Callable[..., object], tuple[object, ...], tuple[int | None, ...], int], object
]


@dataclass(frozen=True)
class CustomDerivativeRule:
    """Exact custom derivative rules for one differentiable vector primitive."""

    name: str
    value_fn: VectorObjective
    jvp_rule: CustomJVPRule | None = None
    vjp_rule: CustomVJPRule | None = None
    parameter_names: tuple[str, ...] = ()
    trainable: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("custom derivative rule name must be non-empty")
        if not callable(self.value_fn):
            raise ValueError("custom derivative value_fn must be callable")
        if self.jvp_rule is None and self.vjp_rule is None:
            raise ValueError("custom derivative rule requires a JVP or VJP rule")
        if self.jvp_rule is not None and not callable(self.jvp_rule):
            raise ValueError("custom derivative jvp_rule must be callable")
        if self.vjp_rule is not None and not callable(self.vjp_rule):
            raise ValueError("custom derivative vjp_rule must be callable")
        if any(not isinstance(name, str) or not name for name in self.parameter_names):
            raise ValueError("parameter_names must contain non-empty strings")
        if any(not isinstance(flag, bool) for flag in self.trainable):
            raise ValueError("trainable mask must contain booleans")
        if (
            self.parameter_names
            and self.trainable
            and len(self.parameter_names) != len(self.trainable)
        ):
            raise ValueError("parameter_names and trainable mask lengths must match")


@dataclass(frozen=True)
class PrimitiveIdentity:
    """Stable typed identity for a differentiable primitive implementation."""

    namespace: str
    name: str
    version: str = "1"

    def __post_init__(self) -> None:
        namespace = _normalise_identity_token("primitive namespace", self.namespace)
        name = _normalise_identity_token("primitive name", self.name)
        version = _normalise_identity_token("primitive version", self.version)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)

    @property
    def key(self) -> str:
        """Return the canonical registry key for this primitive identity."""

        return f"{self.namespace}:{self.name}@{self.version}"

    @staticmethod
    def parse(identity: PrimitiveIdentity | str) -> PrimitiveIdentity:
        """Return a typed identity from an existing identity or canonical string."""

        if isinstance(identity, PrimitiveIdentity):
            return identity
        if not isinstance(identity, str) or not identity:
            raise ValueError("primitive identity must be a PrimitiveIdentity or non-empty string")
        if "@" in identity:
            stem, version = identity.rsplit("@", 1)
        else:
            stem, version = identity, "1"
        if ":" not in stem:
            raise ValueError(
                "primitive identity string must use 'namespace:name[@version]' format"
            )
        namespace, name = stem.split(":", 1)
        return PrimitiveIdentity(namespace=namespace, name=name, version=version)


def _normalise_identity_token(name: str, value: object) -> str:
    """Return a registry-safe identity token."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if any(character.isspace() for character in value):
        raise ValueError(f"{name} must not contain whitespace")
    if any(character in value for character in (":", "@")):
        raise ValueError(f"{name} must not contain ':' or '@'")
    return value


PrimitiveLoweringRule = Callable[[CustomDerivativeRule], object]
PrimitiveShapeRule = Callable[[tuple[object, ...]], tuple[int, ...]]
PrimitiveDTypeRule = Callable[[tuple[object, ...]], str]
PrimitiveStaticArgumentRule = Callable[[tuple[object, ...]], tuple[object, ...]]


@dataclass(frozen=True)
class PrimitiveTransformRule:
    """Combined transform binding for one differentiable primitive identity."""

    identity: PrimitiveIdentity
    derivative_rule: CustomDerivativeRule
    batching_rule: PrimitiveBatchingRule | None = None
    lowering_rule: PrimitiveLoweringRule | None = None
    lowering_metadata: Mapping[str, str] | None = None
    shape_rule: PrimitiveShapeRule | None = None
    dtype_rule: PrimitiveDTypeRule | None = None
    static_argument_rule: PrimitiveStaticArgumentRule | None = None
    nondifferentiable_policy: str = "not_declared"
    effect: str = "pure"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PrimitiveIdentity):
            raise ValueError("transform identity must be a PrimitiveIdentity")
        if not isinstance(self.derivative_rule, CustomDerivativeRule):
            raise ValueError("transform derivative_rule must be a CustomDerivativeRule")
        if self.batching_rule is not None and not callable(self.batching_rule):
            raise ValueError("transform batching_rule must be callable")
        if self.lowering_rule is not None and not callable(self.lowering_rule):
            raise ValueError("transform lowering_rule must be callable")
        if self.shape_rule is not None and not callable(self.shape_rule):
            raise ValueError("transform shape_rule must be callable")
        if self.dtype_rule is not None and not callable(self.dtype_rule):
            raise ValueError("transform dtype_rule must be callable")
        if self.static_argument_rule is not None and not callable(self.static_argument_rule):
            raise ValueError("transform static_argument_rule must be callable")
        if not isinstance(self.nondifferentiable_policy, str) or not self.nondifferentiable_policy:
            raise ValueError("transform nondifferentiable_policy must be a non-empty string")
        if not isinstance(self.effect, str) or not self.effect:
            raise ValueError("transform effect must be a non-empty string")
        metadata = {} if self.lowering_metadata is None else dict(self.lowering_metadata)
        if any(not isinstance(key, str) or not key for key in metadata):
            raise ValueError("lowering metadata keys must be non-empty strings")
        if any(not isinstance(value, str) or not value for value in metadata.values()):
            raise ValueError("lowering metadata values must be non-empty strings")
        object.__setattr__(self, "lowering_metadata", metadata)


@dataclass(frozen=True)
class PrimitiveContract:
    """Unified registered contract for one differentiable primitive identity."""

    identity: PrimitiveIdentity
    derivative_rule: CustomDerivativeRule
    batching_rule: PrimitiveBatchingRule | None
    lowering_rule: PrimitiveLoweringRule | None
    lowering_metadata: Mapping[str, str]
    shape_rule: PrimitiveShapeRule | None
    dtype_rule: PrimitiveDTypeRule | None
    static_argument_rule: PrimitiveStaticArgumentRule | None
    nondifferentiable_policy: str
    effect: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PrimitiveIdentity):
            raise ValueError("contract identity must be a PrimitiveIdentity")
        if not isinstance(self.derivative_rule, CustomDerivativeRule):
            raise ValueError("contract derivative_rule must be a CustomDerivativeRule")
        if self.batching_rule is not None and not callable(self.batching_rule):
            raise ValueError("contract batching_rule must be callable")
        if self.lowering_rule is not None and not callable(self.lowering_rule):
            raise ValueError("contract lowering_rule must be callable")
        if self.shape_rule is not None and not callable(self.shape_rule):
            raise ValueError("contract shape_rule must be callable")
        if self.dtype_rule is not None and not callable(self.dtype_rule):
            raise ValueError("contract dtype_rule must be callable")
        if self.static_argument_rule is not None and not callable(self.static_argument_rule):
            raise ValueError("contract static_argument_rule must be callable")
        if not isinstance(self.nondifferentiable_policy, str) or not self.nondifferentiable_policy:
            raise ValueError("contract nondifferentiable_policy must be a non-empty string")
        if not isinstance(self.effect, str) or not self.effect:
            raise ValueError("contract effect must be a non-empty string")
        metadata = dict(self.lowering_metadata)
        if any(not isinstance(key, str) or not key for key in metadata):
            raise ValueError("contract lowering metadata keys must be non-empty strings")
        if any(not isinstance(value, str) or not value for value in metadata.values()):
            raise ValueError("contract lowering metadata values must be non-empty strings")
        object.__setattr__(self, "lowering_metadata", metadata)

    @staticmethod
    def from_transform(transform: PrimitiveTransformRule) -> PrimitiveContract:
        """Build an immutable primitive contract view from a transform binding."""

        return PrimitiveContract(
            identity=transform.identity,
            derivative_rule=transform.derivative_rule,
            batching_rule=transform.batching_rule,
            lowering_rule=transform.lowering_rule,
            lowering_metadata={}
            if transform.lowering_metadata is None
            else transform.lowering_metadata,
            shape_rule=transform.shape_rule,
            dtype_rule=transform.dtype_rule,
            static_argument_rule=transform.static_argument_rule,
            nondifferentiable_policy=transform.nondifferentiable_policy,
            effect=transform.effect,
        )


PROGRAM_AD_REGISTRY_DISPATCH_CLAIM_BOUNDARY = (
    "registry-dispatched Program AD primitive coverage over declared derivative, "
    "batching, lowering metadata, shape, dtype, static-argument, "
    "nondifferentiability, and effect contracts only; not executable Rust, LLVM, "
    "JIT, provider, hardware, or performance evidence"
)


@dataclass(frozen=True)
class ProgramADRegistryDispatchCoverageRow:
    """Coverage status for one Program AD primitive resolved through the registry."""

    family: str
    primitive: str
    identity: str
    derivative_rule: str | None
    has_batching_rule: bool
    has_lowering_rule: bool
    has_lowering_metadata: bool
    has_shape_rule: bool
    has_dtype_rule: bool
    has_static_argument_rule: bool
    nondifferentiable_policy: str | None
    effect: str | None
    lowering_metadata_keys: tuple[str, ...]
    complete: bool
    blocked_reasons: tuple[str, ...]
    claim_boundary: str = PROGRAM_AD_REGISTRY_DISPATCH_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("registry dispatch coverage family must be non-empty")
        if not self.primitive:
            raise ValueError("registry dispatch coverage primitive must be non-empty")
        if not self.identity:
            raise ValueError("registry dispatch coverage identity must be non-empty")
        if self.derivative_rule is not None and not self.derivative_rule:
            raise ValueError("registry dispatch coverage derivative_rule must be non-empty")
        if any(not key for key in self.lowering_metadata_keys):
            raise ValueError("registry dispatch coverage metadata keys must be non-empty")
        if any(not reason for reason in self.blocked_reasons):
            raise ValueError("registry dispatch coverage blocked reasons must be non-empty")
        if self.complete and self.blocked_reasons:
            raise ValueError("complete registry dispatch coverage rows cannot be blocked")
        if not isinstance(self.complete, bool):
            raise ValueError("registry dispatch coverage complete must be boolean")
        if not self.claim_boundary:
            raise ValueError("registry dispatch coverage claim_boundary must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready primitive registry-dispatch coverage."""

        return {
            "family": self.family,
            "primitive": self.primitive,
            "identity": self.identity,
            "derivative_rule": self.derivative_rule,
            "has_batching_rule": self.has_batching_rule,
            "has_lowering_rule": self.has_lowering_rule,
            "has_lowering_metadata": self.has_lowering_metadata,
            "has_shape_rule": self.has_shape_rule,
            "has_dtype_rule": self.has_dtype_rule,
            "has_static_argument_rule": self.has_static_argument_rule,
            "nondifferentiable_policy": self.nondifferentiable_policy,
            "effect": self.effect,
            "lowering_metadata_keys": list(self.lowering_metadata_keys),
            "complete": self.complete,
            "blocked_reasons": list(self.blocked_reasons),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ProgramADRegistryDispatchCoverageReport:
    """JSON-ready coverage report for registry-dispatched Program AD primitives."""

    rows: tuple[ProgramADRegistryDispatchCoverageRow, ...]
    family_counts: Mapping[str, int]
    covered_primitives: int
    total_primitives: int
    claim_boundary: str = PROGRAM_AD_REGISTRY_DISPATCH_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("registry dispatch coverage report requires rows")
        if any(not isinstance(row, ProgramADRegistryDispatchCoverageRow) for row in self.rows):
            raise ValueError("registry dispatch coverage rows must be coverage row entries")
        family_counts = dict(self.family_counts)
        if any(not family for family in family_counts):
            raise ValueError("registry dispatch coverage families must be non-empty")
        if any(count <= 0 for count in family_counts.values()):
            raise ValueError("registry dispatch coverage family counts must be positive")
        if self.total_primitives != len(self.rows):
            raise ValueError("registry dispatch coverage total must match row count")
        if self.covered_primitives != sum(1 for row in self.rows if row.complete):
            raise ValueError("registry dispatch coverage covered count must match complete rows")
        if self.covered_primitives < 0 or self.covered_primitives > self.total_primitives:
            raise ValueError("registry dispatch coverage covered count is invalid")
        if sum(family_counts.values()) != self.total_primitives:
            raise ValueError("registry dispatch coverage family counts must sum to total")
        if not self.claim_boundary:
            raise ValueError("registry dispatch coverage claim_boundary must be non-empty")
        object.__setattr__(self, "family_counts", family_counts)

    @property
    def supported(self) -> bool:
        """Return true only when every declared primitive has complete metadata."""

        return self.covered_primitives == self.total_primitives

    @property
    def blocked_identities(self) -> tuple[str, ...]:
        """Return primitive identities that did not resolve to complete contracts."""

        return tuple(row.identity for row in self.rows if not row.complete)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready registry-dispatch coverage evidence."""

        return {
            "supported": self.supported,
            "covered_primitives": self.covered_primitives,
            "total_primitives": self.total_primitives,
            "blocked_identities": list(self.blocked_identities),
            "family_counts": dict(self.family_counts),
            "rows": [row.to_dict() for row in self.rows],
            "claim_boundary": self.claim_boundary,
        }


class CustomDerivativeRegistry:
    """Conflict-safe registry binding primitive identities to exact rules."""

    def __init__(self, rules: dict[PrimitiveIdentity, CustomDerivativeRule] | None = None) -> None:
        self._rules: dict[PrimitiveIdentity, CustomDerivativeRule] = {}
        self._transforms: dict[PrimitiveIdentity, PrimitiveTransformRule] = {}
        if rules is not None:
            for identity, rule in rules.items():
                self.register(identity, rule)

    def register(
        self,
        identity: PrimitiveIdentity | str,
        rule: CustomDerivativeRule,
        *,
        overwrite: bool = False,
    ) -> CustomDerivativeRule:
        """Register an exact derivative rule for a primitive identity."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        if not isinstance(rule, CustomDerivativeRule):
            raise ValueError("registered custom derivative rule must be a CustomDerivativeRule")
        existing = self._rules.get(primitive_identity)
        if existing is not None and existing != rule and not overwrite:
            raise ValueError(
                f"custom derivative rule already registered for {primitive_identity.key}"
            )
        self._rules[primitive_identity] = rule
        existing_transform = self._transforms.get(primitive_identity)
        if existing_transform is None or existing_transform.derivative_rule != rule:
            self._transforms[primitive_identity] = PrimitiveTransformRule(
                identity=primitive_identity,
                derivative_rule=rule,
                batching_rule=None
                if existing_transform is None
                else existing_transform.batching_rule,
                lowering_rule=None
                if existing_transform is None
                else existing_transform.lowering_rule,
                lowering_metadata={}
                if existing_transform is None
                else existing_transform.lowering_metadata,
                shape_rule=None if existing_transform is None else existing_transform.shape_rule,
                dtype_rule=None if existing_transform is None else existing_transform.dtype_rule,
                static_argument_rule=None
                if existing_transform is None
                else existing_transform.static_argument_rule,
                nondifferentiable_policy="not_declared"
                if existing_transform is None
                else existing_transform.nondifferentiable_policy,
                effect="pure" if existing_transform is None else existing_transform.effect,
            )
        return rule

    def decorator(
        self,
        identity: PrimitiveIdentity | str,
        *,
        overwrite: bool = False,
    ) -> Callable[[CustomDerivativeRule], CustomDerivativeRule]:
        """Return a decorator that registers a CustomDerivativeRule object."""

        def register_rule(rule: CustomDerivativeRule) -> CustomDerivativeRule:
            return self.register(identity, rule, overwrite=overwrite)

        return register_rule

    def register_transform(
        self,
        transform: PrimitiveTransformRule,
        *,
        overwrite: bool = False,
    ) -> PrimitiveTransformRule:
        """Register derivative, batching, and lowering metadata for one primitive."""

        if not isinstance(transform, PrimitiveTransformRule):
            raise ValueError("transform must be a PrimitiveTransformRule")
        existing = self._transforms.get(transform.identity)
        if existing is not None and existing != transform and not overwrite:
            raise ValueError(
                f"primitive transform already registered for {transform.identity.key}"
            )
        self.register(transform.identity, transform.derivative_rule, overwrite=overwrite)
        self._transforms[transform.identity] = transform
        return transform

    def register_batching_rule(
        self,
        identity: PrimitiveIdentity | str,
        batching_rule: PrimitiveBatchingRule,
        *,
        overwrite: bool = False,
    ) -> PrimitiveBatchingRule:
        """Attach a primitive-specific batching rule to an existing identity."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        if not callable(batching_rule):
            raise ValueError("batching_rule must be callable")
        rule = self.require(primitive_identity)
        existing = self._transforms.get(primitive_identity)
        if existing is not None and existing.batching_rule is not None and not overwrite:
            raise ValueError(f"batching rule already registered for {primitive_identity.key}")
        metadata = {} if existing is None else existing.lowering_metadata
        self._transforms[primitive_identity] = PrimitiveTransformRule(
            identity=primitive_identity,
            derivative_rule=rule,
            batching_rule=batching_rule,
            lowering_rule=None if existing is None else existing.lowering_rule,
            lowering_metadata=metadata,
            shape_rule=None if existing is None else existing.shape_rule,
            dtype_rule=None if existing is None else existing.dtype_rule,
            static_argument_rule=None if existing is None else existing.static_argument_rule,
            nondifferentiable_policy="not_declared"
            if existing is None
            else existing.nondifferentiable_policy,
            effect="pure" if existing is None else existing.effect,
        )
        return batching_rule

    def register_lowering_rule(
        self,
        identity: PrimitiveIdentity | str,
        lowering_rule: PrimitiveLoweringRule,
        *,
        overwrite: bool = False,
    ) -> PrimitiveLoweringRule:
        """Attach an executable compiler lowering rule to an existing identity."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        if not callable(lowering_rule):
            raise ValueError("lowering_rule must be callable")
        rule = self.require(primitive_identity)
        existing = self._transforms.get(primitive_identity)
        if existing is not None and existing.lowering_rule is not None and not overwrite:
            raise ValueError(f"lowering rule already registered for {primitive_identity.key}")
        self._transforms[primitive_identity] = PrimitiveTransformRule(
            identity=primitive_identity,
            derivative_rule=rule,
            batching_rule=None if existing is None else existing.batching_rule,
            lowering_rule=lowering_rule,
            lowering_metadata={} if existing is None else existing.lowering_metadata,
            shape_rule=None if existing is None else existing.shape_rule,
            dtype_rule=None if existing is None else existing.dtype_rule,
            static_argument_rule=None if existing is None else existing.static_argument_rule,
            nondifferentiable_policy="not_declared"
            if existing is None
            else existing.nondifferentiable_policy,
            effect="pure" if existing is None else existing.effect,
        )
        return lowering_rule

    def batching_rule_for(self, identity: PrimitiveIdentity | str) -> PrimitiveBatchingRule | None:
        """Return the registered primitive batching rule, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else transform.batching_rule

    def lowering_rule_for(self, identity: PrimitiveIdentity | str) -> PrimitiveLoweringRule | None:
        """Return the registered executable compiler lowering rule, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else transform.lowering_rule

    def shape_rule_for(self, identity: PrimitiveIdentity | str) -> PrimitiveShapeRule | None:
        """Return the registered primitive shape rule, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else transform.shape_rule

    def dtype_rule_for(self, identity: PrimitiveIdentity | str) -> PrimitiveDTypeRule | None:
        """Return the registered primitive dtype rule, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else transform.dtype_rule

    def static_argument_rule_for(
        self, identity: PrimitiveIdentity | str
    ) -> PrimitiveStaticArgumentRule | None:
        """Return the registered primitive static-argument rule, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else transform.static_argument_rule

    def nondifferentiable_policy_for(self, identity: PrimitiveIdentity | str) -> str | None:
        """Return the registered primitive nondifferentiability policy, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else transform.nondifferentiable_policy

    def effect_for(self, identity: PrimitiveIdentity | str) -> str | None:
        """Return the registered primitive effect classification, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else transform.effect

    def contract_for(self, identity: PrimitiveIdentity | str) -> PrimitiveContract | None:
        """Return the unified registered primitive contract, if present."""

        transform = self._transforms.get(PrimitiveIdentity.parse(identity))
        return None if transform is None else PrimitiveContract.from_transform(transform)

    def require_batching_rule(self, identity: PrimitiveIdentity | str) -> PrimitiveBatchingRule:
        """Return a primitive batching rule or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        rule = self.batching_rule_for(primitive_identity)
        if rule is None:
            raise ValueError(f"no batching rule registered for {primitive_identity.key}")
        return rule

    def require_lowering_rule(self, identity: PrimitiveIdentity | str) -> PrimitiveLoweringRule:
        """Return an executable compiler lowering rule or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        rule = self.lowering_rule_for(primitive_identity)
        if rule is None:
            raise ValueError(f"no lowering rule registered for {primitive_identity.key}")
        return rule

    def require_shape_rule(self, identity: PrimitiveIdentity | str) -> PrimitiveShapeRule:
        """Return a primitive shape rule or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        rule = self.shape_rule_for(primitive_identity)
        if rule is None:
            raise ValueError(f"no shape rule registered for {primitive_identity.key}")
        return rule

    def require_dtype_rule(self, identity: PrimitiveIdentity | str) -> PrimitiveDTypeRule:
        """Return a primitive dtype rule or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        rule = self.dtype_rule_for(primitive_identity)
        if rule is None:
            raise ValueError(f"no dtype rule registered for {primitive_identity.key}")
        return rule

    def require_static_argument_rule(
        self, identity: PrimitiveIdentity | str
    ) -> PrimitiveStaticArgumentRule:
        """Return a primitive static-argument rule or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        rule = self.static_argument_rule_for(primitive_identity)
        if rule is None:
            raise ValueError(f"no static argument rule registered for {primitive_identity.key}")
        return rule

    def require_nondifferentiable_policy(self, identity: PrimitiveIdentity | str) -> str:
        """Return a primitive nondifferentiability policy or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        policy = self.nondifferentiable_policy_for(primitive_identity)
        if policy is None or policy == "not_declared":
            raise ValueError(
                f"no nondifferentiable policy registered for {primitive_identity.key}"
            )
        return policy

    def require_effect(self, identity: PrimitiveIdentity | str) -> str:
        """Return a primitive effect classification or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        effect = self.effect_for(primitive_identity)
        if effect is None:
            raise ValueError(f"no effect registered for {primitive_identity.key}")
        return effect

    def require_contract(self, identity: PrimitiveIdentity | str) -> PrimitiveContract:
        """Return a unified primitive contract or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        contract = self.contract_for(primitive_identity)
        if contract is None:
            raise ValueError(f"no primitive contract registered for {primitive_identity.key}")
        return contract

    def require_complete_contract(self, identity: PrimitiveIdentity | str) -> PrimitiveContract:
        """Return a compiler/vectorization-ready primitive contract or fail closed."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        contract = self.require_contract(primitive_identity)
        missing: list[str] = []
        if contract.batching_rule is None:
            missing.append("batching_rule")
        if contract.lowering_rule is None:
            missing.append("lowering_rule")
        if not contract.lowering_metadata:
            missing.append("lowering_metadata")
        if not contract.lowering_metadata.get("nondifferentiable_boundary"):
            missing.append("nondifferentiable_boundary")
        if contract.lowering_metadata.get("nondifferentiable_boundary_policy") != "fail_closed":
            missing.append("nondifferentiable_boundary_policy")
        if contract.shape_rule is None:
            missing.append("shape_rule")
        if contract.dtype_rule is None:
            missing.append("dtype_rule")
        if contract.static_argument_rule is None:
            missing.append("static_argument_rule")
        if contract.nondifferentiable_policy == "not_declared":
            missing.append("nondifferentiable_policy")
        if not contract.effect:
            missing.append("effect")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"incomplete primitive contract for {primitive_identity.key}: missing {joined}"
            )
        return contract

    def transform_snapshot(self) -> dict[PrimitiveIdentity, PrimitiveTransformRule]:
        """Return a copy of registered primitive transform bindings."""

        return dict(self._transforms)

    def lookup(self, identity: PrimitiveIdentity | str) -> CustomDerivativeRule | None:
        """Return the registered rule for an identity, if present."""

        return self._rules.get(PrimitiveIdentity.parse(identity))

    def require(self, identity: PrimitiveIdentity | str) -> CustomDerivativeRule:
        """Return the registered rule or fail closed with a useful error."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        rule = self._rules.get(primitive_identity)
        if rule is None:
            raise ValueError(f"no custom derivative rule registered for {primitive_identity.key}")
        return rule

    def unregister(self, identity: PrimitiveIdentity | str) -> CustomDerivativeRule:
        """Remove and return a registered rule."""

        primitive_identity = PrimitiveIdentity.parse(identity)
        try:
            self._transforms.pop(primitive_identity, None)
            return self._rules.pop(primitive_identity)
        except KeyError as exc:
            raise ValueError(
                f"no custom derivative rule registered for {primitive_identity.key}"
            ) from exc

    def snapshot(self) -> dict[PrimitiveIdentity, CustomDerivativeRule]:
        """Return an immutable-by-copy snapshot of registered primitive rules."""

        return dict(self._rules)


DEFAULT_CUSTOM_DERIVATIVE_REGISTRY = CustomDerivativeRegistry()

_PROGRAM_AD_ARRAY_PRIMITIVE_NAMESPACE = "scpn.program_ad.array"
_PROGRAM_AD_ARRAY_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_ARRAY_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_ARRAY_PRIMITIVE_NAMESPACE, name, "1")
    for name in ("getitem", "take", "take_along_axis", "delete", "pad", "insert")
}

_PROGRAM_AD_SHAPE_PRIMITIVE_NAMESPACE = "scpn.program_ad.shape"
_PROGRAM_AD_SHAPE_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_SHAPE_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_SHAPE_PRIMITIVE_NAMESPACE, name, "1")
    for name in (
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
        "expand_dims",
        "flip",
        "fliplr",
        "flipud",
        "moveaxis",
        "reshape",
        "ravel",
        "repeat",
        "roll",
        "rot90",
        "squeeze",
        "swapaxes",
        "tile",
        "transpose",
    )
}

_PROGRAM_AD_REDUCTION_PRIMITIVE_NAMESPACE = "scpn.program_ad.reduction"
_PROGRAM_AD_REDUCTION_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_REDUCTION_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_REDUCTION_PRIMITIVE_NAMESPACE, name, "1")
    for name in (
        "sum",
        "prod",
        "mean",
        "var",
        "std",
        "max",
        "min",
        "median",
        "quantile",
        "percentile",
        "trapezoid",
    )
}

_PROGRAM_AD_STENCIL_PRIMITIVE_NAMESPACE = "scpn.program_ad.stencil"
_PROGRAM_AD_STENCIL_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_STENCIL_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_STENCIL_PRIMITIVE_NAMESPACE, name, "1")
    for name in ("gradient",)
}

_PROGRAM_AD_INTERPOLATION_PRIMITIVE_NAMESPACE = "scpn.program_ad.interpolation"
_PROGRAM_AD_INTERPOLATION_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_INTERPOLATION_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_INTERPOLATION_PRIMITIVE_NAMESPACE, name, "1")
    for name in ("interp",)
}

_PROGRAM_AD_ASSEMBLY_PRIMITIVE_NAMESPACE = "scpn.program_ad.assembly"
_PROGRAM_AD_ASSEMBLY_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_ASSEMBLY_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_ASSEMBLY_PRIMITIVE_NAMESPACE, name, "1")
    for name in (
        "concatenate",
        "stack",
        "append",
        "block",
        "broadcast_to",
        "broadcast_arrays",
        "zeros_like",
        "ones_like",
        "full_like",
        "hstack",
        "vstack",
        "column_stack",
        "dstack",
        "tril",
        "triu",
        "diagonal",
        "split",
        "array_split",
        "hsplit",
        "vsplit",
        "dsplit",
    )
}
_PROGRAM_AD_ASSEMBLY_SPLIT_NAMES = frozenset(
    ("split", "array_split", "hsplit", "vsplit", "dsplit")
)

_PROGRAM_AD_SIGNAL_PRIMITIVE_NAMESPACE = "scpn.program_ad.signal"
_PROGRAM_AD_SIGNAL_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_SIGNAL_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_SIGNAL_PRIMITIVE_NAMESPACE, name, "1")
    for name in ("convolve", "correlate")
}

_PROGRAM_AD_ELEMENTWISE_PRIMITIVE_NAMESPACE = "scpn.program_ad.elementwise"
_PROGRAM_AD_ELEMENTWISE_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_ELEMENTWISE_UNARY_NAMES = (
    "sin",
    "cos",
    "exp",
    "expm1",
    "log",
    "log1p",
    "sqrt",
    "tan",
    "tanh",
    "arcsin",
    "arccos",
    "reciprocal",
    "square",
    "abs",
    "negative",
)
_PROGRAM_AD_ELEMENTWISE_DISCONTINUOUS_NAMES = (
    "sign",
    "heaviside",
)
_PROGRAM_AD_ELEMENTWISE_BINARY_NAMES = (
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "maximum",
    "minimum",
)
_PROGRAM_AD_ELEMENTWISE_NAMES = (
    *_PROGRAM_AD_ELEMENTWISE_UNARY_NAMES,
    *_PROGRAM_AD_ELEMENTWISE_DISCONTINUOUS_NAMES,
    *_PROGRAM_AD_ELEMENTWISE_BINARY_NAMES,
)
_PROGRAM_AD_ELEMENTWISE_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_ELEMENTWISE_PRIMITIVE_NAMESPACE, name, "1")
    for name in _PROGRAM_AD_ELEMENTWISE_NAMES
}

_PROGRAM_AD_SELECTION_PRIMITIVE_NAMESPACE = "scpn.program_ad.selection"
_PROGRAM_AD_SELECTION_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_SELECTION_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_SELECTION_PRIMITIVE_NAMESPACE, name, "1")
    for name in (
        "where",
        "clip",
        "sort",
        "select",
        "piecewise",
        "choose",
        "compress",
        "extract",
        "argmax",
        "argmin",
        "argsort",
    )
}

_PROGRAM_AD_PRODUCT_PRIMITIVE_NAMESPACE = "scpn.program_ad.product"
_PROGRAM_AD_PRODUCT_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_PRODUCT_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_PRODUCT_PRIMITIVE_NAMESPACE, name, "1")
    for name in ("dot", "vdot", "inner", "outer", "matmul", "tensordot", "einsum")
}

_PROGRAM_AD_CUMULATIVE_PRIMITIVE_NAMESPACE = "scpn.program_ad.cumulative"
_PROGRAM_AD_CUMULATIVE_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_CUMULATIVE_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_CUMULATIVE_PRIMITIVE_NAMESPACE, name, "1")
    for name in ("cumsum", "cumprod", "diff")
}

_PROGRAM_AD_LINALG_PRIMITIVE_NAMESPACE = "scpn.program_ad.linalg"
_PROGRAM_AD_LINALG_POLICY = "program_ad_trace_exact_fail_closed"
_PROGRAM_AD_LINALG_IDENTITIES: Mapping[str, PrimitiveIdentity] = {
    name: PrimitiveIdentity(_PROGRAM_AD_LINALG_PRIMITIVE_NAMESPACE, name, "1")
    for name in (
        "det",
        "inv",
        "solve",
        "trace",
        "diag",
        "diagflat",
        "matrix_power",
        "multi_dot",
        "eig",
        "eigh",
        "eigvals",
        "eigvalsh",
        "svd",
        "pinv",
    )
}

_PROGRAM_AD_REGISTRY_DISPATCH_IDENTITY_GROUPS: Mapping[str, Mapping[str, PrimitiveIdentity]] = {
    "array": _PROGRAM_AD_ARRAY_IDENTITIES,
    "shape": _PROGRAM_AD_SHAPE_IDENTITIES,
    "reduction": _PROGRAM_AD_REDUCTION_IDENTITIES,
    "stencil": _PROGRAM_AD_STENCIL_IDENTITIES,
    "interpolation": _PROGRAM_AD_INTERPOLATION_IDENTITIES,
    "assembly": _PROGRAM_AD_ASSEMBLY_IDENTITIES,
    "signal": _PROGRAM_AD_SIGNAL_IDENTITIES,
    "elementwise": _PROGRAM_AD_ELEMENTWISE_IDENTITIES,
    "selection": _PROGRAM_AD_SELECTION_IDENTITIES,
    "product": _PROGRAM_AD_PRODUCT_IDENTITIES,
    "cumulative": _PROGRAM_AD_CUMULATIVE_IDENTITIES,
    "linalg": _PROGRAM_AD_LINALG_IDENTITIES,
}


def register_custom_derivative_rule(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    overwrite: bool = False,
    registry: CustomDerivativeRegistry | None = None,
) -> CustomDerivativeRule:
    """Register a custom derivative rule in the selected or default registry."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.register(identity, rule, overwrite=overwrite)


def register_primitive_transform_rule(
    transform: PrimitiveTransformRule,
    *,
    overwrite: bool = False,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveTransformRule:
    """Register a combined derivative/batching/lowering transform binding."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.register_transform(transform, overwrite=overwrite)


def register_primitive_batching_rule(
    identity: PrimitiveIdentity | str,
    batching_rule: PrimitiveBatchingRule,
    *,
    overwrite: bool = False,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveBatchingRule:
    """Register a batching rule for an existing primitive derivative rule."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.register_batching_rule(identity, batching_rule, overwrite=overwrite)


def register_primitive_lowering_rule(
    identity: PrimitiveIdentity | str,
    lowering_rule: PrimitiveLoweringRule,
    *,
    overwrite: bool = False,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveLoweringRule:
    """Register an executable compiler lowering rule for an existing primitive."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.register_lowering_rule(identity, lowering_rule, overwrite=overwrite)


def primitive_shape_rule_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveShapeRule:
    """Resolve a primitive shape rule or fail closed."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require_shape_rule(identity)


def primitive_dtype_rule_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveDTypeRule:
    """Resolve a primitive dtype rule or fail closed."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require_dtype_rule(identity)


def primitive_static_argument_rule_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveStaticArgumentRule:
    """Resolve a primitive static-argument rule or fail closed."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require_static_argument_rule(identity)


def primitive_nondifferentiable_policy_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> str:
    """Resolve a primitive nondifferentiability policy or fail closed."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require_nondifferentiable_policy(identity)


def primitive_effect_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> str:
    """Resolve a primitive effect classification or fail closed."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require_effect(identity)


def primitive_contract_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveContract:
    """Resolve a unified primitive transform contract or fail closed."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require_contract(identity)


def primitive_complete_contract_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> PrimitiveContract:
    """Resolve a compiler/vectorization-ready primitive contract or fail closed."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require_complete_contract(identity)


def _program_ad_registry_dispatch_coverage_row(
    family: str,
    primitive: str,
    identity: PrimitiveIdentity,
    registry: CustomDerivativeRegistry,
) -> ProgramADRegistryDispatchCoverageRow:
    blocked_reasons: list[str] = []
    contract = registry.contract_for(identity)
    complete = False
    if contract is None:
        blocked_reasons.append("missing primitive registry contract")
    else:
        missing: list[str] = []
        if contract.batching_rule is None:
            missing.append("batching_rule")
        if not contract.lowering_metadata:
            missing.append("lowering_metadata")
        if not contract.lowering_metadata.get("nondifferentiable_boundary"):
            missing.append("nondifferentiable_boundary")
        if contract.lowering_metadata.get("nondifferentiable_boundary_policy") != "fail_closed":
            missing.append("nondifferentiable_boundary_policy")
        if contract.shape_rule is None:
            missing.append("shape_rule")
        if contract.dtype_rule is None:
            missing.append("dtype_rule")
        if contract.static_argument_rule is None:
            missing.append("static_argument_rule")
        if contract.nondifferentiable_policy == "not_declared":
            missing.append("nondifferentiable_policy")
        if not contract.effect:
            missing.append("effect")
        if missing:
            blocked_reasons.append(
                f"incomplete registry-dispatch contract: missing {', '.join(missing)}"
            )
        else:
            complete = True

    return ProgramADRegistryDispatchCoverageRow(
        family=family,
        primitive=primitive,
        identity=identity.key,
        derivative_rule=None if contract is None else contract.derivative_rule.name,
        has_batching_rule=contract is not None and contract.batching_rule is not None,
        has_lowering_rule=contract is not None and contract.lowering_rule is not None,
        has_lowering_metadata=contract is not None and bool(contract.lowering_metadata),
        has_shape_rule=contract is not None and contract.shape_rule is not None,
        has_dtype_rule=contract is not None and contract.dtype_rule is not None,
        has_static_argument_rule=contract is not None
        and contract.static_argument_rule is not None,
        nondifferentiable_policy=None if contract is None else contract.nondifferentiable_policy,
        effect=None if contract is None else contract.effect,
        lowering_metadata_keys=()
        if contract is None
        else tuple(sorted(contract.lowering_metadata)),
        complete=complete,
        blocked_reasons=tuple(blocked_reasons),
    )


def program_ad_registry_dispatch_coverage_report(
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> ProgramADRegistryDispatchCoverageReport:
    """Return registry-dispatched coverage for declared Program AD primitives."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    rows: list[ProgramADRegistryDispatchCoverageRow] = []
    family_counts: dict[str, int] = {}
    for family, identities in _PROGRAM_AD_REGISTRY_DISPATCH_IDENTITY_GROUPS.items():
        family_counts[family] = len(identities)
        for primitive, identity in sorted(identities.items()):
            rows.append(
                _program_ad_registry_dispatch_coverage_row(
                    family,
                    primitive,
                    identity,
                    target,
                )
            )

    return ProgramADRegistryDispatchCoverageReport(
        rows=tuple(rows),
        family_counts=family_counts,
        covered_primitives=sum(1 for row in rows if row.complete),
        total_primitives=len(rows),
    )


def custom_derivative_rule_for(
    identity: PrimitiveIdentity | str,
    *,
    registry: CustomDerivativeRegistry | None = None,
) -> CustomDerivativeRule:
    """Resolve a custom derivative rule for a primitive identity."""

    target = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY if registry is None else registry
    return target.require(identity)


__all__ = [
    "CustomDerivativeRegistry",
    "CustomDerivativeRule",
    "DEFAULT_CUSTOM_DERIVATIVE_REGISTRY",
    "PROGRAM_AD_REGISTRY_DISPATCH_CLAIM_BOUNDARY",
    "PrimitiveBatchingRule",
    "PrimitiveContract",
    "PrimitiveDTypeRule",
    "PrimitiveIdentity",
    "PrimitiveLoweringRule",
    "PrimitiveShapeRule",
    "PrimitiveStaticArgumentRule",
    "PrimitiveTransformRule",
    "ProgramADRegistryDispatchCoverageReport",
    "ProgramADRegistryDispatchCoverageRow",
    "custom_derivative_rule_for",
    "primitive_complete_contract_for",
    "primitive_contract_for",
    "primitive_dtype_rule_for",
    "primitive_effect_for",
    "primitive_nondifferentiable_policy_for",
    "primitive_shape_rule_for",
    "primitive_static_argument_rule_for",
    "program_ad_registry_dispatch_coverage_report",
    "register_custom_derivative_rule",
    "register_primitive_batching_rule",
    "register_primitive_lowering_rule",
    "register_primitive_transform_rule",
]
