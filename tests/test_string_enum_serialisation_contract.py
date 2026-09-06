# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — string-enum serialisation contract
"""Pin what every string-valued enum guarantees to a consumer.

These enums are schema vocabulary: their members travel in JSON payloads,
compare against plain strings, and are read back by consumers that never import
the class. That contract is what this module fixes, so it cannot drift while a
declaration is edited or a base class is changed.

The contract is deliberately stated over the whole family rather than one enum
at a time. A rule that holds for twenty-three of twenty-four is not a contract,
and discovering the exception by accident in a payload is the failure this
exists to prevent.
"""

from __future__ import annotations

import importlib
import json
from enum import Enum
from typing import Final

import pytest

STRING_ENUM_OWNERS: Final = (
    ("scpn_quantum_control.analysis.research_lane_registry", "ResearchLaneMaturity"),
    ("scpn_quantum_control.analysis.research_lane_registry", "ResearchLaneDiffHook"),
    ("scpn_quantum_control.analysis.research_lane_registry", "ResearchLaneClaimStatus"),
    ("scpn_quantum_control.analysis.rl_research_governance", "RLResearchLane"),
    ("scpn_quantum_control.analysis.theory_hook_promotion", "TheoryHookTier"),
    ("scpn_quantum_control.analysis.theory_hook_promotion", "TheoryHookRole"),
    ("scpn_quantum_control.analysis.theory_hook_promotion", "TheoryHookStatus"),
    ("scpn_quantum_control.analysis.witness_discovery", "WitnessSearchMode"),
    ("scpn_quantum_control.applications.honesty_kits", "ApplicationSupportStatus"),
    ("scpn_quantum_control.applications.honesty_kits", "ApplicationDataOrigin"),
    ("scpn_quantum_control.applications.quantum_reservoir_product", "ReservoirTaskKind"),
    ("scpn_quantum_control.chimera_control.schema", "SyntheticRegime"),
    ("scpn_quantum_control.codesign.contracts", "CoDesignMode"),
    ("scpn_quantum_control.codesign.contracts", "SafetyAction"),
    ("scpn_quantum_control.codesign.contracts", "StaleGradientAction"),
    ("scpn_quantum_control.control.closed_loop_analysis", "ResponseClass"),
    ("scpn_quantum_control.control.closed_loop_analysis", "ExecutionMode"),
    ("scpn_quantum_control.forecasting.multimodal_schema", "SyntheticDomainTag"),
    ("scpn_quantum_control.hardware.analog_kuramoto", "AnalogKuramotoPlatform"),
    ("scpn_quantum_control.hardware.analog_kuramoto", "AnalogProviderTarget"),
    ("scpn_quantum_control.hardware.hybrid_digital_analog", "HybridRoute"),
    ("scpn_quantum_control.ml_examples.contracts", "ModelFamily"),
    ("scpn_quantum_control.ml_examples.contracts", "FrameworkStatus"),
    ("scpn_quantum_control.phase.kuramoto_variants", "KuramotoVariant"),
)
"""Every string-valued enum in the published surface, with its owning module."""


def _enum(module_name: str, class_name: str) -> type[Enum]:
    """Return one declared enum class.

    Parameters
    ----------
    module_name
        Module that owns the declaration.
    class_name
        Name of the enum.

    Returns
    -------
    type
        The enum class.

    """
    module = importlib.import_module(module_name)
    resolved = getattr(module, class_name)
    assert isinstance(resolved, type) and issubclass(resolved, Enum)
    return resolved


@pytest.mark.parametrize(("module_name", "class_name"), STRING_ENUM_OWNERS)
class TestStringEnumContract:
    """What a consumer may rely on, for every member of every such enum."""

    def test_members_are_strings(self, module_name: str, class_name: str) -> None:
        """A consumer compares against plain strings without converting first.

        Parameters
        ----------
        module_name, class_name
            Identify the enum under test.

        """
        enum_class = _enum(module_name, class_name)

        assert list(enum_class)
        for member in enum_class:
            assert isinstance(member, str)
            assert member == member.value

    def test_members_serialise_as_their_value(self, module_name: str, class_name: str) -> None:
        """JSON carries the value, never the qualified member name.

        Parameters
        ----------
        module_name, class_name
            Identify the enum under test.

        """
        enum_class = _enum(module_name, class_name)

        for member in enum_class:
            assert json.loads(json.dumps(member)) == member.value

    def test_members_round_trip_through_their_value(
        self, module_name: str, class_name: str
    ) -> None:
        """A serialised value reconstructs the same member.

        Parameters
        ----------
        module_name, class_name
            Identify the enum under test.

        """
        enum_class = _enum(module_name, class_name)

        for member in enum_class:
            assert enum_class(member.value) is member

    def test_values_are_unique_and_non_empty(self, module_name: str, class_name: str) -> None:
        """Two members sharing a value would alias silently on read-back.

        Parameters
        ----------
        module_name, class_name
            Identify the enum under test.

        """
        enum_class = _enum(module_name, class_name)
        values = [member.value for member in enum_class]

        assert all(values)
        assert len(values) == len(set(values))

    def test_text_form_is_the_value(self, module_name: str, class_name: str) -> None:
        """Formatting a member yields its value, not ``Class.MEMBER``.

        This is the property that separates a native string enum from a plain
        ``str`` mixin, and it is the reason the family is declared one way
        rather than the other: an interpolated member must read as the value a
        consumer would see in a payload.

        Parameters
        ----------
        module_name, class_name
            Identify the enum under test.

        """
        enum_class = _enum(module_name, class_name)

        for member in enum_class:
            assert str(member) == member.value
            assert f"{member}" == member.value
            assert format(member) == member.value
