# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Enzyme/MLIR Evidence Docstring Tests
"""Documentation-contract tests for Enzyme/MLIR evidence records."""

from __future__ import annotations

import inspect

import scpn_quantum_control as scpn
import scpn_quantum_control.compiler as compiler
import scpn_quantum_control.compiler.mlir as mlir
from scpn_quantum_control.compiler import mlir_enzyme_evidence as evidence

_PUBLIC_CLASS_MEMBERS: dict[str, tuple[str, ...]] = {
    "EnzymeMLIRCompilerADBreadthArtifactFiles": (),
    "EnzymeMLIRToolchainStatus": ("__post_init__", "to_dict"),
    "EnzymeNativeExecutionEvidence": ("__post_init__", "passed", "to_dict"),
    "MLIRLLVMCorrectnessEvidence": ("__post_init__", "passed", "to_dict"),
    "EnzymeMLIRBenchmarkAttachment": (
        "__post_init__",
        "benchmark_artifact_id",
        "promotion_ready",
        "to_dict",
    ),
    "EnzymeMLIRCompilerADBreadthCaseEvidence": (
        "__post_init__",
        "passed",
        "max_abs_error",
        "to_dict",
    ),
    "EnzymeMLIRCompilerADBreadthArtifact": (
        "__post_init__",
        "transform_modes",
        "frontend_languages",
        "failed_case_ids",
        "passed_case_ids",
        "max_abs_error",
        "runtime_seconds",
        "promotion_ready",
        "to_breadth_evidence",
        "to_dict",
    ),
    "EnzymeMLIRCompilerADBreadthEvidence": ("__post_init__", "passed", "to_dict"),
    "EnzymeMLIRMaturityAuditResult": (
        "__post_init__",
        "ready_for_provider_exceedance",
        "to_dict",
    ),
}

_PUBLIC_FUNCTION_NAMES = (
    "build_enzyme_mlir_benchmark_attachment",
    "build_enzyme_mlir_compiler_ad_breadth_artifact",
    "build_enzyme_mlir_compiler_ad_breadth_gap_artifact",
    "build_enzyme_mlir_compiler_ad_breadth_evidence",
    "render_enzyme_mlir_compiler_ad_breadth_artifact_markdown",
    "write_enzyme_mlir_compiler_ad_breadth_artifact",
)


def _docstring(subject: object) -> str:
    doc = inspect.getdoc(subject)
    assert doc is not None
    text = doc.strip()
    assert text
    return text


def _member_descriptor(cls: type[object], name: str) -> object:
    descriptor = getattr(cls, name)
    if isinstance(descriptor, property):
        assert descriptor.fget is not None
        return descriptor.fget
    return descriptor


#: Imperative openers a *method* contract docstring is expected to use.
_METHOD_CONTRACT_VERBS = ("Return", "Validate")


def _assert_property_states_its_value(member_name: str, doc: str) -> None:
    """Require a property docstring to name the value, not command a caller.

    A property is a value, so its summary is a noun phrase. The repository lint
    enforces this as ruff D421; asserting it here as well keeps the rule visible
    at the surface it protects, and catches the case where someone documents one
    of these evidence members by copying a method's phrasing.

    The summary must also carry more than the member's own name re-spelled, so a
    placeholder such as "Passed." cannot satisfy the contract this test exists
    to enforce.
    """
    summary = doc.splitlines()[0].strip()
    first_word = summary.split(" ", 1)[0].rstrip(".,")
    assert first_word not in _METHOD_CONTRACT_VERBS, (
        f"{member_name}: a property docstring states the value it holds rather "
        f"than commanding a caller; got {summary!r}"
    )
    spelled_out = member_name.replace("_", " ").strip().lower()
    assert summary.rstrip(".").lower() != spelled_out, (
        f"{member_name}: the docstring only re-spells the member name; state "
        f"what the value means, got {summary!r}"
    )
    assert len(summary.split()) >= 4, (
        f"{member_name}: the docstring is too thin to be a contract, got {summary!r}"
    )


def test_enzyme_mlir_evidence_public_exports_have_contract_docstrings() -> None:
    """Compiler-AD evidence exports must document public records and builders."""
    for name in (*_PUBLIC_CLASS_MEMBERS, *_PUBLIC_FUNCTION_NAMES):
        module_object = getattr(evidence, name)
        assert getattr(mlir, name) is module_object
        assert getattr(compiler, name) is module_object
        assert getattr(scpn, name) is module_object
        doc = _docstring(module_object)
        assert any(keyword in doc for keyword in ("Enzyme", "MLIR", "compiler", "benchmark"))

    for class_name, member_names in _PUBLIC_CLASS_MEMBERS.items():
        class_object = getattr(evidence, class_name)
        assert isinstance(class_object, type)
        for member_name in member_names:
            member_doc = _docstring(_member_descriptor(class_object, member_name))
            if isinstance(getattr(class_object, member_name), property):
                _assert_property_states_its_value(member_name, member_doc)
            else:
                assert any(verb in member_doc for verb in _METHOD_CONTRACT_VERBS)
