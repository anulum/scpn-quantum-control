# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Public package export compatibility
"""Compare every recorded export with its actual runtime origin."""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import pickle
import typing
from pathlib import Path
from types import ModuleType
from typing import NotRequired, TypedDict, cast

import pytest
from _pickle_identity import IdentityUnpickler as _IdentityUnpickler

pytestmark = pytest.mark.framework_imports

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "tests/fixtures/public_export_identity_snapshot.json"


class ExportMetadata(TypedDict):
    """Recorded defining names and runtime metadata of an inline definition."""

    module: str | None
    qualname: str | None
    annotations: NotRequired[dict[str, str]]
    signature: NotRequired[str]


class ExportBinding(TypedDict):
    """An original public binding and its independently imported origin."""

    name: str
    origin_module: str
    origin_attribute: str | None
    origin_kind: str
    type_module: str
    type_name: str
    local_definition_ast_sha256: str | None
    metadata: ExportMetadata


class PackageSnapshot(TypedDict):
    """Ordered public names and bindings of one actual package initializer."""

    path: str
    module: str
    source_sha256: str
    has_declared_all: bool
    declared_all: list[str]
    all_container_type: str
    star_names: list[str]
    bindings: list[ExportBinding]


class ExportSnapshot(TypedDict):
    """Versioned observations made before package import deferral."""

    schema_version: int
    source_revision: str
    packages: list[PackageSnapshot]


SNAPSHOT = cast(ExportSnapshot, json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8")))
PACKAGES = SNAPSHOT["packages"]


NUMERICAL_METHOD_ADMISSION = "scpn_quantum_control.benchmarks.numerical_method_admission"
APPROVED_EXPORT_ADDITIONS: dict[str, tuple[int, tuple[tuple[str, str], ...]]] = {
    "scpn_quantum_control": (
        3,
        (
            ("ScientificDesign", "scpn_quantum_control.scientific_design"),
            ("ScientificUnits", "scpn_quantum_control.scientific_design"),
            ("DesignObjective", "scpn_quantum_control.scientific_design"),
            ("ScientificProblemParameters", "scpn_quantum_control.scientific_problem_parameters"),
            ("build_scientific_phase_system", "scpn_quantum_control.kuramoto_core"),
            ("KuramotoModelConvention", "scpn_quantum_control.kuramoto_model_conventions"),
            ("kuramoto_convention_matrix", "scpn_quantum_control.kuramoto_model_conventions"),
            ("kuramoto_model_convention", "scpn_quantum_control.kuramoto_model_conventions"),
            ("validate_scientific_design", "scpn_quantum_control.kuramoto_core"),
            ("EigenpairDerivative", "scpn_quantum_control.wirtinger_calculus"),
            ("SpectralDegeneracyError", "scpn_quantum_control.wirtinger_calculus"),
            ("SpectralProjectorDerivative", "scpn_quantum_control.wirtinger_calculus"),
            ("framework_complex_gradient", "scpn_quantum_control.wirtinger_calculus"),
            ("hermitian_eigenpair_derivative", "scpn_quantum_control.wirtinger_calculus"),
            ("real_block_gradient", "scpn_quantum_control.wirtinger_calculus"),
            ("real_block_jacobian", "scpn_quantum_control.wirtinger_calculus"),
            ("spectral_projector_derivative", "scpn_quantum_control.wirtinger_calculus"),
            ("WHOLE_PROGRAM_REPLAY_RANDOMNESS", "scpn_quantum_control.whole_program_ad_api"),
            ("WHOLE_PROGRAM_REPLAY_SCHEMA", "scpn_quantum_control.whole_program_ad_api"),
            ("WholeProgramReplayMismatchError", "scpn_quantum_control.whole_program_ad_api"),
            ("WholeProgramReplayRecord", "scpn_quantum_control.whole_program_ad_api"),
            ("WholeProgramReplayReport", "scpn_quantum_control.whole_program_ad_api"),
            ("capture_whole_program_replay", "scpn_quantum_control.whole_program_ad_api"),
            ("replay_whole_program", "scpn_quantum_control.whole_program_ad_api"),
            ("GENERATOR_EIGENVALUES", "scpn_quantum_control.differentiable_parameter_shift"),
            ("NON_SMOOTH_OPERATIONS", "scpn_quantum_control.differentiable_parameter_shift"),
            (
                "QUANTUM_GRADIENT_ESTIMATOR_PLAN_SCHEMA",
                "scpn_quantum_control.differentiable_parameter_shift",
            ),
            ("GateOccurrence", "scpn_quantum_control.differentiable_parameter_shift"),
            (
                "QuantumGradientEstimatorPlan",
                "scpn_quantum_control.differentiable_parameter_shift",
            ),
            (
                "QuantumGradientEstimatorRefusal",
                "scpn_quantum_control.differentiable_parameter_shift",
            ),
            (
                "execute_quantum_gradient_estimator",
                "scpn_quantum_control.differentiable_parameter_shift",
            ),
            (
                "plan_quantum_gradient_estimator",
                "scpn_quantum_control.differentiable_parameter_shift",
            ),
        ),
    ),
    "scpn_quantum_control.benchmarks": (
        85,
        (
            ("METHOD_SELECTION_OBJECTIVE", NUMERICAL_METHOD_ADMISSION),
            ("NUMERICAL_METHOD_ADMISSION_CLAIM_BOUNDARY", NUMERICAL_METHOD_ADMISSION),
            ("NUMERICAL_METHOD_ADMISSION_SCHEMA", NUMERICAL_METHOD_ADMISSION),
            ("CandidateVerdict", NUMERICAL_METHOD_ADMISSION),
            ("EvidenceReference", NUMERICAL_METHOD_ADMISSION),
            ("MeasuredMethodCandidate", NUMERICAL_METHOD_ADMISSION),
            ("NumericalMethodRequest", NUMERICAL_METHOD_ADMISSION),
            ("NumericalMethodSelection", NUMERICAL_METHOD_ADMISSION),
            ("NumericalMethodSelectionCache", NUMERICAL_METHOD_ADMISSION),
            ("candidates_from_reproducible_comparison", NUMERICAL_METHOD_ADMISSION),
            ("candidates_from_tn_baseline_flagship", NUMERICAL_METHOD_ADMISSION),
            ("plan_digest", NUMERICAL_METHOD_ADMISSION),
            ("render_numerical_method_selection_markdown", NUMERICAL_METHOD_ADMISSION),
            ("select_numerical_method", NUMERICAL_METHOD_ADMISSION),
        ),
    ),
    "scpn_quantum_control.codesign": (
        0,
        (
            ("AlgorithmDesignPlan", "scpn_quantum_control.codesign.algorithm_design"),
            ("AlgorithmDesignSelection", "scpn_quantum_control.codesign.algorithm_design"),
            ("DesignEvaluation", "scpn_quantum_control.codesign.algorithm_design"),
            ("AlgorithmDesignProblem", "scpn_quantum_control.codesign.algorithm_design_contracts"),
            ("DesignBackendSnapshot", "scpn_quantum_control.codesign.algorithm_design_contracts"),
            ("DesignCandidate", "scpn_quantum_control.codesign.algorithm_design_contracts"),
            ("DesignObjectives", "scpn_quantum_control.codesign.algorithm_design_contracts"),
        ),
    ),
    "scpn_quantum_control.compiler": (
        0,
        (
            ("SourceSpan", "scpn_quantum_control.compiler.circuit_pass_records"),
            ("CircuitDiagnostic", "scpn_quantum_control.compiler.circuit_pass_records"),
            ("CircuitPassRefused", "scpn_quantum_control.compiler.circuit_pass_records"),
            ("CircuitOperation", "scpn_quantum_control.compiler.circuit_pass_records"),
            ("CircuitIR", "scpn_quantum_control.compiler.circuit_pass_records"),
            ("CircuitPassRecord", "scpn_quantum_control.compiler.circuit_pass_records"),
            ("import_circuit_source", "scpn_quantum_control.compiler.circuit_source"),
            ("snapshot_circuit", "scpn_quantum_control.compiler.circuit_source"),
            (
                "QualifiedCircuitCompilation",
                "scpn_quantum_control.compiler.circuit_pass_qualification",
            ),
            ("qualify_circuit_pass", "scpn_quantum_control.compiler.circuit_pass_qualification"),
            (
                "compile_circuit_to_mlir",
                "scpn_quantum_control.compiler.circuit_pass_qualification",
            ),
        ),
    ),
    "scpn_quantum_control.mitigation": (
        0,
        (
            ("ReadoutMitigationEstimate", "scpn_quantum_control.mitigation.readout_uncertainty"),
            ("ReadoutZNEResult", "scpn_quantum_control.mitigation.readout_uncertainty"),
            (
                "mitigate_counts_with_uncertainty",
                "scpn_quantum_control.mitigation.readout_uncertainty",
            ),
            (
                "zne_extrapolate_with_readout_uncertainty",
                "scpn_quantum_control.mitigation.readout_uncertainty",
            ),
        ),
    ),
    "scpn_quantum_control.accel": (
        385,
        (
            ("sparse_coupling_from_edges", "oscillatools.accel.sparse_kuramoto"),
            ("SparseStorageReport", "oscillatools.accel.sparse_kuramoto"),
            ("SparseExecutionReport", "oscillatools.accel.sparse_kuramoto"),
            ("sparse_kuramoto_execution_report", "oscillatools.accel.sparse_kuramoto"),
        ),
    ),
}


def _expected_declared_all(package: PackageSnapshot) -> list[str]:
    """Keep the frozen original order plus exact approved public additions.

    Parameters
    ----------
    package
        Original observation retained unchanged in the compatibility fixture.

    Returns
    -------
    list[str]
        Exact current declaration, including every original occurrence.

    """
    index, additions = APPROVED_EXPORT_ADDITIONS.get(package["module"], (0, ()))
    original = package["declared_all"]
    return original[:index] + [name for name, _ in additions] + original[index:]


@pytest.mark.parametrize(
    "public_module,name,origin_module",
    [
        (module, name, origin)
        for module, (_, additions) in APPROVED_EXPORT_ADDITIONS.items()
        for name, origin in additions
    ],
)
def test_approved_additions_resolve_to_their_actual_owning_objects(
    public_module: str, name: str, origin_module: str
) -> None:
    """Resolve each approved addition through its genuine public API and owning object.

    Parameters
    ----------
    public_module
        Public namespace containing the explicitly approved addition.
    name
        Exact exported name from the completed public interface work.
    origin_module
        Original defining module of that object.

    """
    module = importlib.import_module(public_module)
    origin = importlib.import_module(origin_module)
    assert getattr(module, name) is getattr(origin, name)
    assert list(module.__all__).count(name) == 1


def test_pickle_identity_refuses_unregistered_globals() -> None:
    """Reject a genuine pickle of an unregistered builtin before resolving it."""
    with pytest.raises(pickle.UnpicklingError, match="unregistered pickle global"):
        _IdentityUnpickler(pickle.dumps(eval), {}).load()


def test_pickle_identity_refuses_a_changed_public_binding() -> None:
    """Reject a stream whose genuine import no longer matches the expected object."""
    with pytest.raises(pickle.UnpicklingError, match="changed pickle binding"):
        _IdentityUnpickler(pickle.dumps(len), {("builtins", "len"): sum}).load()


def test_snapshot_covers_the_actual_initializer_inventory() -> None:
    """Reject omitted packages, repeated bindings and unversioned expectations."""
    assert SNAPSHOT["schema_version"] == 1
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src/scpn_quantum_control").rglob("__init__.py")
    }
    assert {package["path"] for package in PACKAGES} == actual
    assert len(PACKAGES) == len(actual)
    for package in PACKAGES:
        names = [binding["name"] for binding in package["bindings"]]
        assert len(names) == len(set(names)), package["module"]
        assert package["has_declared_all"], package["module"]
        assert set(names) == set(package["declared_all"]), package["module"]


@pytest.mark.parametrize("package", PACKAGES, ids=[p["module"] for p in PACKAGES])
def test_every_public_binding_is_the_original_object(package: PackageSnapshot) -> None:
    """Exercise all exports against genuine modules without skipping optional names.

    Parameters
    ----------
    package
        Original package observation with actual module and attribute origins.

    """
    module = importlib.import_module(package["module"])
    for binding in package["bindings"]:
        actual = getattr(module, binding["name"])
        origin = importlib.import_module(binding["origin_module"])
        attribute = binding["origin_attribute"]
        expected = origin if attribute is None else getattr(origin, attribute)
        context = f"{package['module']}.{binding['name']}"
        assert actual is expected, context
        assert getattr(module, binding["name"]) is actual, context
        if isinstance(actual, ModuleType):
            assert actual is importlib.import_module(actual.__name__), context
            assert actual.__spec__ is not None, context
            assert actual.__spec__.name == actual.__name__, context
        else:
            metadata = binding["metadata"]
            assert getattr(actual, "__module__", None) == metadata["module"], context
            assert getattr(actual, "__qualname__", None) == metadata["qualname"], context
            assert type(actual).__module__ == binding["type_module"], context
            assert type(actual).__qualname__ == binding["type_name"], context


@pytest.mark.parametrize("package", PACKAGES, ids=[p["module"] for p in PACKAGES])
def test_ordered_all_star_import_and_inspection_remain_compatible(
    package: PackageSnapshot,
) -> None:
    """Preserve explicit order, duplicates, star names and missing-name behavior.

    Parameters
    ----------
    package
        Original initializer's public export order and star-import names.

    """
    module = importlib.import_module(package["module"])
    assert package["has_declared_all"]
    declared = module.__all__
    assert list(declared) == _expected_declared_all(package)
    assert type(declared).__name__ == package["all_container_type"]
    assert set(declared) <= set(dir(module))
    namespace: dict[str, object] = {}
    exec(f"from {package['module']} import *", namespace)
    _, additions = APPROVED_EXPORT_ADDITIONS.get(package["module"], (0, ()))
    star_names = sorted(package["star_names"] + [name for name, _ in additions])
    assert sorted(name for name in namespace if name != "__builtins__") == star_names
    for name in star_names:
        assert namespace[name] is getattr(module, name)
    with pytest.raises(AttributeError, match="undeclared_export_for_compatibility_test"):
        _ = module.undeclared_export_for_compatibility_test
    assert "undeclared_export_for_compatibility_test" not in vars(module)


@pytest.mark.parametrize(
    "package,binding",
    [
        (package, binding)
        for package in PACKAGES
        for binding in package["bindings"]
        if binding["origin_kind"] == "local-definition"
    ],
    ids=[
        f"{package['module']}.{binding['name']}"
        for package in PACKAGES
        for binding in package["bindings"]
        if binding["origin_kind"] == "local-definition"
    ],
)
def test_inline_definitions_keep_metadata_annotations_and_pickle_identity(
    package: PackageSnapshot, binding: ExportBinding
) -> None:
    """Resolve real inline globals and preserve the defining statement and import path.

    Parameters
    ----------
    package
        Package that owns the unchanged public definition.
    binding
        Original definition's AST, annotations, signature and pickle path.

    """
    module = importlib.import_module(package["module"])
    actual = getattr(module, binding["name"])
    tree = ast.parse((ROOT / package["path"]).read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name == binding["name"]
    )
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    assert digest == binding["local_definition_ast_sha256"]
    metadata = binding["metadata"]
    assert {name: repr(value) for name, value in typing.get_type_hints(actual).items()} == (
        metadata["annotations"]
    )
    if "signature" in metadata:
        assert str(inspect.signature(actual)) == metadata["signature"]
    assert isinstance(metadata["module"], str)
    assert isinstance(metadata["qualname"], str)
    allowed = {(metadata["module"], metadata["qualname"]): actual}
    assert _IdentityUnpickler(pickle.dumps(actual), allowed).load() is actual


def test_real_harness_instance_keeps_its_public_pickle_path() -> None:
    """Round-trip the real validated dataset and classical-reference result."""
    from scpn_quantum_control import dla_parity
    from scpn_quantum_control.benchmark_harness import run_phase1_benchmark
    from scpn_quantum_control.dla_parity import FullHarnessResult

    original = run_phase1_benchmark(baselines_backend="numpy")
    names = (
        "FullHarnessResult",
        "DlaParityDataset",
        "DlaParityRun",
        "DlaParityCircuit",
        "DlaParityCircuitMeta",
        "StatisticalSummary",
        "ReproductionResult",
        "ReproductionTolerance",
        "FisherResult",
        "ClassicalLeakageReference",
        "ClassicalLeakagePoint",
    )
    classes = [cast(type[object], getattr(dla_parity, name)) for name in names]
    allowed = {(cls.__module__, cls.__qualname__): cls for cls in classes}
    restored = _IdentityUnpickler(pickle.dumps(original), allowed).load()
    assert type(restored) is FullHarnessResult
    assert restored.dataset.n_circuits_total == original.dataset.n_circuits_total
    assert restored.reproduction.n_circuits_used == original.reproduction.n_circuits_used
    assert restored.classical_reference.backend == "numpy"
    assert restored.classical_reference.is_zero_within_tolerance
