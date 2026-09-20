# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — HAL provider certification gate tests
"""Certification-gate behaviour for the publicly claimed HAL provider matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_quantum_control.hardware.aggregators import (
    AggregatorProviderRoute,
    built_in_aggregator_provider_routes,
)
from scpn_quantum_control.hardware.backends import (
    QuantumBackendDescriptor,
    list_hal_backend_descriptors,
)
from scpn_quantum_control.hardware.hal import (
    BackendCapabilities,
    BackendProfile,
    built_in_backend_profiles,
)
from scpn_quantum_control.hardware.provider_certification import (
    CERTIFICATION_CRITERIA,
    GENERIC_HAL_ADAPTER_TEST,
    PUBLIC_BACKEND_TABLE_DOCUMENT,
    PUBLIC_BACKEND_TABLE_HEADER,
    CertificationCriterion,
    ProviderCertificationRecord,
    ProviderCertificationReport,
    certify_provider_matrix,
    documented_backend_ids,
    focused_adapter_test_path,
    main,
    render_report_table,
    resolve_source_root,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        supports_shots=True,
        supports_counts=True,
        supports_statevector=False,
        supports_mid_circuit_measurement=False,
        supports_analog=False,
        supports_pulse=False,
    )


def _profile(
    backend_id: str = "acme_cloud",
    *,
    provider: str = "acme",
    sdk_package: str = "acme-sdk",
    is_cloud: bool = True,
) -> BackendProfile:
    return BackendProfile(
        backend_id=backend_id,
        provider=provider,
        broker="direct",
        modality="superconducting",
        sdk_package=sdk_package,
        ir_formats=("openqasm3",),
        capabilities=_capabilities(),
        is_cloud=is_cloud,
        submit_requires_approval=is_cloud,
    )


def _descriptor(
    backend_id: str = "acme_cloud",
    *,
    provider: str = "acme",
    sdk_package: str = "acme-sdk",
    adapter_module: str = "scpn_quantum_control.hardware.hal_acme",
    can_submit: bool = True,
    submit_requires_approval: bool = True,
    can_simulate: bool = False,
) -> QuantumBackendDescriptor:
    return QuantumBackendDescriptor(
        name=backend_id,
        provider=provider,
        execution_mode="cloud_qpu",
        sdk_package=sdk_package,
        adapter_module=adapter_module,
        available=False,
        can_simulate=can_simulate,
        can_submit=can_submit,
        submit_requires_approval=submit_requires_approval,
        supports_shots=True,
        supports_statevector=False,
        supports_mid_circuit_measurement=False,
        supports_pulse=False,
        max_qubits=None,
        capabilities=(),
        workloads=(),
    )


def _route(backend_id: str = "acme_cloud") -> AggregatorProviderRoute:
    return AggregatorProviderRoute(
        route_id="direct/acme",
        aggregator="direct",
        provider="acme",
        backend_id=backend_id,
        adapter_module="scpn_quantum_control.hardware.hal_acme",
        sdk_package="acme-sdk",
        ir_formats=("openqasm3",),
        submit_requires_approval=True,
        target_family="acme",
    )


def _checkout(tmp_path: Path, *, backend_ids: tuple[str, ...], tests: tuple[str, ...]) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    rows = "\n".join(
        f"| `{backend_id}` | Acme | direct | superconducting |" for backend_id in backend_ids
    )
    (docs / "backends.md").write_text(
        f"# Backends\n\n{PUBLIC_BACKEND_TABLE_HEADER}\n|---|---|---|---|\n{rows}\n\ntail\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    for relative in tests:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    return tmp_path


def _criterion(record: ProviderCertificationRecord, name: str) -> CertificationCriterion:
    return next(criterion for criterion in record.criteria if criterion.criterion == name)


def test_live_provider_matrix_is_certified_from_repository_evidence() -> None:
    """The declared matrix must pass every public-claim criterion in this checkout."""
    report = certify_provider_matrix(source_root=resolve_source_root(REPOSITORY_ROOT))

    assert report.records
    assert [record.backend_id for record in report.records] == sorted(
        descriptor.name for descriptor in list_hal_backend_descriptors()
    )
    assert report.undeclared_documented_backends == ()
    refused = {
        record.backend_id: record.failures for record in report.records if not record.certified
    }
    assert refused == {}
    assert report.certified is True


def test_every_declared_backend_carries_each_criterion_exactly_once() -> None:
    """Each record must report the full criterion set in canonical order."""
    report = certify_provider_matrix(source_root=resolve_source_root(REPOSITORY_ROOT))

    for record in report.records:
        assert (
            tuple(criterion.criterion for criterion in record.criteria) == CERTIFICATION_CRITERIA
        )


def test_broker_routed_backends_are_inventoried_and_locals_are_marked_not_applicable() -> None:
    """A direct-only HAL target must be named as such, never silently passed."""
    report = certify_provider_matrix(source_root=resolve_source_root(REPOSITORY_ROOT))
    by_id = {record.backend_id: record for record in report.records}
    routed = {route.backend_id for route in built_in_aggregator_provider_routes()}

    ibm = by_id["ibm_quantum"]
    assert "direct/ibm_quantum" in ibm.broker_route_ids
    assert _criterion(ibm, "broker_route_consistency").status == "satisfied"
    assert _criterion(ibm, "capability_catalogue_row").status == "satisfied"

    local = by_id["local_statevector"]
    assert local.backend_id not in routed
    assert local.broker_route_ids == ()
    assert _criterion(local, "broker_route_consistency").status == "not_applicable"
    assert _criterion(local, "optional_dependency_smoke").status == "not_applicable"
    assert "in-repository" in _criterion(local, "optional_dependency_smoke").detail
    assert _criterion(local, "focused_adapter_tests").detail.startswith(GENERIC_HAL_ADAPTER_TEST)


def test_documented_backend_ids_match_the_declared_matrix() -> None:
    """The public table must claim exactly the backends the code declares."""
    claimed = documented_backend_ids(resolve_source_root(REPOSITORY_ROOT))
    declared = {descriptor.name for descriptor in list_hal_backend_descriptors()}

    assert len(set(claimed)) == len(claimed)
    assert set(claimed) == declared


def test_focused_adapter_test_path_is_derived_for_every_declared_adapter() -> None:
    """Every declared adapter must name an existing focused suite."""
    for descriptor in list_hal_backend_descriptors():
        relative = focused_adapter_test_path(descriptor.adapter_module)
        assert (REPOSITORY_ROOT / relative).is_file(), descriptor.name


@pytest.mark.parametrize(
    ("adapter_module", "expected"),
    [
        ("scpn_quantum_control.hardware.hal", GENERIC_HAL_ADAPTER_TEST),
        (
            "scpn_quantum_control.hardware.hal_braket",
            "tests/test_hardware_hal_braket_adapters.py",
        ),
    ],
)
def test_focused_adapter_test_path_maps_known_adapter_shapes(
    adapter_module: str, expected: str
) -> None:
    """Map the generic adapter and a family adapter to their test owners."""
    assert focused_adapter_test_path(adapter_module) == expected


@pytest.mark.parametrize(
    ("adapter_module", "message"),
    [
        ("other.package.hal_acme", "outside the hardware package"),
        ("scpn_quantum_control.hardware.acme_adapter", "hal_<family> contract"),
        ("scpn_quantum_control.hardware.hal_", "hal_<family> contract"),
    ],
)
def test_focused_adapter_test_path_refuses_unmappable_adapters(
    adapter_module: str, message: str
) -> None:
    """An adapter that cannot name its test owner is refused, not exempted."""
    with pytest.raises(ValueError, match=message):
        focused_adapter_test_path(adapter_module)


def test_new_undocumented_backend_is_refused(tmp_path: Path) -> None:
    """A declared backend absent from the public table must not be certified."""
    root = _checkout(
        tmp_path,
        backend_ids=("local_statevector",),
        tests=("tests/test_hardware_hal_acme_adapters.py",),
    )
    report = certify_provider_matrix(
        source_root=resolve_source_root(root),
        descriptors=(_descriptor(),),
        profiles=(_profile(),),
        routes=(),
    )

    record = report.records[0]
    assert report.certified is False
    assert record.certified is False
    assert _criterion(record, "public_documentation").status == "failed"
    assert PUBLIC_BACKEND_TABLE_DOCUMENT in _criterion(record, "public_documentation").detail
    assert report.undeclared_documented_backends == ("local_statevector",)


def test_new_backend_without_focused_adapter_tests_is_refused(tmp_path: Path) -> None:
    """A declared backend without its focused adapter suite must not be certified."""
    root = _checkout(tmp_path, backend_ids=("acme_cloud",), tests=())
    report = certify_provider_matrix(
        source_root=resolve_source_root(root),
        descriptors=(_descriptor(),),
        profiles=(_profile(),),
        routes=(),
    )

    failure = _criterion(report.records[0], "focused_adapter_tests")
    assert failure.status == "failed"
    assert "tests/test_hardware_hal_acme_adapters.py" in failure.detail
    assert report.certified is False


def test_new_backend_without_dependency_evidence_or_adapter_is_refused(tmp_path: Path) -> None:
    """Missing dependency evidence and an unimportable adapter both fail closed."""
    root = _checkout(
        tmp_path,
        backend_ids=("acme_cloud",),
        tests=("tests/test_hardware_hal_acme_adapters.py",),
    )
    report = certify_provider_matrix(
        source_root=resolve_source_root(root),
        descriptors=(_descriptor(),),
        profiles=(_profile(),),
        routes=(),
    )

    record = report.records[0]
    assert _criterion(record, "optional_dependency_smoke").status == "failed"
    assert _criterion(record, "adapter_module_import").status == "failed"
    assert "hal_acme" in _criterion(record, "adapter_module_import").detail
    assert record.failures


def test_declared_route_outside_the_production_selector_is_refused(tmp_path: Path) -> None:
    """A route the selector does not own is refused, not trusted from the table."""
    root = _checkout(
        tmp_path,
        backend_ids=("acme_cloud",),
        tests=("tests/test_hardware_hal_acme_adapters.py",),
    )
    report = certify_provider_matrix(
        source_root=resolve_source_root(root),
        descriptors=(_descriptor(),),
        profiles=(_profile(),),
        routes=(_route(),),
    )

    record = report.records[0]
    assert record.broker_route_ids == ("direct/acme",)
    assert _criterion(record, "broker_route_consistency").status == "failed"
    assert "unresolvable" in _criterion(record, "broker_route_consistency").detail
    assert _criterion(record, "capability_catalogue_row").status == "satisfied"


def test_route_resolving_to_a_different_backend_is_refused(tmp_path: Path) -> None:
    """A declared route must resolve back to the backend that claims it."""
    root = _checkout(
        tmp_path,
        backend_ids=("ionq_cloud",),
        tests=("tests/test_hardware_hal_ionq_adapters.py",),
    )
    profile = next(
        profile for profile in built_in_backend_profiles() if profile.backend_id == "ionq_cloud"
    )
    descriptor = next(
        descriptor
        for descriptor in list_hal_backend_descriptors()
        if descriptor.name == "ionq_cloud"
    )
    borrowed = next(
        route
        for route in built_in_aggregator_provider_routes()
        if route.route_id == "direct/ibm_quantum"
    )
    mismatched = AggregatorProviderRoute(
        route_id=borrowed.route_id,
        aggregator=borrowed.aggregator,
        provider=borrowed.provider,
        backend_id="ionq_cloud",
        adapter_module=descriptor.adapter_module,
        sdk_package=profile.sdk_package,
        ir_formats=borrowed.ir_formats,
        submit_requires_approval=True,
        target_family=borrowed.target_family,
    )

    report = certify_provider_matrix(
        source_root=resolve_source_root(root),
        descriptors=(descriptor,),
        profiles=(profile,),
        routes=(mismatched,),
    )

    failure = _criterion(report.records[0], "broker_route_consistency")
    assert failure.status == "failed"
    assert "resolves to ibm_quantum" in failure.detail


def test_route_that_cannot_be_inventoried_without_submission_is_refused(
    tmp_path: Path,
) -> None:
    """A route whose SDK disagrees with its profile cannot be certified."""
    root = _checkout(
        tmp_path,
        backend_ids=("acme_cloud",),
        tests=("tests/test_hardware_hal_acme_adapters.py",),
    )
    report = certify_provider_matrix(
        source_root=resolve_source_root(root),
        descriptors=(_descriptor(),),
        profiles=(_profile(),),
        routes=(_route(),),
    )
    satisfied = _criterion(report.records[0], "capability_catalogue_row")
    assert satisfied.status == "satisfied"
    assert satisfied.detail.startswith("1 no-submit capability row")

    disagreeing = certify_provider_matrix(
        source_root=resolve_source_root(root),
        descriptors=(_descriptor(),),
        profiles=(_profile(sdk_package="other-sdk"),),
        routes=(_route(),),
    )
    failure = _criterion(disagreeing.records[0], "capability_catalogue_row")
    assert failure.status == "failed"
    assert "cannot be inventoried without submission" in failure.detail
    assert "SDK disagrees" in failure.detail


def test_profile_disagreement_and_absence_are_refused(tmp_path: Path) -> None:
    """A descriptor must agree with an existing authoritative profile."""
    root = _checkout(
        tmp_path,
        backend_ids=("acme_cloud",),
        tests=("tests/test_hardware_hal_acme_adapters.py",),
    )
    missing = certify_provider_matrix(
        source_root=resolve_source_root(root),
        descriptors=(_descriptor(),),
        profiles=(_profile("other_cloud"),),
        routes=(),
    )
    assert _criterion(missing.records[0], "hal_profile_resolution").status == "failed"
    assert (
        "no HAL backend profile" in _criterion(missing.records[0], "hal_profile_resolution").detail
    )
    assert _criterion(missing.records[0], "approval_gated_submission").status == "failed"

    disagreeing = certify_provider_matrix(
        source_root=resolve_source_root(root),
        descriptors=(_descriptor(sdk_package="other-sdk"),),
        profiles=(_profile(),),
        routes=(),
    )
    failure = _criterion(disagreeing.records[0], "hal_profile_resolution")
    assert failure.status == "failed"
    assert "sdk_package" in failure.detail


@pytest.mark.parametrize(
    ("descriptor", "profile", "expected"),
    [
        (
            _descriptor(submit_requires_approval=False),
            _profile(),
            "approval flag False",
        ),
        (
            _descriptor(can_submit=False),
            _profile(),
            "submission capability False",
        ),
        (
            _descriptor(
                can_submit=False,
                submit_requires_approval=False,
                can_simulate=False,
            ),
            _profile(is_cloud=False),
            "exposes no simulation path",
        ),
    ],
)
def test_approval_semantics_must_agree_with_cloud_status(
    tmp_path: Path,
    descriptor: QuantumBackendDescriptor,
    profile: BackendProfile,
    expected: str,
) -> None:
    """Approval, submission, and simulation flags must match the profile class."""
    root = _checkout(
        tmp_path,
        backend_ids=("acme_cloud",),
        tests=("tests/test_hardware_hal_acme_adapters.py",),
    )
    report = certify_provider_matrix(
        source_root=resolve_source_root(root),
        descriptors=(descriptor,),
        profiles=(profile,),
        routes=(),
    )

    failure = _criterion(report.records[0], "approval_gated_submission")
    assert failure.status == "failed"
    assert expected in failure.detail


def test_local_profile_with_simulation_declares_local_policy(tmp_path: Path) -> None:
    """A local profile that can simulate passes with an explicit local policy."""
    root = _checkout(
        tmp_path,
        backend_ids=("acme_cloud",),
        tests=("tests/test_hardware_hal_acme_adapters.py",),
    )
    report = certify_provider_matrix(
        source_root=resolve_source_root(root),
        descriptors=(
            _descriptor(can_submit=False, submit_requires_approval=False, can_simulate=True),
        ),
        profiles=(_profile(is_cloud=False),),
        routes=(),
    )

    criterion = _criterion(report.records[0], "approval_gated_submission")
    assert criterion.status == "satisfied"
    assert "local simulation only" in criterion.detail


def test_duplicate_declared_backend_identifier_is_refused(tmp_path: Path) -> None:
    """A repeated backend identifier is a matrix defect, not a certifiable row."""
    root = _checkout(
        tmp_path,
        backend_ids=("acme_cloud",),
        tests=("tests/test_hardware_hal_acme_adapters.py",),
    )
    with pytest.raises(ValueError, match="repeat a backend identifier"):
        certify_provider_matrix(
            source_root=resolve_source_root(root),
            descriptors=(_descriptor(), _descriptor()),
            profiles=(_profile(),),
            routes=(),
        )


def test_resolve_source_root_requires_complete_evidence(tmp_path: Path) -> None:
    """A checkout without the documentation or test evidence is refused."""
    with pytest.raises(ValueError, match="not a directory"):
        resolve_source_root(tmp_path / "absent")

    with pytest.raises(ValueError, match=PUBLIC_BACKEND_TABLE_DOCUMENT):
        resolve_source_root(tmp_path)

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "backends.md").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks a tests directory"):
        resolve_source_root(tmp_path)


def test_documented_backend_ids_refuse_unreadable_tables(tmp_path: Path) -> None:
    """A missing header or empty table must fail closed, never report a pass."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "tests").mkdir()
    document = docs / "backends.md"

    document.write_text("# Backends\n\nno table here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no claimed backend table header"):
        documented_backend_ids(resolve_source_root(tmp_path))

    document.write_text(
        f"{PUBLIC_BACKEND_TABLE_HEADER}\n|---|---|---|---|\n\nprose\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="has no row"):
        documented_backend_ids(resolve_source_root(tmp_path))

    document.write_text(
        f"{PUBLIC_BACKEND_TABLE_HEADER}\n|---|---|---|---|\n| not backticked | a | b | c |\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="has no row"):
        documented_backend_ids(resolve_source_root(tmp_path))


def test_documented_backend_ids_stop_at_the_end_of_the_table(tmp_path: Path) -> None:
    """Rows after the table must not be read as further public claims."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "tests").mkdir()
    (docs / "backends.md").write_text(
        f"{PUBLIC_BACKEND_TABLE_HEADER}\n"
        "|---|---|---|---|\n"
        "| `first_cloud` | A | direct | superconducting |\n"
        "| free text row | A | direct | superconducting |\n"
        "| `second_cloud` | A | direct | superconducting |\n",
        encoding="utf-8",
    )

    assert documented_backend_ids(resolve_source_root(tmp_path)) == ("first_cloud",)


def test_documented_backend_ids_read_a_table_that_ends_the_document(tmp_path: Path) -> None:
    """A table running to the final line must still be read completely."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "tests").mkdir()
    (docs / "backends.md").write_text(
        f"{PUBLIC_BACKEND_TABLE_HEADER}\n"
        "|---|---|---|---|\n"
        "| `first_cloud` | A | direct | superconducting |\n"
        "| `second_cloud` | A | direct | superconducting |\n",
        encoding="utf-8",
    )

    assert documented_backend_ids(resolve_source_root(tmp_path)) == (
        "first_cloud",
        "second_cloud",
    )


def test_adapter_outside_the_naming_contract_cannot_claim_a_test_owner(
    tmp_path: Path,
) -> None:
    """A descriptor whose adapter cannot name a suite is refused, not exempted."""
    root = _checkout(
        tmp_path,
        backend_ids=("acme_cloud",),
        tests=("tests/test_hardware_hal_acme_adapters.py",),
    )
    report = certify_provider_matrix(
        source_root=resolve_source_root(root),
        descriptors=(_descriptor(adapter_module="scpn_quantum_control.hardware.acme_adapter"),),
        profiles=(_profile(),),
        routes=(),
    )

    failure = _criterion(report.records[0], "focused_adapter_tests")
    assert failure.status == "failed"
    assert "hal_<family> contract" in failure.detail


def test_certification_criterion_refuses_unknown_names_and_silent_details() -> None:
    """Every criterion outcome must be named and explained."""
    with pytest.raises(ValueError, match="unknown certification criterion"):
        CertificationCriterion(criterion="invented", status="satisfied", detail="text")

    with pytest.raises(ValueError, match="must explain the outcome"):
        CertificationCriterion(criterion="public_documentation", status="satisfied", detail="   ")

    criterion = CertificationCriterion(
        criterion="public_documentation", status="satisfied", detail="claimed"
    )
    assert criterion.to_dict() == {
        "criterion": "public_documentation",
        "status": "satisfied",
        "detail": "claimed",
    }


def test_record_requires_the_complete_criterion_set() -> None:
    """A partial criterion set must not be assembled into a record."""
    with pytest.raises(ValueError, match="complete and in order"):
        ProviderCertificationRecord(
            backend_id="acme_cloud",
            provider="acme",
            adapter_module="scpn_quantum_control.hardware.hal_acme",
            broker_route_ids=(),
            criteria=(
                CertificationCriterion(
                    criterion="public_documentation", status="satisfied", detail="claimed"
                ),
            ),
        )


def test_report_serialisation_carries_every_outcome(tmp_path: Path) -> None:
    """The JSON mapping must expose the verdict and each criterion outcome."""
    root = _checkout(
        tmp_path,
        backend_ids=("acme_cloud",),
        tests=("tests/test_hardware_hal_acme_adapters.py",),
    )
    report = certify_provider_matrix(
        source_root=resolve_source_root(root),
        descriptors=(_descriptor(),),
        profiles=(_profile(),),
        routes=(),
    )
    payload = report.to_dict()

    assert payload["certified"] is False
    assert payload["undeclared_documented_backends"] == []
    record_payload = payload["records"][0]  # type: ignore[index]
    assert record_payload["backend_id"] == "acme_cloud"
    assert record_payload["broker_route_ids"] == []
    assert [entry["criterion"] for entry in record_payload["criteria"]] == list(
        CERTIFICATION_CRITERIA
    )


def test_render_report_table_names_failures_and_unimplemented_claims() -> None:
    """The operator table must show the verdict and the reason for a refusal."""
    empty = ProviderCertificationReport(records=(), undeclared_documented_backends=())
    assert render_report_table(empty) == "verdict: certified"

    criteria = tuple(
        CertificationCriterion(
            criterion=name,
            status="failed" if name == "public_documentation" else "satisfied",
            detail=f"{name} evidence",
        )
        for name in CERTIFICATION_CRITERIA
    )
    refused = ProviderCertificationReport(
        records=(
            ProviderCertificationRecord(
                backend_id="acme_cloud",
                provider="acme",
                adapter_module="scpn_quantum_control.hardware.hal_acme",
                broker_route_ids=(),
                criteria=criteria,
            ),
        ),
        undeclared_documented_backends=("ghost_cloud",),
    )
    rendered = render_report_table(refused)

    assert "acme_cloud  REFUSED  public_documentation evidence" in rendered
    assert "ghost_cloud  REFUSED  claimed publicly but never declared" in rendered
    assert rendered.endswith("verdict: refused")


def test_cli_reports_the_live_matrix_as_a_table(capsys: pytest.CaptureFixture[str]) -> None:
    """The default command renders one line per backend and exits clean."""
    exit_code = main(["--source-root", str(REPOSITORY_ROOT)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "verdict: certified" in captured.out
    assert "ibm_quantum" in captured.out


def test_cli_emits_json_for_the_live_matrix(capsys: pytest.CaptureFixture[str]) -> None:
    """The JSON rendering must round-trip the full report payload."""
    exit_code = main(["--source-root", str(REPOSITORY_ROOT), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["certified"] is True
    assert len(payload["records"]) == len(list_hal_backend_descriptors())


def test_cli_fails_closed_on_unreadable_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unusable checkout returns the evidence-error code, never a pass."""
    exit_code = main(["--source-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "provider certification evidence unavailable" in captured.err


def test_cli_refuses_a_checkout_missing_public_claims(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkout whose public table omits declared backends exits refused."""
    root = _checkout(
        tmp_path,
        backend_ids=("local_statevector",),
        tests=("tests/test_hardware_hal.py",),
    )
    monkeypatch.chdir(root)

    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "verdict: refused" in captured.out
