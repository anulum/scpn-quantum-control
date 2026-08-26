# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Domain application honesty-kit tests
"""Tests for typed application-honesty honesty policy and deterministic evidence."""

from __future__ import annotations

from dataclasses import replace

import pytest

import scpn_quantum_control.applications.honesty_kits as honesty
from scpn_quantum_control.applications import (
    ApplicationDataOrigin,
    ApplicationHonestyAuditReport,
    ApplicationSupportStatus,
    DomainApplicationHonestyKit,
    build_application_honesty_audit_report,
    get_domain_application_honesty_kit,
    get_domain_application_honesty_kit_for_dataset,
    list_domain_application_honesty_kits,
    render_application_honesty_audit_markdown,
)
from scpn_quantum_control.applications.dataset_catalog import (
    ApplicationBenchmarkPrivacyAudit,
)


def _kit(**overrides: object) -> DomainApplicationHonestyKit:
    values: dict[str, object] = {
        "kit_id": "test_kit",
        "domain_tag": "test_like_sim",
        "title": "Test simulation kit",
        "support_status": ApplicationSupportStatus.SIMULATION_ONLY,
        "data_origin": ApplicationDataOrigin.SYNTHETIC,
        "synthetic_only": True,
        "dataset_ids": (),
        "source_modules": ("example.module",),
        "allowed_uses": ("test software contracts",),
        "caveats": ("not domain evidence",),
        "claims_forbidden": ("operational use",),
        "forecasting_tags": ("synthetic",),
    }
    values.update(overrides)
    return DomainApplicationHonestyKit(**values)  # type: ignore[arg-type]


def _privacy_row(
    *, dataset_id: str = "dataset", passed: bool = True
) -> ApplicationBenchmarkPrivacyAudit:
    return ApplicationBenchmarkPrivacyAudit(
        dataset_id=dataset_id,
        source_mode="curated",
        privacy_classification="public",
        contains_personal_data=False,
        privacy_boundary="public fixture",
        artifact_hashes={"K_nm_sha256": "a" * 64},
        passed=passed,
    )


def test_builtin_kits_cover_required_domains_and_claim_boundaries() -> None:
    """Every BL-63 family has a complete immutable honesty record."""
    kits = list_domain_application_honesty_kits()
    assert [kit.kit_id for kit in kits] == [
        "power_grid_public_benchmark",
        "josephson_illustrative_simulation",
        "eeg_like_synthetic",
        "iter_disruption_inspired_simulation",
    ]
    assert all(not kit.publication_safe for kit in kits)
    assert all(kit.caveats and kit.claims_forbidden for kit in kits)
    assert get_domain_application_honesty_kit("eeg_like_synthetic") is kits[2]
    payload = kits[2].as_dict()
    assert payload["synthetic_only"] is True
    assert payload["forecasting_tags"] == ["eeg_like_sim"]


def test_power_grid_dataset_resolves_to_unique_kit() -> None:
    """The governed packaged grid case has exactly one honesty kit."""
    kit = get_domain_application_honesty_kit_for_dataset("ieee5bus_power_grid")
    assert kit.data_origin is ApplicationDataOrigin.CURATED_PUBLIC
    assert kit.synthetic_only is False


def test_unknown_kit_and_dataset_fail_closed() -> None:
    """Registry lookups never invent fallback policy."""
    with pytest.raises(KeyError, match="known"):
        get_domain_application_honesty_kit("missing")
    with pytest.raises(KeyError, match="no application honesty kit"):
        get_domain_application_honesty_kit_for_dataset("missing")


def test_duplicate_dataset_governance_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dataset cannot silently inherit two incompatible kits."""
    grid = get_domain_application_honesty_kit("power_grid_public_benchmark")
    monkeypatch.setattr(honesty, "_BUILTIN_KITS", (grid, replace(grid, kit_id="duplicate")))
    with pytest.raises(RuntimeError, match="multiple"):
        get_domain_application_honesty_kit_for_dataset("ieee5bus_power_grid")


@pytest.mark.parametrize("field", ["kit_id", "domain_tag", "title"])
def test_kit_rejects_empty_identity_fields(field: str) -> None:
    """Machine and human identifiers must remain explicit."""
    with pytest.raises(ValueError, match=field):
        _kit(**{field: " "})


def test_kit_rejects_invalid_enum_types() -> None:
    """Raw strings cannot bypass support or provenance enums."""
    with pytest.raises(TypeError, match="support_status"):
        _kit(support_status="simulation_only")
    with pytest.raises(TypeError, match="data_origin"):
        _kit(data_origin="synthetic")


@pytest.mark.parametrize(
    ("origin", "synthetic_only"),
    [
        (ApplicationDataOrigin.SYNTHETIC, False),
        (ApplicationDataOrigin.CURATED_PUBLIC, True),
    ],
)
def test_kit_rejects_inconsistent_synthetic_policy(
    origin: ApplicationDataOrigin,
    synthetic_only: bool,
) -> None:
    """The boolean shorthand cannot disagree with the provenance enum."""
    with pytest.raises(ValueError, match="must match"):
        _kit(data_origin=origin, synthetic_only=synthetic_only)


def test_synthetic_kit_rejects_packaged_dataset_ids() -> None:
    """Synthetic-only routes cannot smuggle in a packaged domain artifact."""
    with pytest.raises(ValueError, match="cannot name packaged"):
        _kit(dataset_ids=("unexpected",))


@pytest.mark.parametrize(
    "field",
    ["source_modules", "allowed_uses", "caveats", "claims_forbidden"],
)
def test_kit_requires_non_empty_policy_sequences(field: str) -> None:
    """Every explanatory policy sequence is mandatory."""
    with pytest.raises(ValueError, match=field):
        _kit(**{field: ()})


@pytest.mark.parametrize(
    "field",
    ["source_modules", "allowed_uses", "caveats", "claims_forbidden"],
)
def test_kit_rejects_blank_and_duplicate_policy_values(field: str) -> None:
    """Policy arrays contain unique, non-blank language."""
    with pytest.raises(ValueError, match=field):
        _kit(**{field: (" ",)})
    with pytest.raises(ValueError, match="unique"):
        _kit(**{field: ("same", "same")})


def test_kit_rejects_bad_dataset_and_forecasting_tags() -> None:
    """Dataset identifiers and BL-37 enum tags are type- and uniqueness-checked."""
    with pytest.raises(ValueError, match="dataset_ids"):
        _kit(
            data_origin=ApplicationDataOrigin.CURATED_PUBLIC,
            synthetic_only=False,
            dataset_ids=(" ",),
        )
    with pytest.raises(ValueError, match="unique"):
        _kit(
            data_origin=ApplicationDataOrigin.CURATED_PUBLIC,
            synthetic_only=False,
            dataset_ids=("same", "same"),
        )
    with pytest.raises(ValueError, match="BL-37"):
        _kit(forecasting_tags=("unregistered_sim",))
    with pytest.raises(ValueError, match="unique"):
        _kit(forecasting_tags=("synthetic", "synthetic"))


def test_report_is_deterministic_and_renders_all_boundaries() -> None:
    """Frozen JSON and Markdown evidence agree on digest, scope, and PASS state."""
    first = build_application_honesty_audit_report()
    second = build_application_honesty_audit_report()
    assert first.passed is True
    assert first.as_dict() == second.as_dict()
    assert first.as_dict()["content_digest"] == first.content_digest()
    markdown = render_application_honesty_audit_markdown(first)
    assert "Result: `PASS`" in markdown
    assert "`none`" in markdown
    assert "`grid_like_sim`" in markdown
    assert first.content_digest() in markdown
    assert "not domain, clinical, facility, hardware, or advantage evidence" in markdown


def test_report_rejects_empty_duplicate_and_unaudited_inputs() -> None:
    """Aggregate evidence is complete and uniquely keyed."""
    kit = _kit()
    row = _privacy_row()
    with pytest.raises(ValueError, match="at least one kit"):
        ApplicationHonestyAuditReport(kits=(), dataset_privacy=(row,))
    with pytest.raises(ValueError, match="dataset privacy"):
        ApplicationHonestyAuditReport(kits=(kit,), dataset_privacy=())
    with pytest.raises(ValueError, match="unique"):
        ApplicationHonestyAuditReport(kits=(kit, kit), dataset_privacy=(row,))
    governed = _kit(
        data_origin=ApplicationDataOrigin.CURATED_PUBLIC,
        synthetic_only=False,
        dataset_ids=("governed",),
    )
    with pytest.raises(ValueError, match="missing privacy evidence"):
        ApplicationHonestyAuditReport(kits=(governed,), dataset_privacy=(row,))


def test_report_can_render_explicit_failed_external_row() -> None:
    """A non-passing aggregate is rendered as FAIL without promotion language."""
    report = ApplicationHonestyAuditReport(
        kits=(_kit(forecasting_tags=()),),
        dataset_privacy=(_privacy_row(passed=False),),
    )
    assert report.passed is False
    assert "Result: `FAIL`" in render_application_honesty_audit_markdown(report)
