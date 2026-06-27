# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable competitive baseline refresh gate.
"""Freshness gate for differentiable-computing competitive baselines."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, cast

from .differentiable_claim_ledger import DEFAULT_LEDGER_PATH, REPO_ROOT, ClaimLedger
from .differentiable_sota_scorecard import (
    DEFAULT_PUBLIC_SOTA_LANGUAGE_PATHS,
    REQUIRED_SOTA_CATEGORIES,
    DifferentiableSOTACategory,
    DifferentiableSOTAPromotionLanguageAudit,
    audit_differentiable_sota_promotion_language,
)

CompetitiveBaselineId = Literal[
    "jax",
    "pytorch",
    "tensorflow",
    "pennylane",
    "qiskit_algorithms",
    "catalyst",
    "enzyme_mlir",
    "julia_ad",
    "emerging_ad",
]
CompetitiveBaselineSourceKind = Literal[
    "official_docs",
    "official_project_docs",
    "official_repository",
]

DIFFERENTIABLE_COMPETITIVE_BASELINE_SCHEMA = (
    "scpn_qc_differentiable_competitive_baseline_refresh_v1"
)
DIFFERENTIABLE_COMPETITIVE_BASELINE_ARTIFACT_ID = "diff-competitive-baseline-refresh-20260627"
DEFAULT_COMPETITIVE_BASELINE_REFRESH_PATH = (
    REPO_ROOT
    / "data"
    / "differentiable_phase_qnode"
    / "differentiable_competitive_baseline_refresh_20260627.json"
)
MAX_BASELINE_AGE_DAYS = 45
REQUIRED_BASELINE_IDS: tuple[CompetitiveBaselineId, ...] = (
    "jax",
    "pytorch",
    "tensorflow",
    "pennylane",
    "qiskit_algorithms",
    "catalyst",
    "enzyme_mlir",
    "julia_ad",
    "emerging_ad",
)
SOURCE_KINDS: frozenset[CompetitiveBaselineSourceKind] = frozenset(
    {"official_docs", "official_project_docs", "official_repository"}
)


@dataclass(frozen=True)
class CompetitiveBaselineRow:
    """One upstream differentiable-computing baseline source."""

    baseline_id: CompetitiveBaselineId
    display_name: str
    upstream_version: str
    source_url: str
    source_kind: CompetitiveBaselineSourceKind
    checked_on: date
    refresh_due_on: date
    max_age_days: int
    scorecard_categories: tuple[DifferentiableSOTACategory, ...]
    required_capabilities: tuple[str, ...]
    hardening_implications: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate local row invariants that do not need repository access."""
        if self.baseline_id not in REQUIRED_BASELINE_IDS:
            raise ValueError(f"unknown competitive baseline: {self.baseline_id}")
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError(f"unknown baseline source kind: {self.source_kind}")
        if not self.source_url.startswith("https://"):
            raise ValueError("competitive baseline source_url must be an HTTPS URL")
        if self.max_age_days <= 0:
            raise ValueError("competitive baseline max_age_days must be positive")
        if self.refresh_due_on != self.checked_on + timedelta(days=self.max_age_days):
            raise ValueError("competitive baseline refresh_due_on must match max_age_days")
        for field_name in (
            "display_name",
            "upstream_version",
            "source_url",
            "claim_boundary",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in (
            "scorecard_categories",
            "required_capabilities",
            "hardening_implications",
        ):
            value = getattr(self, field_name)
            if not value or any(not str(item).strip() for item in value):
                raise ValueError(f"{field_name} must contain non-empty entries")
        unknown_categories = tuple(
            category
            for category in self.scorecard_categories
            if category not in REQUIRED_SOTA_CATEGORIES
        )
        if unknown_categories:
            raise ValueError(
                "competitive baseline row references unknown SOTA categories: "
                + ", ".join(unknown_categories)
            )

    @property
    def classification(self) -> str:
        """Return the evidence classification for this baseline source."""
        return "baseline_refresh_evidence"

    def age_days(self, *, as_of: date) -> int:
        """Return the age of this row in whole days at ``as_of``."""
        return (as_of - self.checked_on).days

    def is_fresh(self, *, as_of: date) -> bool:
        """Return whether this baseline source is within its freshness window."""
        age = self.age_days(as_of=as_of)
        return 0 <= age <= self.max_age_days and as_of <= self.refresh_due_on

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready baseline row."""
        return {
            "baseline_id": self.baseline_id,
            "display_name": self.display_name,
            "upstream_version": self.upstream_version,
            "source_url": self.source_url,
            "source_kind": self.source_kind,
            "checked_on": self.checked_on.isoformat(),
            "refresh_due_on": self.refresh_due_on.isoformat(),
            "max_age_days": self.max_age_days,
            "classification": self.classification,
            "scorecard_categories": list(self.scorecard_categories),
            "required_capabilities": list(self.required_capabilities),
            "hardening_implications": list(self.hardening_implications),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class CompetitiveBaselineRefresh:
    """Committed refresh bundle over differentiable competitive baselines."""

    schema: str
    artifact_id: str
    generated_on: date
    max_age_days: int
    rows: tuple[CompetitiveBaselineRow, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready refresh payload."""
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "generated_on": self.generated_on.isoformat(),
            "max_age_days": self.max_age_days,
            "claim_boundary": self.claim_boundary,
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class CompetitiveBaselineValidation:
    """Validation result for the competitive baseline refresh bundle."""

    passed: bool
    errors: tuple[str, ...]
    checked_baselines: tuple[CompetitiveBaselineId, ...]
    checked_categories: tuple[DifferentiableSOTACategory, ...]
    checked_urls: tuple[str, ...]
    as_of: date
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready validation metadata."""
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "checked_baselines": list(self.checked_baselines),
            "checked_categories": list(self.checked_categories),
            "checked_urls": list(self.checked_urls),
            "as_of": self.as_of.isoformat(),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class CompetitiveBaselinePromotionGate:
    """Combined baseline freshness and public-language promotion gate."""

    passed: bool
    errors: tuple[str, ...]
    baseline_validation: CompetitiveBaselineValidation
    language_audit: DifferentiableSOTAPromotionLanguageAudit
    checked_paths: tuple[str, ...]
    checked_categories: tuple[DifferentiableSOTACategory, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready combined gate metadata."""
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "baseline_validation": self.baseline_validation.to_dict(),
            "language_audit": self.language_audit.to_dict(),
            "checked_paths": list(self.checked_paths),
            "checked_categories": list(self.checked_categories),
            "claim_boundary": self.claim_boundary,
        }


def run_competitive_baseline_refresh(
    *,
    generated_on: date = date(2026, 6, 27),
) -> CompetitiveBaselineRefresh:
    """Build the deterministic competitive-baseline refresh bundle."""
    rows = tuple(_default_baseline_rows(generated_on=generated_on))
    return CompetitiveBaselineRefresh(
        schema=DIFFERENTIABLE_COMPETITIVE_BASELINE_SCHEMA,
        artifact_id=DIFFERENTIABLE_COMPETITIVE_BASELINE_ARTIFACT_ID,
        generated_on=generated_on,
        max_age_days=MAX_BASELINE_AGE_DAYS,
        rows=rows,
        claim_boundary=(
            "Competitive baseline freshness evidence only; it does not promote "
            "state-of-art, world-class, provider, hardware, GPU, QPU, "
            "production-performance, or isolated_affinity claims."
        ),
    )


def load_competitive_baseline_refresh(
    path: Path = DEFAULT_COMPETITIVE_BASELINE_REFRESH_PATH,
) -> CompetitiveBaselineRefresh:
    """Load a committed competitive-baseline refresh artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = _expect_mapping(payload, field_name="competitive baseline refresh")
    rows = tuple(
        _row_from_mapping(_expect_mapping(row, field_name="competitive baseline row"))
        for row in _sequence(mapping, "rows")
    )
    return CompetitiveBaselineRefresh(
        schema=_string(mapping, "schema"),
        artifact_id=_string(mapping, "artifact_id"),
        generated_on=_date(mapping, "generated_on"),
        max_age_days=_int(mapping, "max_age_days"),
        rows=rows,
        claim_boundary=_string(mapping, "claim_boundary"),
    )


def validate_competitive_baseline_refresh(
    refresh: CompetitiveBaselineRefresh | None = None,
    *,
    path: Path = DEFAULT_COMPETITIVE_BASELINE_REFRESH_PATH,
    as_of: date | None = None,
) -> CompetitiveBaselineValidation:
    """Validate baseline freshness, source provenance, and category coverage."""
    candidate = load_competitive_baseline_refresh(path) if refresh is None else refresh
    check_date = date.today() if as_of is None else as_of
    errors: list[str] = []
    baseline_ids = tuple(row.baseline_id for row in candidate.rows)
    categories = tuple(
        dict.fromkeys(category for row in candidate.rows for category in row.scorecard_categories)
    )
    urls = tuple(row.source_url for row in candidate.rows)

    if candidate.schema != DIFFERENTIABLE_COMPETITIVE_BASELINE_SCHEMA:
        errors.append(f"unexpected competitive baseline schema: {candidate.schema}")
    if candidate.artifact_id != DIFFERENTIABLE_COMPETITIVE_BASELINE_ARTIFACT_ID:
        errors.append(f"unexpected competitive baseline artifact_id: {candidate.artifact_id}")
    if candidate.max_age_days != MAX_BASELINE_AGE_DAYS:
        errors.append("competitive baseline bundle max_age_days changed")
    if tuple(sorted(baseline_ids)) != tuple(sorted(REQUIRED_BASELINE_IDS)):
        missing = tuple(
            identifier for identifier in REQUIRED_BASELINE_IDS if identifier not in baseline_ids
        )
        duplicate = _duplicates(baseline_ids)
        if missing:
            errors.append("missing competitive baseline rows: " + ", ".join(missing))
        if duplicate:
            errors.append("duplicate competitive baseline rows: " + ", ".join(duplicate))
    missing_categories = tuple(
        category for category in REQUIRED_SOTA_CATEGORIES if category not in categories
    )
    if missing_categories:
        errors.append("missing SOTA category baseline coverage: " + ", ".join(missing_categories))
    if "does not promote" not in candidate.claim_boundary:
        errors.append("competitive baseline claim boundary must remain non-promotional")

    for row in candidate.rows:
        if row.max_age_days != candidate.max_age_days:
            errors.append(f"{row.baseline_id}: max_age_days does not match bundle")
        if row.checked_on > check_date:
            errors.append(f"{row.baseline_id}: checked_on is in the future")
        if not row.is_fresh(as_of=check_date):
            errors.append(
                f"{row.baseline_id}: baseline source is stale "
                f"(checked_on={row.checked_on.isoformat()}, "
                f"refresh_due_on={row.refresh_due_on.isoformat()}, "
                f"as_of={check_date.isoformat()})"
            )
        if row.classification != "baseline_refresh_evidence":
            errors.append(f"{row.baseline_id}: unexpected classification")
        if row.source_kind not in SOURCE_KINDS:
            errors.append(f"{row.baseline_id}: unsupported source_kind={row.source_kind}")
        if not row.source_url.startswith("https://"):
            errors.append(f"{row.baseline_id}: source_url must use HTTPS")

    return CompetitiveBaselineValidation(
        passed=not errors,
        errors=tuple(errors),
        checked_baselines=baseline_ids,
        checked_categories=categories,
        checked_urls=urls,
        as_of=check_date,
        claim_boundary=(
            "Competitive baseline validation only; freshness and official-source "
            "coverage do not promote SOTA, provider, hardware, GPU, QPU, "
            "performance, or isolated_affinity claims."
        ),
    )


def audit_competitive_baseline_promotion_gate(
    *,
    refresh: CompetitiveBaselineRefresh | None = None,
    refresh_path: Path = DEFAULT_COMPETITIVE_BASELINE_REFRESH_PATH,
    as_of: date | None = None,
    public_texts: Mapping[str, str] | None = None,
    public_paths: Iterable[str] = DEFAULT_PUBLIC_SOTA_LANGUAGE_PATHS,
    ledger: ClaimLedger | None = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
) -> CompetitiveBaselinePromotionGate:
    """Validate freshness and public-language promotion evidence together."""
    candidate = load_competitive_baseline_refresh(refresh_path) if refresh is None else refresh
    baseline_validation = validate_competitive_baseline_refresh(candidate, as_of=as_of)
    language_audit = audit_differentiable_sota_promotion_language(
        public_texts=public_texts,
        public_paths=public_paths,
        ledger=ledger,
        ledger_path=ledger_path,
        repo_root=repo_root,
    )
    categories_with_fresh_baselines = {
        category
        for row in candidate.rows
        if row.is_fresh(as_of=baseline_validation.as_of)
        for category in row.scorecard_categories
    }
    errors = [
        f"competitive baseline validation failed: {error}" for error in baseline_validation.errors
    ]
    errors.extend(
        f"differentiable SOTA language audit failed: {error}" for error in language_audit.errors
    )
    for category in language_audit.checked_promotional_categories:
        if category not in categories_with_fresh_baselines:
            errors.append(f"{category}: public promotion wording lacks fresh baseline evidence")
    checked_categories = tuple(
        sorted(
            set(baseline_validation.checked_categories)
            | set(language_audit.checked_promotional_categories)
        )
    )
    return CompetitiveBaselinePromotionGate(
        passed=not errors,
        errors=tuple(errors),
        baseline_validation=baseline_validation,
        language_audit=language_audit,
        checked_paths=language_audit.checked_paths,
        checked_categories=checked_categories,
        claim_boundary=(
            "Combined competitive-baseline and public-language gate only; it "
            "keeps SOTA/world-class wording blocked unless fresh baseline, "
            "ready scorecard, and promoted claim-ledger evidence all agree."
        ),
    )


def render_competitive_baseline_refresh_markdown(
    refresh: CompetitiveBaselineRefresh,
) -> str:
    """Render a reviewer-facing Markdown summary of baseline sources."""
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — Differentiable competitive baseline refresh",
        "-->",
        "",
        "# Differentiable Competitive Baseline Refresh",
        "",
        f"- Schema: `{refresh.schema}`",
        f"- Artifact ID: `{refresh.artifact_id}`",
        f"- Generated on: `{refresh.generated_on.isoformat()}`",
        f"- Max age: `{refresh.max_age_days}` days",
        f"- Claim boundary: {refresh.claim_boundary}",
        "",
        "| Baseline | Version/source stream | Checked | Refresh due | Categories | Source |",
        "|---|---|---|---|---|---|",
    ]
    for row in refresh.rows:
        lines.append(
            "| `{baseline}` | {version} | `{checked}` | `{due}` | {categories} | "
            "[source]({url}) |".format(
                baseline=row.baseline_id,
                version=_markdown_cell(row.upstream_version),
                checked=row.checked_on.isoformat(),
                due=row.refresh_due_on.isoformat(),
                categories=_markdown_cell(", ".join(row.scorecard_categories)),
                url=row.source_url,
            )
        )
    lines.extend(
        (
            "",
            "This artefact is a freshness gate for comparison baselines. It does not "
            "promote any scorecard category; promotion still requires implementation, "
            "tests, docs, fresh baseline evidence, claim-ledger promotion, and "
            "benchmark artefacts.",
        )
    )
    return "\n".join(lines)


def _default_baseline_rows(*, generated_on: date) -> tuple[CompetitiveBaselineRow, ...]:
    due = generated_on + timedelta(days=MAX_BASELINE_AGE_DAYS)
    boundary = (
        "Official baseline source only; this row records comparison coverage and "
        "does not promote SCPN differentiable claims."
    )
    return (
        CompetitiveBaselineRow(
            baseline_id="jax",
            display_name="JAX",
            upstream_version="PyPI jax 0.10.2; official docs checked 2026-06-27",
            source_url="https://docs.jax.dev/",
            source_kind="official_docs",
            checked_on=generated_on,
            refresh_due_on=due,
            max_age_days=MAX_BASELINE_AGE_DAYS,
            scorecard_categories=("jax_native_transforms", "benchmark_promotion"),
            required_capabilities=(
                "grad/value_and_grad",
                "jacfwd/jacrev/hessian",
                "jvp/vjp",
                "jit/vmap/pmap",
                "PyTrees/sharding/export",
            ),
            hardening_implications=(
                "Keep JAX rows behind_baseline until transform breadth and isolated evidence exist.",
                "Refresh PyTree, sharding, and export blockers before any promotion wording.",
            ),
            claim_boundary=boundary,
        ),
        CompetitiveBaselineRow(
            baseline_id="pytorch",
            display_name="PyTorch",
            upstream_version="PyPI torch 2.12.1; official stable docs checked 2026-06-27",
            source_url="https://docs.pytorch.org/",
            source_kind="official_docs",
            checked_on=generated_on,
            refresh_due_on=due,
            max_age_days=MAX_BASELINE_AGE_DAYS,
            scorecard_categories=("pytorch_autograd_compile", "benchmark_promotion"),
            required_capabilities=(
                "autograd",
                "torch.func",
                "torch.compile",
                "AOTAutograd/Dynamo boundaries",
                "device and training-loop maturity",
            ),
            hardening_implications=(
                "Keep PyTorch rows behind_baseline until compile/device/training evidence exists.",
                "Require CUDA and fullgraph classification before performance wording.",
            ),
            claim_boundary=boundary,
        ),
        CompetitiveBaselineRow(
            baseline_id="tensorflow",
            display_name="TensorFlow",
            upstream_version="PyPI tensorflow 2.21.0; official docs checked 2026-06-27",
            source_url="https://www.tensorflow.org/guide/autodiff",
            source_kind="official_docs",
            checked_on=generated_on,
            refresh_due_on=due,
            max_age_days=MAX_BASELINE_AGE_DAYS,
            scorecard_categories=("pytorch_autograd_compile", "docs_api_maintainability"),
            required_capabilities=(
                "GradientTape",
                "tf.function",
                "XLA route classification",
                "Keras layer integration",
            ),
            hardening_implications=(
                "Use TensorFlow as a maintained compatibility route or explicitly rescope it.",
                "Remove stale Graph/XLA wording when dependency support changes.",
            ),
            claim_boundary=boundary,
        ),
        CompetitiveBaselineRow(
            baseline_id="pennylane",
            display_name="PennyLane",
            upstream_version="PyPI pennylane 0.45.1; official docs checked 2026-06-27",
            source_url="https://docs.pennylane.ai/",
            source_kind="official_docs",
            checked_on=generated_on,
            refresh_due_on=due,
            max_age_days=MAX_BASELINE_AGE_DAYS,
            scorecard_categories=("pennylane_qnode_device_plugin", "provider_hardware_gradients"),
            required_capabilities=(
                "QNode interfaces",
                "diff_method routing",
                "finite-shot gradients",
                "device/plugin execution",
            ),
            hardening_implications=(
                "Attach real provider-plugin artefacts before provider-backed wording.",
                "Keep hardware plugin rows blocked until live-ticket artefacts exist.",
            ),
            claim_boundary=boundary,
        ),
        CompetitiveBaselineRow(
            baseline_id="qiskit_algorithms",
            display_name="Qiskit Algorithms",
            upstream_version="PyPI qiskit-algorithms 0.4.0; official docs checked 2026-06-27",
            source_url="https://qiskit-community.github.io/qiskit-algorithms/",
            source_kind="official_docs",
            checked_on=generated_on,
            refresh_due_on=due,
            max_age_days=MAX_BASELINE_AGE_DAYS,
            scorecard_categories=(
                "qiskit_runtime_provider_gradients",
                "provider_hardware_gradients",
            ),
            required_capabilities=(
                "Estimator/Sampler primitives",
                "parameter-shift gradients",
                "finite-difference/LCU/SPSA gradients",
                "QGT/QFI workflows",
            ),
            hardening_implications=(
                "Require matched Runtime, raw-count, calibration, and gradient-workflow artefacts.",
                "Keep no-submit evidence distinct from live QPU evidence.",
            ),
            claim_boundary=boundary,
        ),
        CompetitiveBaselineRow(
            baseline_id="catalyst",
            display_name="PennyLane Catalyst",
            upstream_version="PyPI pennylane-catalyst 0.15.0; official docs checked 2026-06-27",
            source_url="https://docs.pennylane.ai/projects/catalyst/",
            source_kind="official_docs",
            checked_on=generated_on,
            refresh_due_on=due,
            max_age_days=MAX_BASELINE_AGE_DAYS,
            scorecard_categories=("catalyst_compiler_workflows", "enzyme_compiler_ad"),
            required_capabilities=(
                "qjit",
                "compiled quantum-classical workflows",
                "compiled control flow",
                "compiled differentiation",
            ),
            hardening_implications=(
                "Keep Catalyst parity a hard gap until configured qjit runner evidence exists.",
                "Compare compiled finite-shot and device routes separately.",
            ),
            claim_boundary=boundary,
        ),
        CompetitiveBaselineRow(
            baseline_id="enzyme_mlir",
            display_name="Enzyme/MLIR",
            upstream_version="GitHub EnzymeAD/Enzyme v0.0.276; official docs checked 2026-06-27",
            source_url="https://enzyme.mit.edu/",
            source_kind="official_project_docs",
            checked_on=generated_on,
            refresh_due_on=due,
            max_age_days=MAX_BASELINE_AGE_DAYS,
            scorecard_categories=("enzyme_compiler_ad", "rust_native_program_ad"),
            required_capabilities=(
                "LLVM AD",
                "MLIR-oriented AD",
                "reverse-mode compiler AD",
                "native compiler benchmark evidence",
            ),
            hardening_implications=(
                "Attach raw breadth artefacts before compiler-AD exceedance wording.",
                "Require isolated benchmark IDs for native compiler-performance wording.",
            ),
            claim_boundary=boundary,
        ),
        CompetitiveBaselineRow(
            baseline_id="julia_ad",
            display_name="Julia AD ecosystem",
            upstream_version=(
                "ChainRulesCore.jl v1.26.1; Zygote.jl v0.7.11; "
                "Reactant.jl v0.2.269; PyPI juliacall 0.9.35"
            ),
            source_url="https://juliadiff.org/ChainRulesCore.jl/stable/",
            source_kind="official_project_docs",
            checked_on=generated_on,
            refresh_due_on=due,
            max_age_days=MAX_BASELINE_AGE_DAYS,
            scorecard_categories=("rust_native_program_ad", "docs_api_maintainability"),
            required_capabilities=(
                "custom rules",
                "source-to-source AD",
                "mutation boundary documentation",
                "compiler-backed AD interop",
            ),
            hardening_implications=(
                "Track custom-rule ergonomics and mutation blockers against Julia AD references.",
                "Keep Program AD alias/effect semantics explicit before promotion.",
            ),
            claim_boundary=boundary,
        ),
        CompetitiveBaselineRow(
            baseline_id="emerging_ad",
            display_name="Emerging AD systems",
            upstream_version="PyPI tinygrad 0.13.0; MLIR EmitC docs checked 2026-06-27",
            source_url="https://mlir.llvm.org/docs/Dialects/EmitC/",
            source_kind="official_repository",
            checked_on=generated_on,
            refresh_due_on=due,
            max_age_days=MAX_BASELINE_AGE_DAYS,
            scorecard_categories=("adoption_licensing", "docs_api_maintainability"),
            required_capabilities=(
                "compiler IR interop",
                "stable extension points",
                "licensing/install ergonomics",
                "reviewer reproduction bundles",
            ),
            hardening_implications=(
                "Refresh emergent compiler/AD references before long-horizon promotion plans.",
                "Tie adoption claims to license and packaging decisions.",
            ),
            claim_boundary=boundary,
        ),
    )


def _row_from_mapping(payload: Mapping[str, object]) -> CompetitiveBaselineRow:
    baseline_id = _literal_baseline_id(_string(payload, "baseline_id"))
    source_kind = _literal_source_kind(_string(payload, "source_kind"))
    categories = tuple(
        _literal_sota_category(category) for category in _strings(payload, "scorecard_categories")
    )
    return CompetitiveBaselineRow(
        baseline_id=baseline_id,
        display_name=_string(payload, "display_name"),
        upstream_version=_string(payload, "upstream_version"),
        source_url=_string(payload, "source_url"),
        source_kind=source_kind,
        checked_on=_date(payload, "checked_on"),
        refresh_due_on=_date(payload, "refresh_due_on"),
        max_age_days=_int(payload, "max_age_days"),
        scorecard_categories=categories,
        required_capabilities=_strings(payload, "required_capabilities"),
        hardening_implications=_strings(payload, "hardening_implications"),
        claim_boundary=_string(payload, "claim_boundary"),
    )


def _literal_baseline_id(value: str) -> CompetitiveBaselineId:
    if value not in REQUIRED_BASELINE_IDS:
        raise ValueError(f"unknown competitive baseline: {value}")
    return value


def _literal_source_kind(value: str) -> CompetitiveBaselineSourceKind:
    if value not in SOURCE_KINDS:
        raise ValueError(f"unknown baseline source kind: {value}")
    return value


def _literal_sota_category(value: str) -> DifferentiableSOTACategory:
    if value not in REQUIRED_SOTA_CATEGORIES:
        raise ValueError(f"unknown SOTA category in baseline refresh: {value}")
    return value


def _expect_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return cast(Mapping[str, object], value)


def _sequence(payload: Mapping[str, object], field_name: str) -> tuple[object, ...]:
    value = payload.get(field_name)
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field_name} must be a JSON array")
    return tuple(value)


def _string(payload: Mapping[str, object], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _strings(payload: Mapping[str, object], field_name: str) -> tuple[str, ...]:
    values = _sequence(payload, field_name)
    strings: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must contain non-empty strings")
        strings.append(value)
    if not strings:
        raise ValueError(f"{field_name} must contain at least one value")
    return tuple(strings)


def _int(payload: Mapping[str, object], field_name: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _date(payload: Mapping[str, object], field_name: str) -> date:
    value = _string(payload, field_name)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date") from exc


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _markdown_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|")


__all__ = [
    "DEFAULT_COMPETITIVE_BASELINE_REFRESH_PATH",
    "DIFFERENTIABLE_COMPETITIVE_BASELINE_ARTIFACT_ID",
    "DIFFERENTIABLE_COMPETITIVE_BASELINE_SCHEMA",
    "MAX_BASELINE_AGE_DAYS",
    "REQUIRED_BASELINE_IDS",
    "CompetitiveBaselineId",
    "CompetitiveBaselinePromotionGate",
    "CompetitiveBaselineRefresh",
    "CompetitiveBaselineRow",
    "CompetitiveBaselineSourceKind",
    "CompetitiveBaselineValidation",
    "audit_competitive_baseline_promotion_gate",
    "load_competitive_baseline_refresh",
    "render_competitive_baseline_refresh_markdown",
    "run_competitive_baseline_refresh",
    "validate_competitive_baseline_refresh",
]
