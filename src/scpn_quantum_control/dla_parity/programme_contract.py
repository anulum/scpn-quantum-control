# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA parity — ordinary programme protocol
"""The protocol of the DLA-parity ordinary programme: names, limits and checks.

This module holds what the worker and a consumer of its output must agree on
and nothing numerical: the schema identifiers, the limits, the refusal
categories, a JSON reader that accepts one unambiguous object, the serialiser,
and :func:`validate_output_document`.

The JSON Schema shipped beside this module describes the shape of an output
document. A schema cannot say that the records add up, that the claims are the
complete population for the depths present, or that the outcome agrees with the
comparisons. :func:`validate_output_document` checks those; the worker applies
it to every document before writing it, and a consumer can apply it to a
document it received.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Final

REQUEST_SCHEMA: Final = "scpn_quantum_control.dla_parity.programme_request.v1"
OUTPUT_SCHEMA: Final = "scpn_quantum_control.dla_parity.programme_output.v1"
MAX_REQUEST_BYTES: Final = 64 * 1024
MAX_INPUT_BYTES: Final = 16 * 1024 * 1024
MAX_CIRCUITS: Final = 100_000
RAW_RUN_FILES: Final = 4

REFUSAL_CATEGORIES: Final = (
    "malformed_request",
    "missing_input",
    "input_too_large",
    "digest_mismatch",
    "malformed_input",
    "too_many_circuits",
    "malformed_bitstring",
    "empty_shots",
    "inconsistent_counts",
    "inconsistent_sector",
    "empty_selection",
    "insufficient_replicates",
    "invalid_p_value",
    "non_finite_output",
    "malformed_published_summary",
    "incomplete_depth_set",
)
SELECTION_REASONS: Final = (
    "selected",
    "excluded_qubit_count",
    "excluded_experiment",
    "excluded_sector",
)
PER_DEPTH_CLAIMS: Final = (
    "leakage_even",
    "leakage_odd",
    "sem_even",
    "sem_odd",
    "asymmetry_relative",
    "welch_t",
    "welch_p",
)
OPTIONAL_CLAIMS: Final = ("mean_asymmetry_relative", "peak_asymmetry_relative")
UNCOMPARED_PUBLISHED_FIELDS: Final = (
    {"field": "depth_summaries[].degrees_of_freedom", "why": "not an output of the programme"},
    {"field": "depth_summaries[].ci95_even", "why": "not an output of the programme"},
    {"field": "depth_summaries[].ci95_odd", "why": "not an output of the programme"},
    {"field": "depth_summaries[].asymmetry_absolute", "why": "not an output of the programme"},
    {
        "field": "fisher_combined.combined_p",
        "why": "the existing programme makes no claim on it; emitted as fisher.combined_p",
    },
)
"""Published fields that no run compares, with the reason; always the complete list."""
UNREFERENCED_ALWAYS: Final = ("result.peak_asymmetry_depth", "result.n_circuits_used")
"""Emitted scalars for which the published summary has no reference in any run."""
COVERAGE_TEXT: Final = {
    "combined_p_arithmetic": (
        "1.0 - chi2.cdf(chi2, degrees_of_freedom); a value of 0.0 is a loss of precision "
        "of this arithmetic, not a tail probability"
    ),
    "scaled_rule": (
        "abs(published - recomputed) / max(abs(published), abs(recomputed), 1.0); "
        "absolute for values below one"
    ),
    "selected_domain": "four qubits, experiment beginning A_dla_parity_n4, sector even or odd",
}
"""The fixed statements of the coverage block."""
CLAIM_BINDINGS: Final = {
    "leakage_even": ("leakage_even", "leakage_mean_abs", "absolute"),
    "leakage_odd": ("leakage_odd", "leakage_mean_abs", "absolute"),
    "sem_even": ("leakage_even_sem", "leakage_sem_abs", "absolute"),
    "sem_odd": ("leakage_odd_sem", "leakage_sem_abs", "absolute"),
    "asymmetry_relative": ("asymmetry_relative", "asymmetry_relative_abs", "absolute"),
    "welch_t": ("welch_t", "welch_t_rel", "scaled_by_max_abs_or_one"),
    "welch_p": ("welch_p", "welch_p_rel", "scaled_by_max_abs_or_one"),
    "fisher.chi2": ("chi2", "fisher_chi2_rel", "scaled_by_max_abs_or_one"),
    "mean_asymmetry_relative": ("mean_asymmetry_relative", "mean_asymmetry_abs", "absolute"),
    "peak_asymmetry_relative": ("peak_asymmetry_relative", "peak_asymmetry_abs", "absolute"),
}
"""For each claim: the emitted scalar it compares, its tolerance and its rule."""
COVERAGE_KEYS: Final = frozenset(
    {
        "published_fields_without_comparison",
        "emitted_scalars_without_published_reference",
        "combined_p_arithmetic",
        "scaled_rule",
        "selected_domain",
    }
)
_RESULT_KEYS: Final = frozenset(
    {
        "schema",
        "outcome",
        "inputs",
        "tolerance",
        "selection",
        "depth_summaries",
        "fisher",
        "result",
        "claims",
        "identity_checks",
        "comparison_coverage",
    }
)


def is_number(value: object) -> bool:
    """Tell whether a JSON value is a number that binary64 can hold, and not a boolean.

    An integer too large for binary64 is not a number of this protocol; it is
    refused here instead of failing later in a conversion.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def is_count(value: object) -> bool:
    """Tell whether a JSON value is a non-negative integer and not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _unique_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object, refusing a member name that occurs twice."""
    members = dict(pairs)
    if len(members) != len(pairs):
        raise ValueError("a JSON object repeats a member name")
    return members


def _no_constant(name: str) -> Any:
    """Refuse the non-finite literals that the JSON grammar does not have."""
    raise ValueError(f"{name} is not a JSON value")


def parse_json_object(content: bytes) -> dict[str, Any]:
    """Read bytes as one unambiguous JSON object.

    Parameters
    ----------
    content : bytes
        Complete document.

    Returns
    -------
    dict
        The object.

    Raises
    ------
    ValueError
        The bytes are not UTF-8, not JSON, nested beyond what the reader can
        follow, hold ``NaN`` or an infinity, repeat a member name in any
        object, or the document is not an object.

    """
    try:
        loaded = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_members,
            parse_constant=_no_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError("the document is not UTF-8") from exc
    except RecursionError as exc:
        raise ValueError("the document is nested too deeply") from exc
    if not isinstance(loaded, dict):
        raise ValueError("the document is not a JSON object")
    return loaded


def serialise(document: dict[str, object]) -> str:
    """Write an output document as one deterministic line of JSON.

    Keys are sorted, separators are fixed, non-ASCII characters are escaped
    and a non-finite number is refused rather than written. A binary64 value
    is written in its shortest form that reads back to the same value.

    Parameters
    ----------
    document : dict
        Output document.

    Returns
    -------
    str
        The line, ending in a newline.

    """
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _require(condition: bool, message: str) -> None:
    """Raise the validation error of an output document."""
    if not condition:
        raise ValueError(f"output document: {message}")


def validate_output_document(document: dict[str, Any], request: bytes | None = None) -> None:
    """Check the invariants of an output document that a schema cannot express.

    Parameters
    ----------
    document : dict
        A parsed output document.
    request : bytes, optional
        The request the document answers, exactly as it was given to the
        worker. When supplied, the document must carry the digest of these
        bytes, the lengths and digests of the inputs the request pins, in
        order, and the tolerances the request sets. Without it the document is
        checked against itself only: a consumer that wants to know that the
        document answers *its* request supplies the request.

    Raises
    ------
    ValueError
        The schema identifier or the outcome is unknown; a refusal carries a
        scalar or an unknown category; the records are not the complete,
        ordered, unique account that the counts state; the selected count,
        the replicate counts and the circuits used disagree; the depths are
        not ascending and unique; the claims or the identity checks are not
        exactly the population for the depths and optional references present;
        a claim or an identity check does not carry the emitted scalar, its
        own rule and the tolerance of the document, or states another
        difference than its two values give under its rule;
        the records are not every circuit of every run the inputs declare;
        the coverage block is incomplete, is not the complete fixed statement
        or names a scalar that has a claim; with a request, the document does
        not answer it;
        or the outcome contradicts the comparisons.

    """
    _require(document.get("schema") == OUTPUT_SCHEMA, "unknown schema identifier")
    outcome = document.get("outcome")
    if outcome == "refused":
        refusal = document.get("refusal")
        _require(set(document) == {"schema", "outcome", "refusal"}, "a refusal carries a scalar")
        _require(
            isinstance(refusal, dict)
            and set(refusal) == {"category", "detail"}
            and refusal["category"] in REFUSAL_CATEGORIES
            and isinstance(refusal["detail"], str),
            "malformed refusal",
        )
        return
    _require(outcome in ("completed", "mismatch"), "unknown outcome")
    _require(set(document) == _RESULT_KEYS, "missing or extra top-level member")

    records = document["selection"]["records"]
    counts = document["selection"]["counts"]
    positions = [(record["run"], record["circuit"]) for record in records]
    reasons = [record["reason"] for record in records]
    _require(positions == sorted(set(positions)), "records are not ordered and unique")
    declared = [entry["circuits"] for entry in document["inputs"]["raw_run_files"]]
    _require(
        all(is_count(total) for total in declared)
        and sum(declared) == len(records) <= MAX_CIRCUITS,
        "declared run sizes do not bound the records",
    )
    _require(
        positions
        == [(run, circuit) for run, total in enumerate(declared) for circuit in range(total)],
        "records are not every circuit of every run",
    )
    _require(set(reasons) <= set(SELECTION_REASONS), "unknown selection reason")
    _require(
        counts == {reason: reasons.count(reason) for reason in sorted(set(reasons))},
        "selection counts do not account for the records",
    )

    summaries = document["depth_summaries"]
    depths = [summary["depth"] for summary in summaries]
    used = sum(summary["n_reps_even"] + summary["n_reps_odd"] for summary in summaries)
    _require(depths == sorted(set(depths)) and bool(depths), "depths are not ascending and unique")
    _require(
        used == document["result"]["n_circuits_used"] == counts.get("selected"),
        "selected records, replicates and circuits used disagree",
    )
    _require(document["result"]["peak_asymmetry_depth"] in depths, "peak depth is not a depth")

    coverage = document["comparison_coverage"]
    _require(set(coverage) == COVERAGE_KEYS, "incomplete comparison coverage")
    unreferenced = coverage["emitted_scalars_without_published_reference"]
    _require(
        coverage["published_fields_without_comparison"] == list(UNCOMPARED_PUBLISHED_FIELDS)
        and all(coverage[key] == text for key, text in COVERAGE_TEXT.items()),
        "comparison coverage is not the complete fixed statement",
    )
    _require(
        unreferenced
        == [f"result.{name}" for name in OPTIONAL_CLAIMS if f"result.{name}" in unreferenced]
        + list(UNREFERENCED_ALWAYS),
        "unreferenced scalars are not the complete ordered list",
    )
    optional = [name for name in OPTIONAL_CLAIMS if f"result.{name}" not in unreferenced]
    expected_claims = [f"depth={d}.{label}" for d in depths for label in PER_DEPTH_CLAIMS]
    expected_claims += ["fisher.chi2", *optional]
    claims = document["claims"]
    _require(
        [claim["name"] for claim in claims] == expected_claims,
        "claims are not the complete population",
    )
    expected_checks = [
        f"depth={d}.{label}" for d in depths for label in ("n_reps_even", "n_reps_odd")
    ]
    expected_checks += [
        "fisher.degrees_of_freedom",
        "fisher.n_depths_significant_at_0_05",
        "fisher.n_depths_tested",
    ]
    checks = document["identity_checks"]
    _require(
        [check["name"] for check in checks] == expected_checks,
        "identity checks are not the complete population",
    )
    by_depth = {summary["depth"]: summary for summary in summaries}
    for claim in claims:
        name = claim["name"]
        depth_text, _, label = name.partition(".") if name.startswith("depth=") else ("", "", name)
        field, tolerance_name, rule = CLAIM_BINDINGS[label]
        if depth_text:
            emitted = by_depth[int(depth_text.removeprefix("depth="))][field]
        else:
            emitted = document["fisher" if label == "fisher.chi2" else "result"][field]
        published, recomputed = claim["published"], claim["recomputed"]
        difference = abs(published - recomputed)
        if rule != "absolute":
            difference /= max(abs(published), abs(recomputed), 1.0)
        _require(recomputed == emitted, f"claim {name} does not carry the emitted scalar")
        _require(
            claim["rule"] == rule
            and claim["tolerance_name"] == tolerance_name
            and claim["tolerance"] == document["tolerance"][tolerance_name],
            f"claim {name} does not carry its rule and tolerance",
        )
        _require(claim["difference"] == difference, f"claim {name} misstates its difference")
        _require(
            claim["passed"] is (difference <= claim["tolerance"]),
            f"claim {name} contradicts its difference",
        )
    fisher = document["fisher"]
    for check in checks:
        name = check["name"]
        if name.startswith("depth="):
            depth_text, _, field = name.partition(".")
            emitted = by_depth[int(depth_text.removeprefix("depth="))][field]
        else:
            emitted = fisher[name.removeprefix("fisher.")]
        _require(
            check["recomputed"] == emitted,
            f"identity check {name} does not carry the emitted count",
        )
        _require(
            check["passed"] is (check["published"] == check["recomputed"]),
            f"identity check {name} contradicts its values",
        )
    agreed = all(item["passed"] for item in [*claims, *checks])
    _require(
        outcome == ("completed" if agreed else "mismatch"), "outcome contradicts the comparisons"
    )
    if request is not None:
        _bound_to_request(document, request)


def _bound_to_request(document: dict[str, Any], request: bytes) -> None:
    """Require a result document to answer exactly the given request."""
    inputs = document["inputs"]
    try:
        asked = parse_json_object(request)
        pins = [*asked["raw_run_files"], asked["published_summary"]]
        pinned = [(pin["bytes"], pin["sha256"]) for pin in pins]
        tolerance = {name: float(value) for name, value in asked["tolerance"].items()}
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("output document: the request is not a request of this protocol") from exc
    answered = [*inputs["raw_run_files"], inputs["published_summary"]]
    _require(
        inputs["request_sha256"] == hashlib.sha256(request).hexdigest(),
        "the document answers another request",
    )
    _require(
        [(entry["bytes"], entry["sha256"]) for entry in answered] == pinned,
        "the document read other inputs than the request pins",
    )
    _require(
        document["tolerance"] == tolerance, "the document used other tolerances than the request"
    )
