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


def validate_output_document(document: dict[str, Any]) -> None:
    """Check the invariants of an output document that a schema cannot express.

    Parameters
    ----------
    document : dict
        A parsed output document.

    Raises
    ------
    ValueError
        The schema identifier or the outcome is unknown; a refusal carries a
        scalar or an unknown category; the records are not the complete,
        ordered, unique account that the counts state; the selected count,
        the replicate counts and the circuits used disagree; the depths are
        not ascending and unique; the claims or the identity checks are not
        exactly the population for the depths and optional references present;
        the coverage block is incomplete or names a scalar that has a claim;
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
    for claim in claims:
        _require(
            claim["passed"] is (claim["difference"] <= claim["tolerance"]),
            f"claim {claim['name']} contradicts its difference",
        )
    for check in checks:
        _require(
            check["passed"] is (check["published"] == check["recomputed"]),
            f"identity check {check['name']} contradicts its values",
        )
    agreed = all(item["passed"] for item in [*claims, *checks])
    _require(
        outcome == ("completed" if agreed else "mismatch"), "outcome contradicts the comparisons"
    )
