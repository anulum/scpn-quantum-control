# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA parity — ordinary programme worker
"""Run the DLA-parity statistical reproducer as an ordinary programme.

The worker takes one request document that names every input by path, byte
length and SHA-256. It reads each input once, verifies it before parsing,
checks the records the statistics rest on, runs the existing numerical
functions of :mod:`.reproduce` unchanged, and writes exactly one JSON document
to standard output::

    python -m scpn_quantum_control.dla_parity.programme request.json

Three outcomes exist. ``completed`` (exit status 0): every scalar is emitted
and every comparison holds. ``mismatch`` (1): every scalar is emitted and at
least one comparison fails; all comparisons are listed, not only the first.
``refused`` (2): an input or a precondition is not acceptable; the document
carries one category from :data:`REFUSAL_CATEGORIES` and no scalar.

The numbers are those of :func:`.reproduce.compute_depth_summaries` and its
companions. The worker adds no estimator and repairs no value: the combined
p-value is emitted as the existing arithmetic gives it (one minus the
distribution function), and the document says so.

What is compared is stated, not implied. ``claims`` holds the comparisons of
the existing programme under its two rules. ``identity_checks`` holds exact
integer comparisons that the worker adds and the existing programme does not
make. ``comparison_coverage`` names every published field and every emitted
scalar that has no comparison at all.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Final

from .dataset import _parse_run
from .reproduce import (
    FisherResult,
    ReproductionTolerance,
    _check_claim,
    _fisher_combined,
    _peak_and_mean,
    compute_depth_summaries,
)
from .schema import DlaParityDataset, DlaParityRun, StatisticalSummary

REQUEST_SCHEMA: Final = "scpn_quantum_control.dla_parity.programme_request.v1"
OUTPUT_SCHEMA: Final = "scpn_quantum_control.dla_parity.programme_output.v1"
MAX_INPUT_BYTES: Final = 16 * 1024 * 1024
MAX_CIRCUITS: Final = 100_000
RAW_RUN_FILES: Final = 4
SELECTED_EXPERIMENT_PREFIX: Final = "A_dla_parity_n4"
SELECTED_QUBITS: Final = 4

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

_PER_DEPTH_CLAIMS: Final = (
    ("leakage_even", "mean_even", "leakage_even", "leakage_mean_abs", False),
    ("leakage_odd", "mean_odd", "leakage_odd", "leakage_mean_abs", False),
    ("sem_even", "sem_even", "leakage_even_sem", "leakage_sem_abs", False),
    ("sem_odd", "sem_odd", "leakage_odd_sem", "leakage_sem_abs", False),
    (
        "asymmetry_relative",
        "asymmetry_relative",
        "asymmetry_relative",
        "asymmetry_relative_abs",
        False,
    ),
    ("welch_t", "welch_t", "welch_t", "welch_t_rel", True),
    ("welch_p", "welch_p", "welch_p", "welch_p_rel", True),
)
_PUBLISHED_DEPTH_KEYS: Final = frozenset(
    {
        "asymmetry_absolute",
        "asymmetry_relative",
        "ci95_even",
        "ci95_odd",
        "degrees_of_freedom",
        "depth",
        "mean_even",
        "mean_odd",
        "n_reps_even",
        "n_reps_odd",
        "sem_even",
        "sem_odd",
        "welch_p",
        "welch_t",
    }
)
_PUBLISHED_FISHER_KEYS: Final = frozenset(
    {"chi2", "combined_p", "df", "n_depths_significant_at_0.05", "n_depths_tested"}
)
_TOLERANCE_NAMES: Final = tuple(item.name for item in fields(ReproductionTolerance))


class ProgrammeRefusal(Exception):
    """An input or a precondition of the programme is not acceptable.

    Parameters
    ----------
    category : str
        One of :data:`REFUSAL_CATEGORIES`.
    detail : str
        What was found, without any content of the refused input.

    """

    def __init__(self, category: str, detail: str) -> None:
        super().__init__(f"{category}: {detail}")
        self.category = category
        self.detail = detail


def _is_number(value: object) -> bool:
    """Tell whether a JSON value is a finite number and not a boolean."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_count(value: object) -> bool:
    """Tell whether a JSON value is a non-negative integer and not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _read_pinned(base: Path, pin: object, label: str) -> bytes:
    """Read one pinned input and verify its length and digest before any use.

    Parameters
    ----------
    base : Path
        Directory that relative paths of the request are resolved against.
    pin : object
        Request entry with ``path``, ``bytes`` and ``sha256``.
    label : str
        Name of the input used in a refusal.

    Returns
    -------
    bytes
        The verified content.

    Raises
    ------
    ProgrammeRefusal
        The entry is malformed, the file is missing or too large, or its
        length or digest differs from the pin.

    """
    if (
        not isinstance(pin, dict)
        or set(pin) != {"path", "bytes", "sha256"}
        or not isinstance(pin["path"], str)
        or not pin["path"]
        or not _is_count(pin["bytes"])
        or not isinstance(pin["sha256"], str)
        or len(pin["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in pin["sha256"])
    ):
        raise ProgrammeRefusal("malformed_request", f"{label} needs path, bytes and sha256")
    if pin["bytes"] > MAX_INPUT_BYTES:
        raise ProgrammeRefusal("input_too_large", f"{label} is declared above the input limit")
    path = base / pin["path"]
    try:
        with path.open("rb") as handle:
            content = handle.read(MAX_INPUT_BYTES + 1)
    except OSError as exc:
        raise ProgrammeRefusal("missing_input", f"{label} cannot be read") from exc
    if len(content) > MAX_INPUT_BYTES:
        raise ProgrammeRefusal("input_too_large", f"{label} exceeds the input limit")
    if len(content) != pin["bytes"] or hashlib.sha256(content).hexdigest() != pin["sha256"]:
        raise ProgrammeRefusal("digest_mismatch", f"{label} differs from its pin")
    return content


def _parse_object(content: bytes, category: str, label: str) -> dict[str, Any]:
    """Parse verified bytes as one JSON object, refusing anything else."""
    try:
        loaded = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProgrammeRefusal(category, f"{label} is not a JSON document") from exc
    if not isinstance(loaded, dict):
        raise ProgrammeRefusal(category, f"{label} is not a JSON object")
    return loaded


def _tolerance(raw: object) -> ReproductionTolerance:
    """Read the complete tolerance bundle of a request; nothing is defaulted."""
    if (
        not isinstance(raw, dict)
        or set(raw) != set(_TOLERANCE_NAMES)
        or not all(_is_number(value) and value >= 0 for value in raw.values())
    ):
        raise ProgrammeRefusal(
            "malformed_request", "tolerance needs its eight non-negative numbers"
        )
    return ReproductionTolerance(**{name: float(raw[name]) for name in _TOLERANCE_NAMES})


def _selection(runs: tuple[DlaParityRun, ...]) -> list[tuple[int, int, str, str]]:
    """Account for every record: run, position, experiment and why it is in or out."""
    records: list[tuple[int, int, str, str]] = []
    for run_index, run in enumerate(runs):
        for circuit_index, circuit in enumerate(run.circuits):
            meta = circuit.meta
            if meta.n_qubits != SELECTED_QUBITS:
                reason = "excluded_qubit_count"
            elif not meta.experiment.startswith(SELECTED_EXPERIMENT_PREFIX):
                reason = "excluded_experiment"
            elif meta.sector not in ("even", "odd"):
                reason = "excluded_sector"
            else:
                reason = "selected"
            records.append((run_index, circuit_index, meta.experiment, reason))
    return records


def _check_selected(
    runs: tuple[DlaParityRun, ...], records: list[tuple[int, int, str, str]]
) -> None:
    """Verify what the per-circuit scalar trusts, for every selected record.

    Raises
    ------
    ProgrammeRefusal
        A bitstring or the initial state has the wrong length or alphabet, a
        count is negative, a circuit has no shots, the declared shots differ
        from the counted ones, or the sector contradicts the parity of the
        initial state.

    """
    for run_index, circuit_index, _experiment, reason in records:
        if reason != "selected":
            continue
        where = f"run {run_index} circuit {circuit_index}"
        circuit = runs[run_index].circuits[circuit_index]
        meta = circuit.meta
        states = [meta.initial, *circuit.counts]
        if any(len(state) != SELECTED_QUBITS or set(state) - {"0", "1"} for state in states):
            raise ProgrammeRefusal(
                "malformed_bitstring", f"{where} has a state that is not four bits"
            )
        if any(count < 0 for count in circuit.counts.values()):
            raise ProgrammeRefusal("inconsistent_counts", f"{where} has a negative count")
        total = sum(circuit.counts.values())
        if total == 0:
            raise ProgrammeRefusal("empty_shots", f"{where} has no shots")
        if total != meta.shots:
            raise ProgrammeRefusal(
                "inconsistent_counts", f"{where} declares other shots than it counts"
            )
        if (meta.initial.count("1") % 2 == 0) != (meta.sector == "even"):
            raise ProgrammeRefusal("inconsistent_sector", f"{where} contradicts its initial state")


def _published(raw: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Validate the published summary and index its rows by depth.

    Raises
    ------
    ProgrammeRefusal
        The depth rows or the Fisher block are missing, malformed, hold a
        missing or an extra field, a non-finite number, or repeat a depth.

    """
    rows, fisher = raw.get("depth_summaries"), raw.get("fisher_combined")
    if not isinstance(rows, list) or not isinstance(fisher, dict):
        raise ProgrammeRefusal(
            "malformed_published_summary", "depth rows or the Fisher block are missing"
        )
    by_depth: dict[int, dict[str, Any]] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != _PUBLISHED_DEPTH_KEYS
            or not all(_is_count(row[key]) for key in ("depth", "n_reps_even", "n_reps_odd"))
            or not all(
                _is_number(row[key]) for key in _PUBLISHED_DEPTH_KEYS - {"ci95_even", "ci95_odd"}
            )
            or row["depth"] in by_depth
        ):
            raise ProgrammeRefusal(
                "malformed_published_summary", "a depth row is malformed or repeated"
            )
        by_depth[int(row["depth"])] = row
    if (
        set(fisher) != _PUBLISHED_FISHER_KEYS
        or not all(_is_number(fisher[key]) for key in ("chi2", "combined_p"))
        or not all(
            _is_count(fisher[key])
            for key in ("df", "n_depths_significant_at_0.05", "n_depths_tested")
        )
    ):
        raise ProgrammeRefusal("malformed_published_summary", "the Fisher block is malformed")
    return by_depth, fisher


def _claims(
    summaries: tuple[StatisticalSummary, ...],
    fisher: FisherResult,
    published: dict[int, dict[str, Any]],
    published_fisher: dict[str, Any],
    tolerance: ReproductionTolerance,
) -> list[dict[str, object]]:
    """Make every comparison of the existing programme and keep all outcomes.

    The existing check stops at the first failure. Here each claim is made on
    its own with the same function, so the difference and the rule are the
    programme's, and a failure does not hide the claims after it.
    """
    claims: list[dict[str, object]] = []

    def claim(
        name: str, expected: float, actual: float, tolerance_name: str, scaled: bool
    ) -> None:
        allowed = float(getattr(tolerance, tolerance_name))
        checked: list[tuple[str, float, float, float]] = []
        try:
            _check_claim(checked, name, expected, actual, allowed, rel=scaled)
            passed = True
        except AssertionError:
            passed = False
        claims.append(
            {
                "name": name,
                "published": expected,
                "recomputed": actual,
                "difference": checked[0][3],
                "rule": "scaled_by_max_abs_or_one" if scaled else "absolute",
                "tolerance_name": tolerance_name,
                "tolerance": allowed,
                "passed": passed,
            }
        )

    for summary in summaries:
        row = published[summary.depth]
        for label, key, attribute, tolerance_name, scaled in _PER_DEPTH_CLAIMS:
            claim(
                f"depth={summary.depth}.{label}",
                float(row[key]),
                float(getattr(summary, attribute)),
                tolerance_name,
                scaled,
            )
    claim("fisher.chi2", float(published_fisher["chi2"]), fisher.chi2, "fisher_chi2_rel", True)
    return claims


def _identity_checks(
    summaries: tuple[StatisticalSummary, ...],
    fisher: FisherResult,
    published: dict[int, dict[str, Any]],
    published_fisher: dict[str, Any],
) -> list[dict[str, object]]:
    """Compare the integer outputs exactly; these are the worker's own checks."""
    pairs: list[tuple[str, int, int]] = []
    for summary in summaries:
        row = published[summary.depth]
        pairs.append(
            (f"depth={summary.depth}.n_reps_even", int(row["n_reps_even"]), summary.n_reps_even)
        )
        pairs.append(
            (f"depth={summary.depth}.n_reps_odd", int(row["n_reps_odd"]), summary.n_reps_odd)
        )
    pairs.append(
        ("fisher.degrees_of_freedom", int(published_fisher["df"]), fisher.degrees_of_freedom)
    )
    pairs.append(
        (
            "fisher.n_depths_significant_at_0_05",
            int(published_fisher["n_depths_significant_at_0.05"]),
            fisher.n_depths_significant_at_0_05,
        )
    )
    pairs.append(
        (
            "fisher.n_depths_tested",
            int(published_fisher["n_depths_tested"]),
            fisher.n_depths_tested,
        )
    )
    return [
        {"name": name, "published": expected, "recomputed": actual, "passed": expected == actual}
        for name, expected, actual in pairs
    ]


_COVERAGE: Final[dict[str, object]] = {
    "published_fields_without_comparison": [
        {"field": "depth_summaries[].degrees_of_freedom", "why": "not an output of the programme"},
        {"field": "depth_summaries[].ci95_even", "why": "not an output of the programme"},
        {"field": "depth_summaries[].ci95_odd", "why": "not an output of the programme"},
        {"field": "depth_summaries[].asymmetry_absolute", "why": "not an output of the programme"},
        {
            "field": "fisher_combined.combined_p",
            "why": "the existing programme makes no claim on it; emitted as fisher.combined_p",
        },
    ],
    "emitted_scalars_without_published_reference": [
        "result.peak_asymmetry_relative",
        "result.peak_asymmetry_depth",
        "result.mean_asymmetry_relative",
        "result.n_circuits_used",
    ],
    "combined_p_arithmetic": "1.0 - chi2.cdf(chi2, degrees_of_freedom); a value of 0.0 is a loss of precision of this arithmetic, not a tail probability",
    "scaled_rule": "abs(published - recomputed) / max(abs(published), abs(recomputed), 1.0); absolute for values below one",
    "selected_domain": "four qubits, experiment beginning A_dla_parity_n4, sector even or odd",
}


def run_request(request_path: Path) -> dict[str, object]:
    """Run the programme for one request and return its output document.

    Parameters
    ----------
    request_path : Path
        Request document. Relative input paths are resolved against its
        directory.

    Returns
    -------
    dict
        The output document of a ``completed`` or ``mismatch`` outcome.

    Raises
    ------
    ProgrammeRefusal
        An input or a precondition is not acceptable.

    """
    try:
        request_bytes = request_path.read_bytes()
    except OSError as exc:
        raise ProgrammeRefusal("missing_input", "the request cannot be read") from exc
    request = _parse_object(request_bytes, "malformed_request", "the request")
    if (
        set(request) != {"schema", "raw_run_files", "published_summary", "tolerance"}
        or request["schema"] != REQUEST_SCHEMA
        or not isinstance(request["raw_run_files"], list)
        or len(request["raw_run_files"]) != RAW_RUN_FILES
    ):
        raise ProgrammeRefusal("malformed_request", "the request does not match its schema")
    tolerance = _tolerance(request["tolerance"])
    base = request_path.resolve().parent
    raw_bytes = [
        _read_pinned(base, pin, f"raw run file {index}")
        for index, pin in enumerate(request["raw_run_files"])
    ]
    summary_bytes = _read_pinned(base, request["published_summary"], "the published summary")

    parsed: list[DlaParityRun] = []
    for index, content in enumerate(raw_bytes):
        raw = _parse_object(content, "malformed_input", f"raw run file {index}")
        try:
            parsed.append(_parse_run(raw, f"raw run file {index}"))
        except (ValueError, TypeError, KeyError) as exc:
            raise ProgrammeRefusal(
                "malformed_input", f"raw run file {index} breaks the record schema"
            ) from exc
    runs = tuple(parsed)
    if sum(len(run.circuits) for run in runs) > MAX_CIRCUITS:
        raise ProgrammeRefusal("too_many_circuits", "the inputs exceed the circuit limit")
    published, published_fisher = _published(
        _parse_object(summary_bytes, "malformed_published_summary", "the published summary")
    )

    records = _selection(runs)
    reasons = [record[3] for record in records]
    if "selected" not in reasons:
        raise ProgrammeRefusal("empty_selection", "no record belongs to the selected domain")
    _check_selected(runs, records)

    summaries = compute_depth_summaries(DlaParityDataset(runs=runs))
    if any(summary.n_reps_even < 2 or summary.n_reps_odd < 2 for summary in summaries):
        raise ProgrammeRefusal(
            "insufficient_replicates", "a sector of a depth has fewer than two replicates"
        )
    if any(not 0.0 < summary.welch_p <= 1.0 for summary in summaries):
        raise ProgrammeRefusal("invalid_p_value", "a depth has no usable p-value")
    fisher = _fisher_combined(summaries)
    peak, peak_depth, mean = _peak_and_mean(summaries)
    scalars = [value for summary in summaries for value in asdict(summary).values()]
    scalars += [fisher.chi2, fisher.combined_p, peak, mean]
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ProgrammeRefusal("non_finite_output", "a computed scalar is not finite")
    if {summary.depth for summary in summaries} != set(published):
        raise ProgrammeRefusal("incomplete_depth_set", "computed and published depths differ")

    claims = _claims(summaries, fisher, published, published_fisher, tolerance)
    identity_checks = _identity_checks(summaries, fisher, published, published_fisher)
    agreed = all(bool(item["passed"]) for item in [*claims, *identity_checks])
    return {
        "schema": OUTPUT_SCHEMA,
        "outcome": "completed" if agreed else "mismatch",
        "inputs": {
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
            "raw_run_files": [
                {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                for content in raw_bytes
            ],
            "published_summary": {
                "bytes": len(summary_bytes),
                "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            },
        },
        "tolerance": asdict(tolerance),
        "selection": {
            "records": [
                {"run": run, "circuit": circuit, "experiment": experiment, "reason": reason}
                for run, circuit, experiment, reason in records
            ],
            "counts": {reason: reasons.count(reason) for reason in sorted(set(reasons))},
        },
        "depth_summaries": [asdict(summary) for summary in summaries],
        "fisher": asdict(fisher),
        "result": {
            "peak_asymmetry_relative": peak,
            "peak_asymmetry_depth": peak_depth,
            "mean_asymmetry_relative": mean,
            "n_circuits_used": sum(s.n_reps_even + s.n_reps_odd for s in summaries),
        },
        "claims": claims,
        "identity_checks": identity_checks,
        "comparison_coverage": _COVERAGE,
    }


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


def main(argv: list[str] | None = None) -> int:
    """Run one request and write one document to standard output.

    Parameters
    ----------
    argv : list of str, optional
        Arguments after the programme name: exactly one request path.

    Returns
    -------
    int
        0 for ``completed``, 1 for ``mismatch``, 2 for ``refused``.

    """
    arguments = sys.argv[1:] if argv is None else argv
    try:
        if len(arguments) != 1:
            raise ProgrammeRefusal("malformed_request", "exactly one request path is expected")
        document = run_request(Path(arguments[0]))
    except ProgrammeRefusal as refusal:
        document = {
            "schema": OUTPUT_SCHEMA,
            "outcome": "refused",
            "refusal": {"category": refusal.category, "detail": refusal.detail},
        }
    sys.stdout.write(serialise(document))
    return {"completed": 0, "mismatch": 1, "refused": 2}[str(document["outcome"])]


if __name__ == "__main__":
    raise SystemExit(main())
