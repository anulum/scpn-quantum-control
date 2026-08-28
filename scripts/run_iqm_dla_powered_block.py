# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — IQM powered DLA backend-sensitivity block runner
"""Powered DLA backend-sensitivity block runner (Lane 1, ``.venv-iqm`` side).

Implements the circuit matrix of
``docs/campaigns/iqm_dla_backend_sensitivity_powered_prereg_2026-07-21.md``:
2 states (`0011` even / `0001` odd) × 3 depths (4, 6, 10) × 4 repetitions at
1,024 shots each, plus 4 readout states (`0011`, `0001`, `0000`, `1111`) at
2,048 shots, on the pinned layout `[2, 7, 12, 13]` (fallback `[9, 4, 3, 8]`,
substitution recorded). Circuits come from the committed campaign builders
(`scripts/iqm_fake_transpile_payload.py`), identical to the May 13 runs.

``dry-run`` (default) targets ``IQMFakeGarnet`` and enforces the live
readiness gates: full-matrix transpilation, the depth envelope (May 13 d10
transpiled depth 159 plus 25 %), and a full noisy execution with counts.
``submit`` needs ``--i-have-owner-go`` and writes a crash-safe journal before
each provider call. A restart reuses completed groups and fails closed on an
ambiguous call; ``recover`` can bind a dashboard-verified IQM job only after
its complete provider payload matches the frozen pre-submit digest.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import UUID

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "scripts" / "iqm_fake_transpile_payload.py"
VAULT_PATH = Path("~/.config/scpn-quantum-control/credentials.md").expanduser()

PRIMARY_LAYOUT = (2, 7, 12, 13)
FALLBACK_LAYOUT = (9, 4, 3, 8)
SECTORS = {"even": "0011", "odd": "0001"}
READOUT_STATES = ("0011", "0001", "0000", "1111")
REPETITIONS = 4
MAIN_SHOTS = 1024
READOUT_SHOTS = 2048
JOURNAL_SCHEMA = "scpn.iqm-submission-journal.v2"
RECOVERY_REQUIRED_EXIT = 4
_PAYLOAD_PARAMETER_FIELDS = (
    "calibration_set_id",
    "qubit_mapping",
    "shots",
    "max_circuit_duration_over_t2",
    "heralding_mode",
    "move_gate_validation",
    "move_gate_frame_tracking",
    "active_reset_cycles",
    "dd_mode",
    "dd_strategy",
)


class _IQMCircuitJob(Protocol):
    """Subset of the IQM client job contract used by the journal."""

    @property
    def job_id(self) -> object:
        """Return the durable provider identifier."""

    def payload(self) -> tuple[list[object], object]:
        """Return the submitted circuits and execution parameters."""


class _IQMClient(Protocol):
    """Subset of the IQM client required for submit and recovery."""

    def submit_run_request(
        self, run_request: object, use_timeslot: bool = False
    ) -> _IQMCircuitJob:
        """Submit an already validated run request."""

    def get_job(self, job_id: UUID) -> _IQMCircuitJob:
        """Return a provider job by its durable identifier."""


class _IQMResult(Protocol):
    """Counts-bearing result contract returned by a completed IQM job."""

    def get_counts(self) -> dict[str, int] | list[dict[str, int]]:
        """Return one count mapping per submitted circuit."""


class _IQMQiskitJob(Protocol):
    """Polling surface exposed by Qiskit's IQM job wrapper."""

    def done(self) -> bool:
        """Return whether the job reached a terminal state."""

    def status(self) -> object:
        """Return the current provider status."""

    def result(self) -> _IQMResult:
        """Return the terminal result."""


class _IQMBackend(Protocol):
    """Live IQM backend surface used by this runner."""

    @property
    def client(self) -> _IQMClient:
        """Return the underlying client used for exact payload submission."""

    def create_run_request(self, run_input: list[Any], *, shots: int) -> object:
        """Create and validate the exact provider payload without submitting it."""

    def retrieve_job(self, job_id: str) -> _IQMQiskitJob:
        """Return the Qiskit polling wrapper for an existing provider job."""


#: Per-campaign frozen depths and per-depth transpiled-depth envelopes. The
#: powered block uses the May 13 d10 reference (159) + 25 % for every depth;
#: the depth-profile follow-up freezes the interpolated ladder (~15 layers
#: per Trotter step: d8 -> 129, d12 -> 189) + 25 % per its preregistration.
CAMPAIGNS: dict[str, dict[str, Any]] = {
    "powered": {
        "campaign_id": "iqm_dla_backend_sensitivity_powered_prereg_2026-07-21",
        "depths": (4, 6, 10),
        "envelope": {4: int(159 * 1.25), 6: int(159 * 1.25), 10: int(159 * 1.25)},
    },
    "depth-profile": {
        "campaign_id": "iqm_dla_depth_profile_prereg_2026-07-22",
        "depths": (8, 12),
        "envelope": {8: int(129 * 1.25), 12: int(189 * 1.25)},
    },
    # d10 sign-replication: 8 execution-order repetitions batched into ONE
    # main job + one readout job (frozen batching disclosure in the prereg).
    "d10-retest": {
        "campaign_id": "iqm_dla_d10_retest_prereg_2026-07-22",
        "depths": (10,),
        "envelope": {10: int(159 * 1.25)},
        "repetitions": 8,
        "batch_all": True,
    },
    # Window variability: every window submits the full 36-circuit
    # matrix in one pass (mains one job + a per-window readout job); the
    # WINDOW is the independent unit and `--window` stamps the artefacts.
    "window-variability": {
        "campaign_id": "iqm_dla_window_variability_prereg_2026-07-22",
        "depths": (4, 8, 10, 12),
        "envelope": {
            4: int(69 * 1.25),
            8: int(129 * 1.25),
            10: int(159 * 1.25),
            12: int(189 * 1.25),
        },
        "batch_all": True,
    },
}


def _load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("iqm_fake_transpile_payload", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load IQM circuit helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_powered_plan(
    *,
    layout: tuple[int, int, int, int],
    depths: tuple[int, ...] = (4, 6, 10),
    repetitions: int = REPETITIONS,
) -> list[dict[str, Any]]:
    """Preregistered matrix for ``depths``: mains per repetition + 4 readout rows."""
    rows: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for depth in depths:
            for sector, initial in SECTORS.items():
                rows.append(
                    {
                        "tier": "dla_parity_powered_backend_sensitivity",
                        "circuit_name": f"iqm_dla_pinned_n4_d{depth}_{sector}",
                        "label": f"main_d{depth}_{sector}_rep{repetition}",
                        "kind": "dla_parity",
                        "repetition": repetition,
                        "shots": MAIN_SHOTS,
                        "requested_initial_layout": list(layout),
                        "meta": {
                            "experiment": "A_dla_parity_n4",
                            "n_qubits": 4,
                            "depth": depth,
                            "sector": sector,
                            "initial": initial,
                            "t_step": 0.3,
                            "paper_source": "phase1_dla_parity",
                        },
                    }
                )
    for initial in READOUT_STATES:
        rows.append(
            {
                "tier": "dla_readout_powered_baseline",
                "circuit_name": f"iqm_readout_pinned_state_{initial}",
                "label": f"readout_{initial}",
                "kind": "readout_baseline",
                "repetition": 0,
                "shots": READOUT_SHOTS,
                "requested_initial_layout": list(layout),
                "meta": {
                    "experiment": "C_readout_baseline",
                    "n_qubits": 4,
                    "sector": "calibration",
                    "initial": initial,
                    "paper_source": "phase1_dla_parity",
                },
            }
        )
    return rows


def _utc_now() -> str:
    """Return a stable UTC timestamp for journal transitions."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_default(value: object) -> object:
    """Serialise IQM dataclasses/Pydantic models, UUIDs, enums, and containers."""
    if is_dataclass(value) and not isinstance(value, type):
        return cast(object, asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return cast(
            object,
            model_dump(
                mode="json",
                exclude={"move_validation_mode", "move_gate_frame_tracking_mode"},
            ),
        )
    enum_value = getattr(value, "value", None)
    if enum_value is not None and isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"cannot serialise {type(value).__name__} into an IQM payload digest")


def _payload_digest(circuits: list[object], parameters: object) -> str:
    """Hash the exact circuit payload and all execution-affecting parameters."""
    material: dict[str, object] = {"circuits": circuits}
    for field in _PAYLOAD_PARAMETER_FIELDS:
        if not hasattr(parameters, field):
            raise ValueError(f"IQM payload is missing execution parameter {field!r}")
        material[field] = getattr(parameters, field)
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _job_identifier(job: object) -> str:
    """Return a non-empty provider job identifier from either IQM job surface."""
    raw = getattr(job, "job_id", None)
    value = raw() if callable(raw) else raw
    job_id = str(value or "").strip()
    if not job_id:
        raise RuntimeError("IQM provider returned a job without a durable identifier")
    try:
        UUID(job_id)
    except ValueError as exc:
        raise RuntimeError(f"IQM provider returned a non-UUID job identifier {job_id!r}") from exc
    return job_id


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object, rejecting arrays and scalar payloads."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON journal and fsync both file and directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def _journal_lock(path: Path) -> Iterator[None]:
    """Serialise writers for one journal without adding artefacts beside it."""
    git_path = REPO_ROOT / ".git"
    if git_path.is_file():
        prefix = "gitdir: "
        marker = git_path.read_text(encoding="utf-8").strip()
        if not marker.startswith(prefix):
            raise RuntimeError(f"invalid linked-worktree marker: {git_path}")
        git_path = (REPO_ROOT / marker[len(prefix) :]).resolve()
        common_dir_marker = git_path / "commondir"
        if common_dir_marker.is_file():
            common_dir = common_dir_marker.read_text(encoding="utf-8").strip()
            if not common_dir:
                raise RuntimeError(f"empty linked-worktree common-dir marker: {common_dir_marker}")
            git_path = (git_path / common_dir).resolve()
    if not git_path.is_dir():
        raise RuntimeError(f"Git metadata directory is unavailable: {git_path}")
    lock_root = git_path / "scpn-qpu-journal-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    lock_path = lock_root / f"{key}.lock"
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _journal_jobs(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return and structurally validate the journal job-group list."""
    raw_jobs = record.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("submission journal must contain at least one job group")
    jobs: list[dict[str, Any]] = []
    group_ids: set[str] = set()
    provider_ids: set[str] = set()
    for index, raw in enumerate(raw_jobs):
        if not isinstance(raw, dict):
            raise ValueError(f"submission journal job {index} must be an object")
        group_id = raw.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"submission journal job {index} has no group_id")
        if group_id in group_ids:
            raise ValueError(f"duplicate submission journal group_id {group_id!r}")
        if group_id not in {"main", "readout"}:
            raise ValueError(f"submission journal has unknown group_id {group_id!r}")
        group_ids.add(group_id)
        state = raw.get("state")
        if state not in {"prepared", "submitting", "recovery_required", "submitted"}:
            raise ValueError(f"submission journal group {group_id!r} has invalid state {state!r}")
        job_id = raw.get("job_id")
        shots = raw.get("shots")
        labels = raw.get("labels")
        circuit_names = raw.get("circuit_names")
        payload_sha256 = raw.get("payload_sha256")
        if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
            raise ValueError(f"submission journal group {group_id!r} has invalid shots")
        if (
            not isinstance(labels, list)
            or not labels
            or not all(isinstance(label, str) and label for label in labels)
        ):
            raise ValueError(f"submission journal group {group_id!r} has invalid labels")
        if (
            not isinstance(circuit_names, list)
            or len(circuit_names) != len(labels)
            or not all(isinstance(name, str) and name for name in circuit_names)
        ):
            raise ValueError(f"submission journal group {group_id!r} has invalid circuit_names")
        if (
            not isinstance(payload_sha256, str)
            or len(payload_sha256) != 64
            or any(character not in "0123456789abcdef" for character in payload_sha256)
        ):
            raise ValueError(f"submission journal group {group_id!r} has invalid payload_sha256")
        if state == "submitted":
            if not isinstance(job_id, str) or not job_id:
                raise ValueError(f"submitted journal group {group_id!r} has no job_id")
            try:
                UUID(job_id)
            except ValueError as exc:
                raise ValueError(
                    f"submitted journal group {group_id!r} has a non-UUID job_id"
                ) from exc
            if job_id in provider_ids:
                raise ValueError(f"provider job_id {job_id!r} is bound more than once")
            provider_ids.add(job_id)
        elif job_id is not None:
            raise ValueError(f"non-submitted journal group {group_id!r} already has a job_id")
        jobs.append(raw)
    return jobs


def _validate_journal_header(
    record: dict[str, Any],
    *,
    args: argparse.Namespace,
    campaign_id: str,
    layout: tuple[int, int, int, int],
) -> None:
    """Reject reuse of an output journal for a different frozen submission."""
    expected = {
        "schema": JOURNAL_SCHEMA,
        "campaign": campaign_id,
        "quantum_computer": args.quantum_computer,
        "date": args.date,
        "repetition": args.repetition,
        "window": args.window,
        "layout": list(layout),
        "layout_choice": args.layout,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(
                f"submission journal {field!r} mismatch: expected {value!r}, "
                f"found {record.get(field)!r}"
            )
    _journal_jobs(record)


def _fake_backend() -> Any:
    """Return an ``IQMFakeGarnet`` instance (import deferred to ``.venv-iqm``)."""
    module = importlib.import_module("iqm.qiskit_iqm.fake_backends.fake_garnet")
    backend_class = module.IQMFakeGarnet
    return backend_class()


def dry_run(args: argparse.Namespace) -> int:
    """Transpile + noisily execute the full matrix on IQMFakeGarnet."""
    from qiskit import transpile

    helper = _load_helper()
    campaign = CAMPAIGNS[args.campaign]
    layout = PRIMARY_LAYOUT if args.layout == "primary" else FALLBACK_LAYOUT
    backend = _fake_backend()
    rows = build_powered_plan(
        layout=layout,
        depths=campaign["depths"],
        repetitions=int(campaign.get("repetitions", REPETITIONS)),
    )

    # Repetitions reuse the identical circuit; build/transpile each unique one once.
    unique: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    envelope_violations: list[str] = []
    for row in rows:
        name = row["circuit_name"]
        if name not in unique:
            circuit = helper._build_circuit({"circuit_name": name, "meta": row["meta"]})
            circuit.name = name
            isa = transpile(
                circuit, backend=backend, initial_layout=list(layout), optimization_level=1
            )
            unique[name] = isa
        isa = unique[name]
        depth = int(isa.depth())
        bound = (
            int(campaign["envelope"][int(row["meta"].get("depth", 0))])
            if row["kind"] == "dla_parity"
            else max(campaign["envelope"].values())
        )
        if depth > bound:
            envelope_violations.append(f"{row['label']} depth {depth} > {bound}")
        records.append(
            {
                "label": row["label"],
                "circuit_name": name,
                "shots": row["shots"],
                "transpiled_depth": depth,
                "transpiled_ops": {str(k): int(v) for k, v in isa.count_ops().items()},
            }
        )
        result = backend.run(isa, shots=row["shots"]).result()
        counts[row["label"]] = {str(k): int(v) for k, v in result.get_counts().items()}

    payload = {
        "campaign": campaign["campaign_id"],
        "kind": "fake_backend_dry_run",
        "backend": "IQMFakeGarnet",
        "date": args.date,
        "layout": list(layout),
        "layout_choice": args.layout,
        "depth_envelope": {str(k): int(v) for k, v in campaign["envelope"].items()},
        "envelope_violations": envelope_violations,
        "circuit_count": len(rows),
        "shot_count": sum(row["shots"] for row in rows),
        "records": records,
        "counts": counts,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"dry run: {out_path}")
    print(f"circuits: {len(rows)} (unique transpiled: {len(unique)})")
    print(f"shots: {payload['shot_count']}")
    depths = {r["circuit_name"]: r["transpiled_depth"] for r in records}
    for name, depth in sorted(depths.items()):
        print(f"  {name}: transpiled depth {depth} (envelopes {campaign['envelope']})")
    if envelope_violations:
        print(f"DEPTH ENVELOPE VIOLATIONS: {envelope_violations}", file=sys.stderr)
        return 1
    print("all circuits inside the depth envelope")
    return 0


def _load_credentials() -> tuple[str, str]:
    """Read the Resonance URL and token from the vault (never printed)."""
    in_section = False
    url = token = None
    for raw in VAULT_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## IQM Resonance"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section:
            continue
        if line.lower().startswith("- url:"):
            url = line.split(":", 1)[1].strip()
        if line.lower().startswith("- token:"):
            token = line.split(":", 1)[1].strip()
    if not url or not token:
        raise RuntimeError("missing IQM Resonance URL or token in vault")
    return url, token


def _live_backend(quantum_computer: str) -> _IQMBackend:
    """Return a live Resonance backend (token vault-only)."""
    module = importlib.import_module("iqm.qiskit_iqm.iqm_provider")
    provider_class = module.IQMProvider
    url, token = _load_credentials()
    provider = provider_class(url, quantum_computer=quantum_computer, token=token)
    return cast(_IQMBackend, provider.get_backend())


def _build_submission_groups(
    backend: _IQMBackend,
    prepared: dict[int, list[tuple[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Build provider requests and their exact pre-submit payload digests."""
    groups: list[dict[str, Any]] = []
    run_requests: dict[str, object] = {}
    for shots, group in prepared.items():
        if not group:
            continue
        group_id = "main" if shots == MAIN_SHOTS else "readout"
        circuits = [circuit for _, circuit in group]
        run_request = backend.create_run_request(circuits, shots=shots)
        raw_circuits = getattr(run_request, "circuits", None)
        if not isinstance(raw_circuits, list):
            raise ValueError(f"IQM run request for {group_id!r} has no circuit list")
        groups.append(
            {
                "group_id": group_id,
                "state": "prepared",
                "job_id": None,
                "shots": shots,
                "labels": [label for label, _ in group],
                "circuit_names": [str(getattr(circuit, "name", "")) for circuit in circuits],
                "payload_sha256": _payload_digest(cast(list[object], raw_circuits), run_request),
            }
        )
        run_requests[group_id] = run_request
    if not groups:
        raise ValueError("no IQM submission groups were prepared")
    return groups, run_requests


def _validate_group_plan(
    record: dict[str, Any],
    expected_groups: list[dict[str, Any]],
) -> None:
    """Prove a partial journal still describes the newly prepared frozen payload."""
    actual = {entry["group_id"]: entry for entry in _journal_jobs(record)}
    expected = {entry["group_id"]: entry for entry in expected_groups}
    if actual.keys() != expected.keys():
        raise ValueError("submission journal job groups do not match the prepared matrix")
    compared_fields = ("shots", "labels", "circuit_names", "payload_sha256")
    for group_id, expected_entry in expected.items():
        actual_entry = actual[group_id]
        for field in compared_fields:
            if actual_entry.get(field) != expected_entry[field]:
                raise ValueError(
                    f"submission journal group {group_id!r} {field!r} does not match "
                    "the freshly prepared provider payload"
                )


def _journal_status(jobs: list[dict[str, Any]]) -> str:
    """Summarise group states without hiding an ambiguous provider call."""
    states = {str(entry["state"]) for entry in jobs}
    if states == {"submitted"}:
        return "submitted"
    if states & {"submitting", "recovery_required"}:
        return "recovery_required"
    if "submitted" in states:
        return "partially_submitted"
    return "prepared"


def _new_journal(
    *,
    args: argparse.Namespace,
    campaign_id: str,
    layout: tuple[int, int, int, int],
    depths: dict[str, int],
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create the complete write-ahead record before the first provider call."""
    now = _utc_now()
    return {
        "schema": JOURNAL_SCHEMA,
        "status": "prepared",
        "campaign": campaign_id,
        "quantum_computer": args.quantum_computer,
        "date": args.date,
        "repetition": args.repetition,
        "window": args.window,
        "layout": list(layout),
        "layout_choice": args.layout,
        "transpiled_depths": depths,
        "created_at": now,
        "updated_at": now,
        "jobs": groups,
    }


def submit(args: argparse.Namespace) -> int:
    """Submit one frozen block through a fail-closed durable journal."""
    if not args.i_have_owner_go:
        print(
            "REFUSED: QPU submission requires the per-submit owner GO "
            "(--i-have-owner-go). See the preregistration submission boundary.",
            file=sys.stderr,
        )
        return 2

    if args.campaign == "window-variability" and not 1 <= args.window <= 10:
        print(
            "REFUSED: window-variability submissions require --window in the frozen range 1..10",
            file=sys.stderr,
        )
        return 2

    from qiskit import transpile

    helper = _load_helper()
    campaign = CAMPAIGNS[args.campaign]
    layout = PRIMARY_LAYOUT if args.layout == "primary" else FALLBACK_LAYOUT
    out_path = Path(args.out)
    with _journal_lock(out_path):
        record: dict[str, Any] | None = None
        if out_path.exists():
            try:
                record = _load_json_object(out_path)
                _validate_journal_header(
                    record,
                    args=args,
                    campaign_id=str(campaign["campaign_id"]),
                    layout=layout,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"REFUSED: existing submission journal is invalid: {exc}", file=sys.stderr)
                return RECOVERY_REQUIRED_EXIT
            existing_jobs = _journal_jobs(record)
            ambiguous = [
                entry["group_id"]
                for entry in existing_jobs
                if entry["state"] in {"submitting", "recovery_required"}
            ]
            if ambiguous:
                print(
                    "REFUSED: provider acceptance is ambiguous for journal group(s) "
                    f"{ambiguous}; inspect the IQM dashboard and bind each exact job with "
                    "the recover subcommand. No provider submission was attempted.",
                    file=sys.stderr,
                )
                return RECOVERY_REQUIRED_EXIT
            if all(entry["state"] == "submitted" for entry in existing_jobs):
                print(
                    f"already submitted; reusing durable journal without provider contact: {out_path}"
                )
                return 0

        backend = _live_backend(args.quantum_computer)
        all_rows = build_powered_plan(
            layout=layout,
            depths=campaign["depths"],
            repetitions=int(campaign.get("repetitions", REPETITIONS)),
        )
        if campaign.get("batch_all"):
            # Frozen batching disclosure: the whole matrix goes in one pass —
            # mains batch into one job, readout states into a second.
            rows = all_rows
        else:
            # Readout calibration states run ONCE (with repetition 1); later
            # blocks are mains-only so the matrix stays the preregistered count.
            wanted = {args.repetition} | ({0} if args.repetition == 1 else set())
            rows = [row for row in all_rows if row["repetition"] in wanted]

        prepared: dict[int, list[tuple[str, Any]]] = {MAIN_SHOTS: [], READOUT_SHOTS: []}
        depths: dict[str, int] = {}
        for row in rows:
            circuit = helper._build_circuit(
                {"circuit_name": row["circuit_name"], "meta": row["meta"]}
            )
            circuit.name = row["circuit_name"]
            isa = transpile(
                circuit, backend=backend, initial_layout=list(layout), optimization_level=1
            )
            depth = int(isa.depth())
            depths[row["label"]] = depth
            bound = (
                int(campaign["envelope"][int(row["meta"].get("depth", 0))])
                if row["kind"] == "dla_parity"
                else max(campaign["envelope"].values())
            )
            if depth > bound:
                print(
                    f"DEPTH ENVELOPE VIOLATION at submit: {row['label']} {depth} > "
                    f"{bound} — refusing to submit",
                    file=sys.stderr,
                )
                return 1
            prepared[int(row["shots"])].append((row["label"], isa))

        expected_groups, run_requests = _build_submission_groups(backend, prepared)
        if record is None:
            record = _new_journal(
                args=args,
                campaign_id=str(campaign["campaign_id"]),
                layout=layout,
                depths=depths,
                groups=expected_groups,
            )
            _atomic_write_json(out_path, record)
        else:
            try:
                _validate_group_plan(record, expected_groups)
            except ValueError as exc:
                print(f"REFUSED: prepared payload does not match journal: {exc}", file=sys.stderr)
                return RECOVERY_REQUIRED_EXIT

        jobs = _journal_jobs(record)
        by_group = {entry["group_id"]: entry for entry in jobs}
        for expected in expected_groups:
            group_id = str(expected["group_id"])
            entry = by_group[group_id]
            if entry["state"] == "submitted":
                print(f"reusing submitted {group_id} job {entry['job_id']}")
                continue
            entry["state"] = "submitting"
            entry["submission_started_at"] = _utc_now()
            record["status"] = "recovery_required"
            record["updated_at"] = _utc_now()
            _atomic_write_json(out_path, record)
            try:
                provider_job = backend.client.submit_run_request(run_requests[group_id])
                job_id = _job_identifier(provider_job)
            except Exception as exc:
                entry["state"] = "recovery_required"
                entry["last_error_type"] = type(exc).__name__
                entry["last_error"] = str(exc)
                record["status"] = "recovery_required"
                record["updated_at"] = _utc_now()
                _atomic_write_json(out_path, record)
                print(
                    f"RECOVERY REQUIRED: provider call for {group_id!r} did not return a durable "
                    f"job ID ({type(exc).__name__}: {exc}). Do not resubmit. Inspect IQM and "
                    "bind the exact job with the recover subcommand.",
                    file=sys.stderr,
                )
                return RECOVERY_REQUIRED_EXIT
            if any(other.get("job_id") == job_id for other in jobs if other is not entry):
                entry["state"] = "recovery_required"
                record["status"] = "recovery_required"
                record["updated_at"] = _utc_now()
                _atomic_write_json(out_path, record)
                print(
                    f"RECOVERY REQUIRED: provider reused job ID {job_id!r} across groups",
                    file=sys.stderr,
                )
                return RECOVERY_REQUIRED_EXIT
            entry["job_id"] = job_id
            entry["state"] = "submitted"
            entry["submitted_at"] = _utc_now()
            entry.pop("last_error", None)
            entry.pop("last_error_type", None)
            record["status"] = _journal_status(jobs)
            record["updated_at"] = _utc_now()
            _atomic_write_json(out_path, record)
            print(
                f"submitted {len(entry['labels'])} circuits @ {entry['shots']} shots -> job {job_id}"
            )

        record["status"] = _journal_status(jobs)
        record["updated_at"] = _utc_now()
        _atomic_write_json(out_path, record)
        print(f"submission journal: {out_path}")
        return 0


def recover(args: argparse.Namespace) -> int:
    """Bind one dashboard-verified job to an ambiguous journal group."""
    if not args.i_confirm_provider_job:
        print(
            "REFUSED: recovery requires --i-confirm-provider-job after exact dashboard verification",
            file=sys.stderr,
        )
        return 2
    record_path = Path(args.record)
    with _journal_lock(record_path):
        try:
            record = _load_json_object(record_path)
            if record.get("schema") != JOURNAL_SCHEMA:
                raise ValueError(f"record is not a {JOURNAL_SCHEMA} journal")
            jobs = _journal_jobs(record)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"REFUSED: invalid recovery journal: {exc}", file=sys.stderr)
            return RECOVERY_REQUIRED_EXIT
        matches = [entry for entry in jobs if entry["group_id"] == args.group]
        if len(matches) != 1:
            print(f"REFUSED: journal has no unique group {args.group!r}", file=sys.stderr)
            return RECOVERY_REQUIRED_EXIT
        entry = matches[0]
        if entry["state"] == "submitted":
            if entry["job_id"] == args.job_id:
                print(f"journal group {args.group!r} is already bound to job {args.job_id}")
                return 0
            print(
                f"REFUSED: journal group {args.group!r} is already bound to job {entry['job_id']}",
                file=sys.stderr,
            )
            return RECOVERY_REQUIRED_EXIT
        if entry["state"] not in {"submitting", "recovery_required"}:
            print(
                f"REFUSED: group {args.group!r} is {entry['state']!r}, not an ambiguous call",
                file=sys.stderr,
            )
            return RECOVERY_REQUIRED_EXIT
        if any(other.get("job_id") == args.job_id for other in jobs if other is not entry):
            print(
                f"REFUSED: job {args.job_id!r} is already bound to another group", file=sys.stderr
            )
            return RECOVERY_REQUIRED_EXIT
        try:
            provider_uuid = UUID(args.job_id)
        except ValueError:
            print(f"REFUSED: {args.job_id!r} is not an IQM UUID", file=sys.stderr)
            return RECOVERY_REQUIRED_EXIT

        backend = _live_backend(str(record["quantum_computer"]))
        try:
            provider_job = backend.client.get_job(provider_uuid)
            circuits, parameters = provider_job.payload()
            observed_digest = _payload_digest(circuits, parameters)
        except Exception as exc:
            print(
                f"REFUSED: could not retrieve and verify IQM job {args.job_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return RECOVERY_REQUIRED_EXIT
        if observed_digest != entry.get("payload_sha256"):
            print(
                f"REFUSED: IQM job {args.job_id} payload digest {observed_digest} does not "
                f"match journal group {args.group!r}",
                file=sys.stderr,
            )
            return RECOVERY_REQUIRED_EXIT

        entry["job_id"] = args.job_id
        entry["state"] = "submitted"
        entry["recovered_at"] = _utc_now()
        entry["recovery_method"] = "provider_payload_digest_match"
        entry.pop("last_error", None)
        entry.pop("last_error_type", None)
        record["status"] = _journal_status(jobs)
        record["updated_at"] = _utc_now()
        _atomic_write_json(record_path, record)
        print(
            f"recovered journal group {args.group!r} -> job {args.job_id}; "
            f"journal status {record['status']}"
        )
        return 0


def retrieve(args: argparse.Namespace) -> int:
    """Poll the submitted jobs and write the counts JSON."""
    import time

    record = _load_json_object(Path(args.record))
    if record.get("schema") == JOURNAL_SCHEMA:
        jobs = _journal_jobs(record)
        if _journal_status(jobs) != "submitted":
            print(
                "REFUSED: submission journal is incomplete or recovery-required; "
                "no retrieval was attempted",
                file=sys.stderr,
            )
            return RECOVERY_REQUIRED_EXIT
    else:
        raw_jobs = record.get("jobs")
        if not isinstance(raw_jobs, list) or not raw_jobs:
            print("REFUSED: legacy submission record has no jobs", file=sys.stderr)
            return RECOVERY_REQUIRED_EXIT
        jobs = cast(list[dict[str, Any]], raw_jobs)
    backend = _live_backend(record["quantum_computer"])

    counts: dict[str, dict[str, int]] = {}
    for entry in jobs:
        job = backend.retrieve_job(entry["job_id"])
        deadline = time.monotonic() + float(args.timeout_minutes) * 60.0
        while not job.done():
            if time.monotonic() > deadline:
                print(f"job {entry['job_id']} not finished within timeout", file=sys.stderr)
                return 3
            print(f"job {entry['job_id']}: {job.status()} — waiting")
            time.sleep(float(args.poll_seconds))
        all_counts = job.result().get_counts()
        if not isinstance(all_counts, list):
            all_counts = [all_counts]
        for label, circuit_counts in zip(entry["labels"], all_counts, strict=True):
            counts[label] = {str(k): int(v) for k, v in circuit_counts.items()}
            print(f"{label}: {sum(counts[label].values())} shots retrieved")

    payload = {
        "campaign": record["campaign"],
        "backend": record["quantum_computer"],
        "date": record["date"],
        "repetition": record["repetition"],
        "window": record.get("window", 0),
        "layout": record["layout"],
        "job_ids": [entry["job_id"] for entry in jobs],
        "counts": counts,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"counts: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the subcommand and run it, returning the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dry = sub.add_parser("dry-run", help="IQMFakeGarnet full-matrix readiness dry run")
    dry.add_argument("--campaign", choices=tuple(CAMPAIGNS), default="powered")
    dry.add_argument("--layout", choices=("primary", "fallback"), default="primary")
    dry.add_argument("--date", required=True, help="artefact date stamp (YYYY-MM-DD)")
    dry.add_argument("--out", required=True, help="output dry-run JSON")
    dry.set_defaults(func=dry_run)

    sub_submit = sub.add_parser("submit", help="submit one repetition block (owner-gated)")
    sub_submit.add_argument("--campaign", choices=tuple(CAMPAIGNS), default="powered")
    sub_submit.add_argument("--quantum-computer", default="garnet:mock")
    sub_submit.add_argument("--layout", choices=("primary", "fallback"), default="primary")
    sub_submit.add_argument("--repetition", type=int, default=1, choices=(1, 2, 3, 4))
    sub_submit.add_argument(
        "--window",
        type=int,
        default=0,
        help="window index for the window-variability campaign (stamped into the record)",
    )
    sub_submit.add_argument("--date", required=True, help="artefact date stamp (YYYY-MM-DD)")
    sub_submit.add_argument("--out", required=True, help="submission record JSON")
    sub_submit.add_argument(
        "--i-have-owner-go",
        action="store_true",
        help="assert the explicit per-submit owner GO exists for this block",
    )
    sub_submit.set_defaults(func=submit)

    sub_recover = sub.add_parser(
        "recover",
        help="bind a dashboard-verified IQM job to an ambiguous journal group",
    )
    sub_recover.add_argument("--record", required=True, help="submission journal JSON")
    sub_recover.add_argument("--group", required=True, choices=("main", "readout"))
    sub_recover.add_argument("--job-id", required=True, help="exact provider job UUID")
    sub_recover.add_argument(
        "--i-confirm-provider-job",
        action="store_true",
        help="confirm the job was identified on the provider dashboard",
    )
    sub_recover.set_defaults(func=recover)

    sub_retrieve = sub.add_parser("retrieve", help="poll jobs and write counts JSON")
    sub_retrieve.add_argument("--record", required=True, help="submission record JSON")
    sub_retrieve.add_argument("--out", required=True, help="output counts JSON")
    sub_retrieve.add_argument("--poll-seconds", default=20.0, type=float)
    sub_retrieve.add_argument("--timeout-minutes", default=60.0, type=float)
    sub_retrieve.set_defaults(func=retrieve)

    args = parser.parse_args(argv)
    result = args.func(args)
    assert isinstance(result, int)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
