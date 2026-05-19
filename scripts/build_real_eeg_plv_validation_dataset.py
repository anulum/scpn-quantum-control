# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Real EEG PLV K_nm Validation Dataset Builder
"""Build a measured EEG alpha-band PLV coupling dataset for K_nm validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import requests
from scipy.signal import butter, hilbert, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / ".coordination" / "datasets" / "eegmmidb" / "S001"
DEFAULT_EDF = DEFAULT_RAW_DIR / "S001R01.edf"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "knm_physical_validation" / "measured_couplings.json"
DEFAULT_SOURCE_URL = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf"
DEFAULT_RECORDS = ["S001R01"]
DEFAULT_DATASET_BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"
DEFAULT_CHANNELS = ["Fp1.", "Fp2.", "F3..", "F4..", "C3..", "C4..", "O1..", "O2.."]
CANONICAL_LABELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]
MAX_DOWNLOAD_REDIRECTS = 3
EEGMMIDB_SUBJECTS = tuple(range(1, 110))
EEGMMIDB_RUN_CONDITIONS = {
    "01": "baseline eyes open",
    "02": "baseline eyes closed",
}
RECORD_PRESETS = {
    "baseline-open-109": tuple(f"S{subject:03d}R01" for subject in EEGMMIDB_SUBJECTS),
    "baseline-closed-109": tuple(f"S{subject:03d}R02" for subject in EEGMMIDB_SUBJECTS),
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a local raw EEG file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_https_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Only absolute HTTPS EDF URLs are allowed: {url!r}")
    return url


def download_https_file(source_url: str, path: Path, *, redirects_remaining: int) -> None:
    """Download an EDF file over HTTPS into place using an atomic partial file."""
    if redirects_remaining < 0:
        raise RuntimeError(f"Could not download EDF from {source_url}: redirect limit exceeded")
    validated_url = _validated_https_url(source_url)
    try:
        with requests.get(
            validated_url,
            headers={"User-Agent": "scpn-quantum-control-eeg-validation/1.0"},
            stream=True,
            timeout=60,
        ) as response:
            _validated_https_url(response.url)
            response.raise_for_status()
            partial_path = path.with_suffix(f"{path.suffix}.part")
            with partial_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            partial_path.replace(path)
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not download EDF from {source_url}: {exc}") from exc


def ensure_edf(path: Path, *, source_url: str, download: bool) -> None:
    """Ensure an EDF file exists locally, downloading it only when requested."""
    if path.exists():
        return
    if not download:
        raise FileNotFoundError(
            f"Missing EDF file: {path}. Re-run with --download or provide --edf."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    download_https_file(source_url, path, redirects_remaining=MAX_DOWNLOAD_REDIRECTS)


def record_to_path(record: str, *, raw_root: Path) -> Path:
    """Map an EEGMMIDB record identifier to its local raw EDF path."""
    subject = record[:4]
    return raw_root / subject / f"{record}.edf"


def record_to_url(record: str, *, dataset_base_url: str) -> str:
    """Map an EEGMMIDB record identifier to its PhysioNet HTTPS URL."""
    subject = record[:4]
    return f"{dataset_base_url.rstrip('/')}/{subject}/{record}.edf"


def records_for_subject_run(subjects: Sequence[int], *, run: int) -> list[str]:
    """Return validated EEGMMIDB record identifiers for one run and subject set."""
    if run < 1 or run > 14:
        raise ValueError(f"EEGMMIDB run must be in [1, 14], got {run}")
    invalid_subjects = [subject for subject in subjects if subject not in EEGMMIDB_SUBJECTS]
    if invalid_subjects:
        raise ValueError(f"EEGMMIDB subjects must be in [1, 109], got {invalid_subjects}")
    return [f"S{subject:03d}R{run:02d}" for subject in subjects]


def record_condition(records: Sequence[str]) -> str:
    """Return the shared EEGMMIDB condition label for a record cohort."""
    run_codes = {record[-2:] for record in records}
    if len(run_codes) != 1:
        return "mixed EEGMMIDB runs"
    return EEGMMIDB_RUN_CONDITIONS.get(next(iter(run_codes)), f"run {next(iter(run_codes))}")


def load_edf_channels(path: Path, channels: list[str]) -> tuple[np.ndarray, float]:
    """Load selected EDF channels and return channel data plus sampling rate."""
    import mne

    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    missing = sorted(set(channels) - set(raw.ch_names))
    if missing:
        raise ValueError(f"EDF file is missing expected channels: {missing}")
    raw.pick(channels)
    return raw.get_data(), float(raw.info["sfreq"])


def bandpass_phase(data: np.ndarray, *, sfreq: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Return analytic-signal phase after zero-phase bandpass filtering."""
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, data, axis=1)
    return np.angle(hilbert(filtered, axis=1))


def segment_starts(n_samples: int, *, sfreq: float, window_s: float, step_s: float) -> list[int]:
    """Return deterministic segment starts for the PLV sliding-window estimator."""
    window = int(round(window_s * sfreq))
    step = int(round(step_s * sfreq))
    if window <= 0 or step <= 0:
        raise ValueError("Window and step must be positive")
    if n_samples < window:
        raise ValueError("Signal is shorter than one PLV window")
    return list(range(0, n_samples - window + 1, step))


def plv_edges(
    phase: np.ndarray,
    *,
    sfreq: float,
    window_s: float,
    step_s: float,
) -> list[dict[str, Any]]:
    """Estimate all pairwise PLV edges from channel phase traces."""
    starts = segment_starts(phase.shape[1], sfreq=sfreq, window_s=window_s, step_s=step_s)
    window = int(round(window_s * sfreq))
    edges = []
    for i in range(phase.shape[0]):
        for j in range(i + 1, phase.shape[0]):
            segment_values = []
            for start in starts:
                delta = phase[i, start : start + window] - phase[j, start : start + window]
                segment_values.append(float(abs(np.mean(np.exp(1j * delta)))))
            values = np.asarray(segment_values, dtype=np.float64)
            uncertainty = float(np.std(values, ddof=1) / np.sqrt(values.size))
            edges.append(
                {
                    "i": i + 1,
                    "j": j + 1,
                    "value": float(np.mean(values)),
                    "uncertainty": uncertainty,
                    "uncertainty_type": "standard_error_across_overlapping_plv_windows",
                    "n_segments": int(values.size),
                    "source": "PhysioNet EEGMMIDB S001R01 alpha-band PLV",
                }
            )
    return edges


def build_record_edges(
    *,
    edf_path: Path,
    channels: list[str],
    low_hz: float,
    high_hz: float,
    window_s: float,
    step_s: float,
) -> tuple[list[dict[str, Any]], float]:
    """Build per-edge PLV records for one EDF file."""
    data, sfreq = load_edf_channels(edf_path, channels)
    phase = bandpass_phase(data, sfreq=sfreq, low_hz=low_hz, high_hz=high_hz)
    return plv_edges(phase, sfreq=sfreq, window_s=window_s, step_s=step_s), sfreq


def aggregate_record_edges(
    record_edges: list[dict[str, Any]], *, source_label: str | None = None
) -> list[dict[str, Any]]:
    """Aggregate per-record PLV edges into cohort-median coupling rows."""
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in record_edges:
        grouped.setdefault((int(item["i"]), int(item["j"])), []).append(item)

    aggregated = []
    for (i_1, j_1), rows in sorted(grouped.items()):
        values = np.asarray([float(row["value"]) for row in rows], dtype=np.float64)
        segment_counts = np.asarray([int(row["n_segments"]) for row in rows], dtype=np.int64)
        record_count = int(values.size)
        median = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        uncertainty = (
            float(1.4826 * np.median(np.abs(values - median)) / np.sqrt(record_count))
            if record_count > 1
            else float(rows[0]["uncertainty"])
        )
        aggregated.append(
            {
                "i": i_1,
                "j": j_1,
                "value": median,
                "uncertainty": uncertainty,
                "uncertainty_type": "median_absolute_deviation_standard_error_across_records",
                "n_records": record_count,
                "n_segments_total": int(np.sum(segment_counts)),
                "q25": float(q25),
                "q75": float(q75),
                "source": source_label or "PhysioNet EEGMMIDB alpha-band PLV cohort median",
            }
        )
    return aggregated


def build_payload(
    *,
    edf_path: Path,
    source_url: str,
    channels: list[str],
    low_hz: float,
    high_hz: float,
    window_s: float,
    step_s: float,
    command: list[str],
) -> dict[str, Any]:
    """Build the single-record measured EEG PLV coupling artefact payload."""
    data, sfreq = load_edf_channels(edf_path, channels)
    phase = bandpass_phase(data, sfreq=sfreq, low_hz=low_hz, high_hz=high_hz)
    couplings = plv_edges(phase, sfreq=sfreq, window_s=window_s, step_s=step_s)
    return {
        "schema_version": "scpn-quantum-control.measured-couplings.v1",
        "system": "PhysioNet EEGMMIDB subject 1 run 1 baseline eyes open",
        "unit": "phase_locking_value",
        "normalisation": (
            "PLV = |mean(exp(1j * phase_difference))| after 8-13 Hz Butterworth "
            "bandpass; dimensionless and locked to [0, 1]."
        ),
        "normalisation_locked": True,
        "source_dataset": {
            "name": "EEG Motor Movement/Imagery Dataset v1.0.0",
            "record": "S001R01.edf",
            "source_url": source_url,
            "licence": "Open Data Commons Attribution License v1.0",
            "citation": (
                "Schalk et al., IEEE Transactions on Biomedical Engineering 51(6):1034-1043, "
                "2004; Goldberger et al., Circulation 101(23):e215-e220, 2000."
            ),
            "raw_sha256": sha256_file(edf_path),
        },
        "signal_processing": {
            "channels_edf": channels,
            "channels_canonical": CANONICAL_LABELS[: len(channels)],
            "sampling_hz": sfreq,
            "bandpass_hz": [low_hz, high_hz],
            "window_s": window_s,
            "step_s": step_s,
            "estimator": "segment mean PLV with standard error across windows",
        },
        "couplings": couplings,
        "provenance": {
            "repo_root": str(REPO_ROOT),
            "git_commit": _git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "command": command,
        },
    }


def build_cohort_payload(
    *,
    records: list[str],
    raw_root: Path,
    dataset_base_url: str,
    channels: list[str],
    low_hz: float,
    high_hz: float,
    window_s: float,
    step_s: float,
    download: bool,
    command: list[str],
) -> dict[str, Any]:
    """Build the cohort-level measured EEG PLV coupling artefact payload."""
    all_edges = []
    record_metadata = []
    sampling_rates = set()
    condition = record_condition(records)
    cohort_source = f"PhysioNet EEGMMIDB {condition} alpha-band PLV cohort median"
    for record in records:
        edf_path = record_to_path(record, raw_root=raw_root)
        source_url = record_to_url(record, dataset_base_url=dataset_base_url)
        ensure_edf(edf_path, source_url=source_url, download=download)
        edges, sfreq = build_record_edges(
            edf_path=edf_path,
            channels=channels,
            low_hz=low_hz,
            high_hz=high_hz,
            window_s=window_s,
            step_s=step_s,
        )
        sampling_rates.add(float(sfreq))
        raw_hash = sha256_file(edf_path)
        for edge in edges:
            edge["record"] = record
            edge["source"] = f"PhysioNet EEGMMIDB {record} alpha-band PLV"
        all_edges.extend(edges)
        record_metadata.append(
            {
                "record": record,
                "source_url": source_url,
                "raw_sha256": raw_hash,
            }
        )

    if len(sampling_rates) != 1:
        raise ValueError(
            f"Expected one sampling rate across records, got {sorted(sampling_rates)}"
        )
    return {
        "schema_version": "scpn-quantum-control.measured-couplings.v1",
        "system": f"PhysioNet EEGMMIDB {condition} cohort median",
        "unit": "phase_locking_value",
        "normalisation": (
            "Per-record PLV = |mean(exp(1j * phase_difference))| after 8-13 Hz "
            "Butterworth bandpass; cohort value is the per-edge median across records."
        ),
        "normalisation_locked": True,
        "source_dataset": {
            "name": "EEG Motor Movement/Imagery Dataset v1.0.0",
            "records": records,
            "condition": condition,
            "n_records": len(records),
            "dataset_base_url": dataset_base_url,
            "licence": "Open Data Commons Attribution License v1.0",
            "citation": (
                "Schalk et al., IEEE Transactions on Biomedical Engineering 51(6):1034-1043, "
                "2004; Goldberger et al., Circulation 101(23):e215-e220, 2000."
            ),
            "raw_records": record_metadata,
        },
        "signal_processing": {
            "channels_edf": channels,
            "channels_canonical": CANONICAL_LABELS[: len(channels)],
            "sampling_hz": sorted(sampling_rates)[0],
            "bandpass_hz": [low_hz, high_hz],
            "window_s": window_s,
            "step_s": step_s,
            "per_record_estimator": "segment mean PLV with standard error across windows",
            "cohort_estimator": "per-edge median with MAD standard error across records",
        },
        "couplings": aggregate_record_edges(all_edges, source_label=cohort_source),
        "per_record_edges": all_edges,
        "provenance": {
            "repo_root": str(REPO_ROOT),
            "git_commit": _git_commit(),
            "python": sys.version,
            "platform": platform.platform(),
            "command": command,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the measured EEG PLV dataset builder CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", type=Path, default=DEFAULT_EDF)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--dataset-base-url", default=DEFAULT_DATASET_BASE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--record", action="append", dest="records")
    parser.add_argument("--record-preset", choices=sorted(RECORD_PRESETS))
    parser.add_argument("--subject-start", type=int, default=1)
    parser.add_argument("--subject-stop", type=int, default=109)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument(
        "--raw-root", type=Path, default=REPO_ROOT / ".coordination" / "datasets" / "eegmmidb"
    )
    parser.add_argument("--single-record", action="store_true")
    parser.add_argument("--low-hz", type=float, default=8.0)
    parser.add_argument("--high-hz", type=float, default=13.0)
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--step-s", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write a single-record or cohort measured EEG PLV coupling artefact."""
    args = parse_args(argv)
    command = [Path(sys.executable).name, *sys.argv]
    if args.single_record:
        ensure_edf(args.edf, source_url=args.source_url, download=bool(args.download))
        payload = build_payload(
            edf_path=args.edf,
            source_url=str(args.source_url),
            channels=DEFAULT_CHANNELS,
            low_hz=float(args.low_hz),
            high_hz=float(args.high_hz),
            window_s=float(args.window_s),
            step_s=float(args.step_s),
            command=command,
        )
    else:
        if args.records:
            records = args.records
        elif args.record_preset:
            records = list(RECORD_PRESETS[args.record_preset])
        elif args.subject_start != 1 or args.subject_stop != 109 or args.run != 1:
            records = records_for_subject_run(
                range(args.subject_start, args.subject_stop + 1),
                run=int(args.run),
            )
        else:
            records = DEFAULT_RECORDS
        payload = build_cohort_payload(
            records=records,
            raw_root=args.raw_root,
            dataset_base_url=str(args.dataset_base_url),
            channels=DEFAULT_CHANNELS,
            low_hz=float(args.low_hz),
            high_hz=float(args.high_hz),
            window_s=float(args.window_s),
            step_s=float(args.step_s),
            download=bool(args.download),
            command=command,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote measured EEG PLV couplings: {args.output}")
    print(f"Edges: {len(payload['couplings'])}")
    if "raw_sha256" in payload["source_dataset"]:
        print(f"Raw SHA-256: {payload['source_dataset']['raw_sha256']}")
    else:
        print(f"Records: {len(payload['source_dataset']['records'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
