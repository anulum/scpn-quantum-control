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
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / ".coordination" / "datasets" / "eegmmidb" / "S001"
DEFAULT_EDF = DEFAULT_RAW_DIR / "S001R01.edf"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "knm_physical_validation" / "measured_couplings.json"
DEFAULT_SOURCE_URL = "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf"
DEFAULT_CHANNELS = ["Fp1.", "Fp2.", "F3..", "F4..", "C3..", "C4..", "O1..", "O2.."]
CANONICAL_LABELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "O1", "O2"]


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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_edf(path: Path, *, source_url: str, download: bool) -> None:
    if path.exists():
        return
    if not download:
        raise FileNotFoundError(
            f"Missing EDF file: {path}. Re-run with --download or provide --edf."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(source_url, path)


def load_edf_channels(path: Path, channels: list[str]) -> tuple[np.ndarray, float]:
    import mne

    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    missing = sorted(set(channels) - set(raw.ch_names))
    if missing:
        raise ValueError(f"EDF file is missing expected channels: {missing}")
    raw.pick(channels)
    return raw.get_data(), float(raw.info["sfreq"])


def bandpass_phase(data: np.ndarray, *, sfreq: float, low_hz: float, high_hz: float) -> np.ndarray:
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, data, axis=1)
    return np.angle(hilbert(filtered, axis=1))


def segment_starts(n_samples: int, *, sfreq: float, window_s: float, step_s: float) -> list[int]:
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
            "licence": "PhysioNet open-access credentialed-use terms",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", type=Path, default=DEFAULT_EDF)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--low-hz", type=float, default=8.0)
    parser.add_argument("--high-hz", type=float, default=13.0)
    parser.add_argument("--window-s", type=float, default=2.0)
    parser.add_argument("--step-s", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_edf(args.edf, source_url=args.source_url, download=bool(args.download))
    payload = build_payload(
        edf_path=args.edf,
        source_url=str(args.source_url),
        channels=DEFAULT_CHANNELS,
        low_hz=float(args.low_hz),
        high_hz=float(args.high_hz),
        window_s=float(args.window_s),
        step_s=float(args.step_s),
        command=[Path(sys.executable).name, *sys.argv],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote measured EEG PLV couplings: {args.output}")
    print(f"Edges: {len(payload['couplings'])}")
    print(f"Raw SHA-256: {payload['source_dataset']['raw_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
