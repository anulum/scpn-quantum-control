# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — multimodal-forecasting deterministic multimodal evidence runner
"""Regenerate bounded synthetic multimodal forecasting evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scpn_quantum_control.forecasting.multimodal_bridge import (
    forecast_to_controller_initialisation,
    plan_forecast_active_sensing,
)
from scpn_quantum_control.forecasting.multimodal_forecaster import (
    evaluate_point_forecast,
    fit_multimodal_ridge_forecaster,
)
from scpn_quantum_control.forecasting.multimodal_report import (
    MultimodalForecastingEvidence,
    MultimodalSupportRow,
    render_multimodal_forecasting_markdown,
    write_multimodal_forecasting_evidence,
)
from scpn_quantum_control.forecasting.partial_observation import (
    evaluate_partial_observation_batch,
)
from scpn_quantum_control.forecasting.synthetic_multimodal import (
    SyntheticMultimodalConfig,
    generate_synthetic_multimodal_dataset,
)
from scpn_quantum_control.forecasting.uncertainty import (
    apply_residual_interval,
    certify_interval_coverage,
    fit_residual_interval_calibrator,
)

DEFAULT_JSON = Path("data/multimodal_forecasting/multimodal_forecasting_evidence.json")
DEFAULT_MARKDOWN = Path("data/multimodal_forecasting/multimodal_forecasting_evidence.md")


def _support_rows() -> tuple[MultimodalSupportRow, ...]:
    """Return the exact supported and blocked forecasting surface matrix."""
    return (
        MultimodalSupportRow(
            surface="synthetic_multimodal_schema",
            status="synthetic_supported",
            evidence="Immutable series, graph, event, mask, target, tag, and split custody.",
            boundary="All four tags identify stylised simulator configurations only.",
        ),
        MultimodalSupportRow(
            surface="missingness_aware_ridge",
            status="synthetic_supported",
            evidence="Training-only imputation/scaling and held-out persistence comparison.",
            boundary="Linear reference baseline, not BRITS or a production forecaster.",
        ),
        MultimodalSupportRow(
            surface="partial_observation_objective",
            status="synthetic_supported",
            evidence="Observed wrapped error plus exact Kuramoto forward residual.",
            boundary="Known simulator coupling; no hidden-state or parameter inference.",
        ),
        MultimodalSupportRow(
            surface="split_residual_intervals",
            status="synthetic_supported",
            evidence="Independent calibration rows and empirical held-out test coverage.",
            boundary="Not sequential EnbPI, conditional coverage, or domain transfer.",
        ),
        MultimodalSupportRow(
            surface="active_sensing_bridge",
            status="bounded_supported",
            evidence="Interval-width proxies enter the existing no-submit sensing planner.",
            boundary="Not adaptive hardware sensing or optimal sensor placement.",
        ),
        MultimodalSupportRow(
            surface="codesign_controller_initialisation",
            status="bounded_supported",
            evidence="Terminal forecast creates a clipped existing ControllerProposal.",
            boundary="Proposal remains unapplied and is not a safety decision.",
        ),
        MultimodalSupportRow(
            surface="real_eeg_clinical_data",
            status="blocked_dependency",
            evidence="No governed real EEG or clinical dataset is in custody.",
            boundary="The eeg_like_sim tag provides no clinical or neuroscience validity.",
        ),
        MultimodalSupportRow(
            surface="real_grid_scada_data",
            status="blocked_dependency",
            evidence="No governed grid or SCADA dataset is in custody.",
            boundary="The grid_like_sim tag is not a power-system operational model.",
        ),
        MultimodalSupportRow(
            surface="real_plasma_plant_data",
            status="blocked_dependency",
            evidence="No governed plasma diagnostic or plant dataset is in custody.",
            boundary="The plasma_like_sim tag provides no reactor or plant evidence.",
        ),
        MultimodalSupportRow(
            surface="hardware_qpu_execution",
            status="blocked_dependency",
            evidence="No hardware or provider request is made by this evidence runner.",
            boundary="Local deterministic simulation only; no QPU, provider, or spend.",
        ),
    )


def build_evidence() -> MultimodalForecastingEvidence:
    """Build the complete deterministic multimodal forecasting evidence bundle."""
    dataset = generate_synthetic_multimodal_dataset(
        SyntheticMultimodalConfig(
            train_samples=64,
            calibration_samples=24,
            test_samples=32,
            history_steps=12,
            horizon_steps=4,
            missing_fraction=0.20,
            seed=3701,
        )
    )
    model = fit_multimodal_ridge_forecaster(dataset.train, ridge=10.0)
    calibration_forecast = model.predict(dataset.calibration)
    calibration_accuracy = evaluate_point_forecast(
        calibration_forecast,
        dataset.calibration,
    )
    calibrator = fit_residual_interval_calibrator(
        model,
        calibration_forecast,
        dataset.calibration,
        alpha=0.10,
    )
    test_forecast = model.predict(dataset.test)
    test_accuracy = evaluate_point_forecast(test_forecast, dataset.test)
    interval = apply_residual_interval(calibrator, test_forecast)
    interval_coverage = certify_interval_coverage(model, calibrator, interval, dataset.test)

    partial_mask = np.zeros_like(dataset.test.target_mask)
    partial_mask[:, :, ::2] = True
    partial_observation = evaluate_partial_observation_batch(
        test_forecast,
        dataset.test,
        partial_mask,
    )
    active_sensing = plan_forecast_active_sensing(
        interval,
        dataset.test,
        sample_index=0,
        candidate_nodes=(0, 2),
        noise_variances=(0.04, 0.06),
        policy_id="ci_dry_run_only",
        shots_per_observable=128,
    )
    controller_initialisation = forecast_to_controller_initialisation(
        interval,
        sample_index=0,
        current_parameters=(0.2, -0.1),
        target_order_parameter=0.9,
        gain_scale=0.25,
        max_abs_update=0.05,
    )
    return MultimodalForecastingEvidence(
        dataset=dataset,
        model=model,
        calibration_accuracy=calibration_accuracy,
        test_accuracy=test_accuracy,
        partial_observation=partial_observation,
        calibrator=calibrator,
        interval_coverage=interval_coverage,
        active_sensing=active_sensing,
        controller_initialisation=controller_initialisation,
        support_rows=_support_rows(),
    )


def main() -> int:
    """Write or exact-check the deterministic forecasting evidence files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    evidence = build_evidence()
    if args.check:
        expected_json = json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n"
        expected_markdown = render_multimodal_forecasting_markdown(evidence)
        if not args.json.is_file() or args.json.read_text(encoding="utf-8") != expected_json:
            raise SystemExit(f"stale or missing evidence: {args.json}")
        if (
            not args.markdown.is_file()
            or args.markdown.read_text(encoding="utf-8") != expected_markdown
        ):
            raise SystemExit(f"stale or missing evidence: {args.markdown}")
        print(
            json.dumps(
                {"check": "passed", "content_digest": evidence.to_dict()["content_digest"]},
                sort_keys=True,
            )
        )
        return 0

    json_digest, markdown_digest = write_multimodal_forecasting_evidence(
        evidence,
        json_path=args.json,
        markdown_path=args.markdown,
    )
    print(
        json.dumps(
            {
                "content_digest": evidence.to_dict()["content_digest"],
                "json_sha256": json_digest,
                "markdown_sha256": markdown_digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
