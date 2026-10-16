# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Complete synthetic comparison export example
"""Export six declared synthetic observations for the Studio comparison instrument.

Costs are synthetic arithmetic controls, not measured performance. No quantum
processor, simulator, provider or workspace storage is contacted by this example.
"""

from scpn_quantum_control.benchmarks.method_comparison_contracts import (
    ComparisonBudget,
    ComparisonMeaning,
    MethodRun,
)
from scpn_quantum_control.studio.method_comparison_artifact import export_method_comparison


def main() -> None:
    """Print an original native export with every declared repetition retained."""
    meaning = ComparisonMeaning(
        "synthetic-arithmetic-control",
        "declared scalar",
        "1",
        "synthetic",
        "float64",
        None,
        "none",
        "fixed",
        "none",
        "none",
        "exact",
    )
    budget = ComparisonBudget(2.0, 3.0, 5.0, 128, 1024, 100.0)
    runs = [
        MethodRun(
            f"{method}-{seed}",
            method,
            "synthetic-declaration",
            seed,
            meaning,
            budget,
            "completed",
            value,
        )
        for method, values in [("reference", [-1.0, 1.0, 3.0]), ("candidate", [0.0, 2.0, 4.0])]
        for seed, value in enumerate(values)
    ]
    print(export_method_comparison(runs, baseline_method="reference"))


if __name__ == "__main__":
    main()
