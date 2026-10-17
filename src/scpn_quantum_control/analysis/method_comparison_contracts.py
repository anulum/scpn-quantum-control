# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Analysis method comparison public interface
"""Public Analysis records retaining their original v1 defining-module identities.

The original definitions are Analysis-owned per-file sources under the stable
benchmarks namespace. Reexporting their exact objects preserves trusted pickle
lookup, nominal type identity and the original scientific producer.
"""

from ..benchmarks.method_comparison_contracts import (
    MAX_COMPARISON_RUNS,
    BudgetRecord,
    ComparisonBudget,
    ComparisonMeaning,
    IntervalRecord,
    MethodComparisonRecord,
    MethodRun,
    RunRecord,
    SummaryRecord,
)

__all__ = [
    "MAX_COMPARISON_RUNS",
    "ComparisonMeaning",
    "ComparisonBudget",
    "MethodRun",
    "BudgetRecord",
    "IntervalRecord",
    "RunRecord",
    "SummaryRecord",
    "MethodComparisonRecord",
]
