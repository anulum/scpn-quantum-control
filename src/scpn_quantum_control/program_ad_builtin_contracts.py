# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — shared builtin primitive registration
"""Initialize builtin primitive families without importing a public facade."""

from .program_ad_array_indexing import _register_program_ad_array_primitive_contracts
from .program_ad_assembly_primitives import _register_program_ad_assembly_primitive_contracts
from .program_ad_cumulative_primitives import _register_program_ad_cumulative_primitive_contracts
from .program_ad_elementwise_primitives import (
    _register_program_ad_elementwise_primitive_contracts,
)
from .program_ad_interpolation_primitives import (
    _register_program_ad_interpolation_primitive_contracts,
)
from .program_ad_linalg_primitives import _register_program_ad_linalg_primitive_contracts
from .program_ad_product_primitives import _register_program_ad_product_primitive_contracts
from .program_ad_reduction_primitives import _register_program_ad_reduction_primitive_contracts
from .program_ad_selection_primitives import _register_program_ad_selection_primitive_contracts
from .program_ad_shape_transforms import _register_program_ad_shape_primitive_contracts
from .program_ad_signal_primitives import _register_program_ad_signal_primitive_contracts
from .program_ad_stencil_primitives import _register_program_ad_stencil_primitive_contracts


def _register_program_ad_builtin_contracts() -> None:
    """Fill missing builtin contracts while preserving existing user bindings."""
    _register_program_ad_array_primitive_contracts()
    _register_program_ad_interpolation_primitive_contracts()
    _register_program_ad_assembly_primitive_contracts()
    _register_program_ad_signal_primitive_contracts()
    _register_program_ad_shape_primitive_contracts()
    _register_program_ad_reduction_primitive_contracts()
    _register_program_ad_stencil_primitive_contracts()
    _register_program_ad_elementwise_primitive_contracts()
    _register_program_ad_selection_primitive_contracts()
    _register_program_ad_product_primitive_contracts()
    _register_program_ad_cumulative_primitive_contracts()
    _register_program_ad_linalg_primitive_contracts()
