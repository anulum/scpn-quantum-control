# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- compiler exports
"""Compiler frontends and interchange formats."""

from .mlir import (
    CompilerADTransformPlan,
    DifferentiableMLIRCompileConfig,
    MLIRCompileConfig,
    MLIRModule,
    PrimitiveLoweringStatus,
    build_compiler_ad_transform_plan,
    compile_compiler_ad_transform_plan_to_mlir,
    compile_custom_derivative_rule_to_mlir,
    compile_kuramoto_to_mlir,
    compile_whole_program_ad_trace_to_mlir,
)

__all__ = [
    "CompilerADTransformPlan",
    "DifferentiableMLIRCompileConfig",
    "MLIRCompileConfig",
    "PrimitiveLoweringStatus",
    "MLIRModule",
    "build_compiler_ad_transform_plan",
    "compile_compiler_ad_transform_plan_to_mlir",
    "compile_custom_derivative_rule_to_mlir",
    "compile_whole_program_ad_trace_to_mlir",
    "compile_kuramoto_to_mlir",
]
