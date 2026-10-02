# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — frozen objective call identities
"""Freeze admitted NumPy identities, output positions and storage classifications."""

from __future__ import annotations

import math
import random
from typing import cast

import numpy as np

_PURE = (
    abs,
    bool,
    dict,
    enumerate,
    float,
    int,
    len,
    list,
    max,
    min,
    range,
    sum,
    tuple,
    zip,
    cast,
    *(vars(math)[name] for name in ("sin", "cos", "exp", "log", "sqrt", "tanh")),
    *(
        vars(np)[name]
        for name in (
            "absolute",
            "add",
            "append",
            "arange",
            "arccos",
            "arcsin",
            "arctan2",
            "argmax",
            "argmin",
            "argsort",
            "array",
            "array_split",
            "atleast_1d",
            "atleast_2d",
            "atleast_3d",
            "block",
            "broadcast_arrays",
            "broadcast_to",
            "choose",
            "clip",
            "column_stack",
            "compress",
            "concatenate",
            "convolve",
            "correlate",
            "cos",
            "cumprod",
            "cumsum",
            "delete",
            "diag",
            "diagflat",
            "diagonal",
            "diff",
            "divide",
            "dot",
            "dsplit",
            "dstack",
            "einsum",
            "exp",
            "expand_dims",
            "expm1",
            "extract",
            "flip",
            "fliplr",
            "flipud",
            "full_like",
            "gradient",
            "hsplit",
            "hstack",
            "inner",
            "insert",
            "interp",
            "log",
            "log1p",
            "matmul",
            "max",
            "maximum",
            "mean",
            "median",
            "min",
            "minimum",
            "moveaxis",
            "multiply",
            "negative",
            "ones",
            "ones_like",
            "outer",
            "pad",
            "percentile",
            "piecewise",
            "power",
            "prod",
            "quantile",
            "ravel",
            "reciprocal",
            "repeat",
            "reshape",
            "roll",
            "rot90",
            "select",
            "sin",
            "sort",
            "split",
            "sqrt",
            "square",
            "squeeze",
            "stack",
            "std",
            "subtract",
            "sum",
            "swapaxes",
            "take",
            "take_along_axis",
            "tan",
            "tanh",
            "tensordot",
            "tile",
            "trace",
            "transpose",
            "trapezoid",
            "tril",
            "triu",
            "var",
            "vdot",
            "vsplit",
            "vstack",
            "where",
            "zeros",
            "zeros_like",
        )
    ),
    *(
        vars(np.linalg)[name]
        for name in (
            "det",
            "eig",
            "eigh",
            "eigvals",
            "eigvalsh",
            "inv",
            "matrix_power",
            "multi_dot",
            "norm",
            "pinv",
            "solve",
            "svd",
        )
    ),
)

_PURE_IDS = frozenset(id(value) for value in _PURE)

_ARRAY_ALLOCATOR = np.array

_NUMPY_OUTPUT_POSITIONS = {
    id(vars(np)[name]): position
    for position, names in (
        (
            2,
            (
                "argmax",
                "argmin",
                "choose",
                "concatenate",
                "dot",
                "max",
                "min",
                "outer",
                "round",
                "stack",
            ),
        ),
        (
            3,
            ("clip", "compress", "cumprod", "cumsum", "mean", "prod", "std", "sum", "take", "var"),
        ),
        (5, ("trace",)),
    )
    for name in names
}

_ARRAY_OUTPUT_POSITIONS = {
    "max": 1,
    "min": 1,
    **{name: 2 for name in ("cumsum", "cumprod", "mean", "prod", "std", "sum", "var")},
}

_UFUNC_AT_METHODS = tuple((value, value.at) for value in _PURE if type(value) is np.ufunc)

_VIEW_IDS = frozenset(
    id(vars(np)[name])
    for name in (
        "reshape",
        "broadcast_arrays",
        "broadcast_to",
        "diagonal",
        "ravel",
        "transpose",
        "split",
        "array_split",
        "hsplit",
        "vsplit",
        "dsplit",
        "atleast_1d",
        "atleast_2d",
        "squeeze",
        "expand_dims",
        "swapaxes",
        "moveaxis",
        "atleast_3d",
        "rot90",
        "flip",
        "flipud",
        "fliplr",
    )
)

_MULTI_VIEW_IDS = frozenset(
    id(vars(np)[name]) for name in ("broadcast_arrays", "atleast_1d", "atleast_2d", "atleast_3d")
)

_RANDOM_MODULES = (random, np.random)

_SCATTER_ADD = np.add

_READ_METHODS = frozenset(
    {
        "copy",
        "sum",
        "mean",
        "prod",
        "cumsum",
        "cumprod",
        "max",
        "min",
        "var",
        "std",
        "reshape",
        "ravel",
        "flatten",
        "item",
        "get",
    }
)

_WRITE_METHODS = frozenset(
    {
        "append",
        "extend",
        "insert",
        "pop",
        "remove",
        "clear",
        "update",
        "sort",
        "reverse",
        "fill",
        "resize",
    }
)
