# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — source-visible function binding tests
"""Exercise helper source and captured bindings through public effect admission."""

from __future__ import annotations

import ast
import builtins
import inspect
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from types import CodeType, FunctionType
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control import TraceADArray, TraceADScalar, whole_program_value_and_grad
from scpn_quantum_control.differentiable import program_adjoint_replay_gradient
from scpn_quantum_control.execution_reservations import active_reserved_bytes
from scpn_quantum_control.program_ad_effect_admission import find_objective_effects


@pytest.mark.parametrize("binding", ["builtin", "shadowed_builtin", "class", "comprehension"])
def test_large_namespace_preserves_nested_effects_and_builtin_lookup(
    tmp_path: Path, binding: str
) -> None:
    """Unused globals cannot alter actual builtin or nested-code storage provenance.

    Parameters
    ----------
    tmp_path
        Owned location for the actual loaded objective and helper definitions.
    binding
        Native builtin, a shadowing effectful helper or nested executable scope.

    """
    definitions = {
        "builtin": "def objective(values):\n    return sum(values)\n",
        "shadowed_builtin": (
            "def sum(values):\n    state[0] = 9.0\n    return values[0]\n\n"
            "def objective(values):\n    return sum(values)\n"
        ),
        "class": (
            "def objective(values):\n"
            "    class Local:\n        state[0] = 9.0\n"
            "    return values[0] ** 2\n"
        ),
        "comprehension": (
            "def objective(values):\n"
            "    outputs = [np.add(1.0, 2.0, out=state) for once in (0,)]\n"
            "    return values[0] ** 2\n"
        ),
    }
    source = "import numpy as np\nstate = np.array([7.0])\n\n" + definitions[binding]
    path = tmp_path / "namespace_binding.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(source, str(path), "exec"), namespace)
    namespace.update({f"unused_{index}": float(index) for index in range(4096 - len(namespace))})
    function = cast(FunctionType, namespace["objective"])
    baseline = active_reserved_bytes()
    findings = find_objective_effects(function, ast.parse(source))
    if binding == "builtin":
        assert findings == ()
        result = whole_program_value_and_grad(function, [2.0, 3.0], trace=False)
        assert result.value == 5.0
        np.testing.assert_array_equal(result.gradient, [1.0, 1.0])
        np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [1.0, 1.0])
    else:
        assert any(finding.semantic == "captured_mutation" for finding in findings)
        with pytest.raises(ValueError, match="captured_mutation.*line="):
            whole_program_value_and_grad(function, [2.0, 3.0], trace=False)
    np.testing.assert_array_equal(namespace["state"], [7.0])
    assert active_reserved_bytes() == baseline


def test_recursive_helper_refuses_before_external_storage_changes() -> None:
    """Recursive source cannot execute before its located unsupported effect is returned."""
    state = np.array([7.0])

    def helper(value: TraceADScalar) -> object:
        result = helper(value)
        state[0] = 9.0
        return result

    def objective(values: TraceADArray) -> object:
        return helper(cast(TraceADScalar, values[0]))

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="recursion or inspection bound.*line="):
        whole_program_value_and_grad(objective, [2.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("replacement", ["callback", "constant"])
@pytest.mark.parametrize("entry_point", ["compiler", "runtime"])
def test_root_loaded_code_cannot_borrow_a_different_source_definition(
    tmp_path: Path, replacement: str, entry_point: str
) -> None:
    """Refuse stale root source before executing substituted code and recover.

    Parameters
    ----------
    tmp_path
        Owned directory containing the actual source of the loaded objective.
    replacement
        Effectful callback or a different numerical constant in the loaded code.
    entry_point
        Public compiler or numerical differentiation entry point.

    """
    from scpn_quantum_control.differentiable import compile_whole_program_frontend

    source = (
        "ledger = []\n"
        "def callback(values):\n"
        "    ledger.append('executed')\n"
        "    return values[0] * 3.0\n\n"
        "def objective(values):\n"
        "    return values[0] ** 2\n"
    )
    path = tmp_path / "root_binding.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(source, str(path), "exec"), namespace)
    function = cast(FunctionType, namespace["objective"])
    original_code = function.__code__
    changed = source.replace(
        "return values[0] ** 2",
        "return callback(values)" if replacement == "callback" else "return values[0] ** 3",
    )
    other: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(changed, str(path), "exec"), other)
    function.__code__ = cast(FunctionType, other["objective"]).__code__
    baseline = active_reserved_bytes()
    if entry_point == "compiler":
        report = compile_whole_program_frontend(function)
        assert not report.frontend_ready
        diagnostic = next(
            item
            for item in report.unsupported_semantic_diagnostics
            if item.semantic == "external_callback"
        )
        assert diagnostic.detail == "external callback source does not match captured function"
        assert diagnostic.line_number == 1
        assert diagnostic.absolute_line_number == original_code.co_firstlineno
    else:
        with pytest.raises(ValueError, match="source does not match captured function.*line="):
            whole_program_value_and_grad(function, [2.0], trace=False)
    assert namespace["ledger"] == []
    assert active_reserved_bytes() == baseline

    function.__code__ = original_code
    assert compile_whole_program_frontend(function).frontend_ready
    result = whole_program_value_and_grad(function, [2.0], trace=False)
    assert result.value == 4.0
    np.testing.assert_array_equal(result.gradient, [4.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [4.0])
    assert namespace["ledger"] == []
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("scope", ["direct", "conditional", "try", "loop"])
def test_root_source_binding_preserves_original_module_import_scopes(
    tmp_path: Path, scope: str
) -> None:
    """Keep numerical roots whose module imports occur in ordinary source scopes.

    Parameters
    ----------
    tmp_path
        Owned location containing the actual imported module and objective.
    scope
        Direct, conditional, exception-handling or loop module import.

    """
    imports = {
        "direct": "import numpy as np\n",
        "conditional": "if True:\n    import numpy as np\n",
        "try": "try:\n    import numpy as np\nexcept ImportError:\n    raise\n",
        "loop": "for once in (0,):\n    import numpy as np\n",
    }
    source = imports[scope] + "\ndef objective(values):\n    return np.sum(values ** 2)\n"
    path = tmp_path / "root_module_import.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(source, str(path), "exec"), namespace)
    function = cast(FunctionType, namespace["objective"])
    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(function, [2.0, 3.0], trace=False)
    assert result.value == 13.0
    np.testing.assert_array_equal(result.gradient, [4.0, 6.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [4.0, 6.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("fault", ["missing", "empty", "module_syntax", "line_mismatch"])
def test_root_source_loss_refuses_before_module_effects_and_recovers(
    tmp_path: Path, fault: str
) -> None:
    """Refuse lost root identity without executing file statements and recover.

    Parameters
    ----------
    tmp_path
        Owned source location for the actual loaded objective.
    fault
        Missing file, empty file, invalid module syntax or changed line metadata.

    """
    source = "calls.append('module')\n\ndef objective(values):\n    return values[0] ** 2\n"
    path = tmp_path / "root_source_loss.py"
    path.write_text(source, encoding="utf-8")
    calls: list[str] = []
    namespace: dict[str, object] = {"__builtins__": vars(builtins), "calls": calls}
    exec(compile(source, str(path), "exec"), namespace)
    assert calls == ["module"]
    calls.clear()
    function = cast(FunctionType, namespace["objective"])
    original_code = function.__code__
    tree = ast.parse(source)
    if fault == "missing":
        path.unlink()
    elif fault == "empty":
        path.write_text("", encoding="utf-8")
    elif fault == "module_syntax":
        path.write_text(source + "\ninvalid module syntax\n", encoding="utf-8")
    else:
        function.__code__ = original_code.replace(co_firstlineno=99)

    baseline = active_reserved_bytes()
    findings = find_objective_effects(function, tree)
    assert len(findings) == 1
    assert findings[0].semantic == "external_callback"
    assert findings[0].detail == (
        "external callback source is unavailable"
        if fault == "missing"
        else "external callback source does not match captured function"
    )
    assert getattr(findings[0].node, "lineno", 0) == 3
    assert calls == []
    assert active_reserved_bytes() == baseline

    function.__code__ = original_code
    path.write_text(source, encoding="utf-8")
    result = whole_program_value_and_grad(function, [3.0], trace=False)
    assert result.value == 9.0
    np.testing.assert_array_equal(result.gradient, [6.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [6.0])
    assert calls == []
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("calls", [63, 64])
def test_helper_inspection_bound_has_a_supported_numeric_neighbor(
    tmp_path: Path, calls: int
) -> None:
    """Actual source calls retain a numerical neighbor below the inspection limit.

    Parameters
    ----------
    tmp_path
        Owned temporary directory for actual compiled objective source.
    calls
        Helper calls below or beyond the root-inclusive inspection bound.

    """
    source = (
        "def helper(values):\n    return values[0]\n\n"
        "def objective(values):\n    return " + " + ".join(["helper(values)"] * calls) + "\n"
    )
    path = tmp_path / "helper_inspection.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins)}
    exec(compile(source, str(path), "exec"), namespace)
    objective = cast(Callable[..., object], namespace["objective"])
    baseline = active_reserved_bytes()
    if calls == 64:
        with pytest.raises(ValueError, match="recursion or inspection bound.*line="):
            whole_program_value_and_grad(objective, [2.0], trace=False)
    else:
        result = whole_program_value_and_grad(objective, [2.0], trace=False)
        assert result.value == 126.0
        np.testing.assert_array_equal(result.gradient, [63.0])
        np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [63.0])
    assert active_reserved_bytes() == baseline


def test_async_helper_refuses_at_the_objective_callsite() -> None:
    """Async helpers cannot create or run a coroutine during numerical capture."""
    state = np.array([7.0])

    async def helper(value: TraceADScalar) -> object:
        state[0] = 9.0
        return value

    def objective(values: TraceADArray) -> object:
        return helper(cast(TraceADScalar, values[0]))

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="asynchronous external callback.*line="):
        whole_program_value_and_grad(objective, [2.0], trace=False)
    np.testing.assert_array_equal(state, [7.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("route", ["positional", "keyword_only"])
@pytest.mark.parametrize("captured", [True, False])
def test_helper_default_buffer_binding_preserves_ownership(route: str, captured: bool) -> None:
    """Defaults keep caller ownership while explicit owned operands remain executable.

    Parameters
    ----------
    route
        Positional or keyword-only helper default binding.
    captured
        Whether the default caller buffer is selected.

    """
    state = np.array([7.0])

    def positional_helper(value: TraceADScalar, output: NDArray[np.float64] = state) -> object:
        np.add(1.0, 2.0, out=output)
        return value * output[0]

    def keyword_helper(value: TraceADScalar, *, output: NDArray[np.float64] = state) -> object:
        np.add(1.0, 2.0, out=output)
        return value * output[0]

    if route == "positional" and captured:

        def objective(values: TraceADArray) -> object:
            return positional_helper(cast(TraceADScalar, values[0]))

    elif route == "positional":

        def objective(values: TraceADArray) -> object:
            return positional_helper(cast(TraceADScalar, values[0]), np.zeros(1))

    elif captured:

        def objective(values: TraceADArray) -> object:
            return keyword_helper(cast(TraceADScalar, values[0]))

    else:

        def objective(values: TraceADArray) -> object:
            return keyword_helper(cast(TraceADScalar, values[0]), output=np.zeros(1))

    baseline = active_reserved_bytes()
    if captured:
        with pytest.raises(ValueError, match="captured_mutation.*line="):
            whole_program_value_and_grad(objective, [2.0], trace=False)
    else:
        result = whole_program_value_and_grad(objective, [2.0], trace=False)
        assert result.value == 6.0
        np.testing.assert_array_equal(result.gradient, [3.0])
        np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    np.testing.assert_array_equal(state, [7.0])
    assert active_reserved_bytes() == baseline


def test_empty_helper_capture_returns_a_located_effect_finding() -> None:
    """A deleted closure binding produces an authored refusal without evaluating it."""
    coefficient = 3.0

    def helper() -> float:
        return coefficient

    def objective(values: TraceADArray) -> object:
        return values[0] * helper()

    closure = cast(FunctionType, helper).__closure__
    assert closure is not None
    del closure[0].cell_contents
    tree = ast.parse(textwrap.dedent(inspect.getsource(objective)))
    baseline = active_reserved_bytes()
    findings = find_objective_effects(objective, tree)
    assert len(findings) == 1
    assert findings[0].semantic == "external_callback"
    assert findings[0].detail == "external callback has an empty captured binding"
    assert getattr(findings[0].node, "lineno", 0) == 2
    assert active_reserved_bytes() == baseline


def test_wrapped_helper_refuses_without_calling_the_wrapper() -> None:
    """A wrapper marker cannot borrow another function's source effect contract."""
    state = np.array([7.0])

    def helper() -> float:
        state[0] = 9.0
        return 3.0

    helper.__dict__["__wrapped__"] = helper

    def objective(values: TraceADArray) -> object:
        return values[0] * helper()

    tree = ast.parse(textwrap.dedent(inspect.getsource(objective)))
    baseline = active_reserved_bytes()
    findings = find_objective_effects(objective, tree)
    assert len(findings) == 1
    assert findings[0].semantic == "external_callback"
    assert findings[0].detail == "external callback wrapper identity is unsupported"
    assert getattr(findings[0].node, "lineno", 0) == 2
    np.testing.assert_array_equal(state, [7.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "route",
    ["globals", "attributes", "keyword_defaults", "oversized", "tuple_subclass", "dict_subclass"],
)
def test_function_namespace_refusal_does_not_invoke_user_protocols(route: str) -> None:
    """Malformed or unbounded namespaces refuse before protocol dispatch or execution.

    Parameters
    ----------
    route
        Native namespace corruption, size bound or dictionary/tuple subclass.

    """
    calls: list[str] = []

    class Defaults(tuple[float, ...]):
        def __len__(self) -> int:
            """Record any attempt to inspect defaults through an overridden protocol."""
            calls.append("tuple-length")
            return super().__len__()

    class Namespace(dict[str, object]):
        def __len__(self) -> int:
            """Record any attempt to measure a dictionary subclass through user code."""
            calls.append("dictionary-length")
            return super().__len__()

        def __iter__(self) -> Iterator[str]:
            """Record any attempt to iterate a dictionary subclass through user code."""
            calls.append("dictionary-iteration")
            return super().__iter__()

    def objective(values: TraceADArray, coefficient: float = 3.0) -> object:
        return values[0] * coefficient

    function = cast(FunctionType, objective)
    if route == "globals":
        namespace = dict(function.__globals__)
        cast(dict[object, object], namespace)[0] = calls
        function = FunctionType(function.__code__, namespace, argdefs=function.__defaults__)
    elif route == "attributes":
        cast(dict[object, object], function.__dict__)[0] = calls
    elif route == "keyword_defaults":
        function.__kwdefaults__ = cast(dict[str, object], {0: 3.0})
    elif route == "oversized":
        function.__dict__.update({f"coefficient_{index}": index for index in range(4097)})
    elif route == "tuple_subclass":
        function.__defaults__ = Defaults((3.0,))
    else:
        function.__kwdefaults__ = Namespace(coefficient=3.0)

    tree = ast.parse(textwrap.dedent(inspect.getsource(objective)))
    baseline = active_reserved_bytes()
    findings = find_objective_effects(function, tree)
    assert len(findings) == 1
    assert findings[0].semantic == "external_callback"
    assert findings[0].detail == "external callback namespace must be a bounded plain dictionary"
    assert getattr(findings[0].node, "lineno", 0) == 1
    assert calls == []
    assert active_reserved_bytes() == baseline


def test_empty_objective_capture_returns_a_located_effect_finding() -> None:
    """A deleted root closure binding is refused without evaluating the objective."""
    coefficient = 3.0

    def objective(values: TraceADArray) -> object:
        return values[0] * coefficient

    closure = cast(FunctionType, objective).__closure__
    assert closure is not None
    del closure[0].cell_contents
    tree = ast.parse(textwrap.dedent(inspect.getsource(objective)))
    baseline = active_reserved_bytes()
    findings = find_objective_effects(objective, tree)
    assert len(findings) == 1
    assert findings[0].semantic == "external_callback"
    assert findings[0].detail == "external callback has an empty captured binding"
    assert getattr(findings[0].node, "lineno", 0) == 1
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("field", ["name", "qualified_name", "filename"])
def test_root_code_metadata_refuses_before_name_protocols(field: str) -> None:
    """Root source selection refuses foreign metadata before comparing function names.

    Parameters
    ----------
    field
        Root function name, qualified name or source filename.

    """
    calls: list[str] = []

    class CodeName(str):
        def __eq__(self, other: object) -> bool:
            """Record an unwanted source-definition comparison."""
            calls.append("equality")
            return super().__eq__(other)

        def __hash__(self) -> int:
            """Record an unwanted source-definition hash."""
            calls.append("hashing")
            return super().__hash__()

    def objective(values: TraceADArray) -> object:
        return values[0] * 3.0

    tree = ast.parse(textwrap.dedent(inspect.getsource(objective)))
    function = cast(FunctionType, objective)
    code = function.__code__
    if field == "name":
        function.__code__ = code.replace(co_name=CodeName(code.co_name))
    elif field == "qualified_name":
        function.__code__ = code.replace(co_qualname=CodeName(code.co_qualname))
    else:
        function.__code__ = code.replace(co_filename=CodeName(code.co_filename))

    baseline = active_reserved_bytes()
    findings = find_objective_effects(function, tree)
    assert calls == []
    assert len(findings) == 1
    assert findings[0].semantic == "external_callback"
    if field == "name":
        assert findings[0].detail == "external callback source definition is unavailable"
        assert findings[0].node is tree
    else:
        assert (
            findings[0].detail == "external callback code metadata must use plain immutable values"
        )
        assert getattr(findings[0].node, "lineno", 0) == 1
    assert active_reserved_bytes() == baseline


def test_unknown_arity_required_helper_operands_keep_product_derivatives() -> None:
    """Unknown expansion binds both required active operands before native replay."""

    def product(first: TraceADScalar, second: TraceADScalar) -> object:
        return first * second

    def objective(values: TraceADArray) -> object:
        return product(*values)

    baseline = active_reserved_bytes()
    result = whole_program_value_and_grad(objective, [2.0, 3.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0, 2.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0, 2.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("field", ["name", "qualified_name", "filename"])
def test_root_code_metadata_runtime_refuses_without_protocols(field: str) -> None:
    """The public numerical entry point refuses root metadata before string protocols.

    Parameters
    ----------
    field
        Root loaded name, lexical name or source path metadata.

    """
    calls: list[str] = []

    class CodeName(str):
        def __eq__(self, other: object) -> bool:
            """Record an unwanted numerical-entry metadata comparison."""
            calls.append("equality")
            return super().__eq__(other)

        def __hash__(self) -> int:
            """Record an unwanted numerical-entry metadata hash."""
            calls.append("hashing")
            return super().__hash__()

    def objective(values: TraceADArray) -> object:
        return values[0] * 3.0

    function = cast(FunctionType, objective)
    code = function.__code__
    if field == "name":
        function.__code__ = code.replace(co_name=CodeName(code.co_name))
    elif field == "qualified_name":
        function.__code__ = code.replace(co_qualname=CodeName(code.co_qualname))
    else:
        function.__code__ = code.replace(co_filename=CodeName(code.co_filename))

    baseline = active_reserved_bytes()
    reason = (
        "objective source filename must be a plain string"
        if field == "filename"
        else "external_callback.*line="
    )
    with pytest.raises(ValueError, match=reason):
        whole_program_value_and_grad(function, [2.0], trace=False)
    assert calls == []
    assert active_reserved_bytes() == baseline

    function.__code__ = code
    result = whole_program_value_and_grad(function, [2.0], trace=False)
    assert result.value == 6.0
    np.testing.assert_array_equal(result.gradient, [3.0])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    assert calls == []
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "binding",
    [
        "default_captured",
        "keyword_captured",
        "keyword_owned",
        "required_keyword_captured",
        "required_keyword_owned",
    ],
)
def test_unknown_arity_helper_binding_keeps_default_and_keyword_storage(binding: str) -> None:
    """Unknown expansion arity retains default or keyword buffer ownership.

    Parameters
    ----------
    binding
        Default, optional keyword or required keyword selecting caller or local storage.

    """
    state = np.array([7.0])

    def required_helper(output: NDArray[np.float64]) -> float:
        np.add(1.0, 2.0, out=output)
        return float(output[0])

    def default_helper(output: NDArray[np.float64] = state) -> float:
        np.add(1.0, 2.0, out=output)
        return float(output[0])

    invoke = cast(
        Callable[..., float],
        required_helper if binding.startswith("required_") else default_helper,
    )
    if binding == "default_captured":

        def objective(values: TraceADArray) -> object:
            return values[0] * invoke(*range(0))

    elif binding.endswith("captured"):

        def objective(values: TraceADArray) -> object:
            return values[0] * invoke(*range(0), output=state)

    else:

        def objective(values: TraceADArray) -> object:
            return values[0] * invoke(*range(0), output=np.zeros(1))

    baseline = active_reserved_bytes()
    if binding.endswith("captured"):
        with pytest.raises(ValueError, match="captured_mutation.*line="):
            whole_program_value_and_grad(objective, [2.0], trace=False)
    else:
        result = whole_program_value_and_grad(objective, [2.0], trace=False)
        assert result.value == 6.0
        np.testing.assert_array_equal(result.gradient, [3.0])
        np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [3.0])
    np.testing.assert_array_equal(state, [7.0])
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("field", ["name", "qualified_name", "filename"])
def test_nested_helper_code_metadata_refuses_without_string_protocols(field: str) -> None:
    """Nested executable code retains the same plain metadata boundary as its parent.

    Parameters
    ----------
    field
        Nested function name, qualified name or source filename.

    """
    calls: list[str] = []

    class CodeName(str):
        def __eq__(self, other: object) -> bool:
            """Record an unwanted nested-code comparison."""
            calls.append("equality")
            return super().__eq__(other)

        def __hash__(self) -> int:
            """Record an unwanted nested-code hash."""
            calls.append("hashing")
            return super().__hash__()

    def helper() -> float:
        def coefficient() -> float:
            return 3.0

        return coefficient()

    function = cast(FunctionType, helper)
    constants = list(function.__code__.co_consts)
    index = next(index for index, item in enumerate(constants) if type(item) is CodeType)
    nested = cast(CodeType, constants[index])
    if field == "name":
        replacement = nested.replace(co_name=CodeName(nested.co_name))
    elif field == "qualified_name":
        replacement = nested.replace(co_qualname=CodeName(nested.co_qualname))
    else:
        replacement = nested.replace(co_filename=CodeName(nested.co_filename))
    constants[index] = replacement
    function.__code__ = function.__code__.replace(co_consts=tuple(constants))

    def objective(values: TraceADArray) -> object:
        return values[0] * helper()

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="source does not match captured function.*line="):
        whole_program_value_and_grad(objective, [2.0], trace=False)
    assert calls == []
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("field", ["name", "qualified_name", "filename"])
def test_helper_code_names_refuse_without_invoking_string_protocols(field: str) -> None:
    """Foreign code names cannot execute protocols during source admission.

    Parameters
    ----------
    field
        Loaded function name, qualified name or source filename.

    """
    calls: list[str] = []

    class CodeName(str):
        def __eq__(self, other: object) -> bool:
            """Record unwanted code metadata comparison."""
            calls.append("equality")
            return super().__eq__(other)

        def __hash__(self) -> int:
            """Record unwanted code metadata hashing."""
            calls.append("hashing")
            return super().__hash__()

        def __str__(self) -> str:
            """Record unwanted code metadata conversion."""
            calls.append("conversion")
            return super().__str__()

    def helper() -> float:
        return 3.0

    function = cast(FunctionType, helper)
    code = function.__code__
    if field == "name":
        function.__code__ = code.replace(co_name=CodeName(code.co_name))
    elif field == "qualified_name":
        function.__code__ = code.replace(co_qualname=CodeName(code.co_qualname))
    else:
        function.__code__ = code.replace(co_filename=CodeName(code.co_filename))

    def objective(values: TraceADArray) -> object:
        return values[0] * helper()

    tree = ast.parse(textwrap.dedent(inspect.getsource(objective)))
    baseline = active_reserved_bytes()
    findings = find_objective_effects(objective, tree)
    assert calls == []
    assert len(findings) == 1
    assert findings[0].semantic == "external_callback"
    assert findings[0].detail == "external callback code metadata must use plain immutable values"
    assert getattr(findings[0].node, "lineno", 0) == 2
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize(
    "fault", ["missing", "empty", "malformed", "tokenizer", "renamed", "module_syntax"]
)
def test_helper_source_loss_refuses_without_executing_original_code(
    tmp_path: Path, fault: str
) -> None:
    """Real source loss or corruption returns a located refusal before callback effects.

    Parameters
    ----------
    tmp_path
        Owned temporary location for an actual compiled helper source.
    fault
        Missing, truncated, malformed, unterminated or mismatched file source.

    """
    calls: list[str] = []
    source = "def helper():\n    calls.append('called')\n    return 3.0\n"
    path = tmp_path / "source_helper.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins), "calls": calls}
    exec(compile(source, str(path), "exec"), namespace)
    helper = cast(Callable[[], float], namespace["helper"])

    def objective(values: TraceADArray) -> object:
        return values[0] * helper()

    expected = "external callback source cannot be inspected"
    if fault == "missing":
        path.unlink()
        expected = "external callback source is unavailable"
    elif fault == "empty":
        path.write_text("", encoding="utf-8")
    elif fault == "malformed":
        path.write_text("def helper():\n    return +\n", encoding="utf-8")
    elif fault == "tokenizer":
        path.write_text("def helper():\n    return (\n", encoding="utf-8")
    elif fault == "module_syntax":
        path.write_text(source + "\ninvalid module syntax\n", encoding="utf-8")
        expected = "external callback source does not match captured function"
    else:
        path.write_text("def other():\n    return 3.0\n", encoding="utf-8")
        expected = "external callback source definition is unavailable"

    tree = ast.parse(textwrap.dedent(inspect.getsource(objective)))
    baseline = active_reserved_bytes()
    findings = find_objective_effects(objective, tree)
    assert len(findings) == 1
    assert findings[0].semantic == "external_callback"
    assert findings[0].detail == expected
    assert getattr(findings[0].node, "lineno", 0) == 2
    with pytest.raises(ValueError, match=f"{expected}.*line="):
        whole_program_value_and_grad(objective, [2.0], trace=False)
    assert calls == []
    assert active_reserved_bytes() == baseline


def test_replaced_helper_source_cannot_hide_the_loaded_functions_write(tmp_path: Path) -> None:
    """Changed source cannot admit loaded callback bytecode that writes caller state.

    Parameters
    ----------
    tmp_path
        Owned source location for the actually compiled callback.

    """
    calls: list[str] = []
    source = "def helper():\n    calls.append('called')\n    return 3.0\n"
    path = tmp_path / "replaced_helper.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict[str, object] = {"__builtins__": vars(builtins), "calls": calls}
    exec(compile(source, str(path), "exec"), namespace)
    helper = cast(Callable[[], float], namespace["helper"])
    path.write_text("def helper():\n    return 3.0\n", encoding="utf-8")

    def objective(values: TraceADArray) -> object:
        return values[0] * helper()

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="source does not match captured function.*line="):
        whole_program_value_and_grad(objective, [2.0], trace=False)
    assert calls == []
    assert active_reserved_bytes() == baseline


def test_loaded_helper_lexical_identity_must_match_the_actual_file() -> None:
    """A requalified code object cannot borrow the original callback's source custody."""
    calls: list[str] = []

    def helper() -> float:
        calls.append("called")
        return 3.0

    function = cast(FunctionType, helper)
    function.__code__ = function.__code__.replace(co_qualname="other_scope.helper")

    def objective(values: TraceADArray) -> object:
        return values[0] * helper()

    baseline = active_reserved_bytes()
    with pytest.raises(ValueError, match="source does not match captured function.*line="):
        whole_program_value_and_grad(objective, [2.0], trace=False)
    assert calls == []
    assert active_reserved_bytes() == baseline


@pytest.mark.parametrize("mode", ["owned", "alternative"])
def test_helper_code_matching_preserves_docstrings_and_folded_sets(mode: str) -> None:
    """Loaded-source matching retains closure scope, literal documentation and set constants.

    Parameters
    ----------
    mode
        Selector with independent analytic coefficient three or five.

    """

    def helper(value: TraceADScalar) -> object:
        """Select one fixed coefficient.

        The compiler must preserve this literal's original indentation.
        """
        return value * (3.0 if mode in {"owned", "passive"} else 5.0)

    def objective(values: TraceADArray) -> object:
        return helper(cast(TraceADScalar, values[0]))

    coefficient = 3.0 if mode == "owned" else 5.0
    result = whole_program_value_and_grad(objective, [2.0], trace=False)
    assert result.value == 2.0 * coefficient
    np.testing.assert_array_equal(result.gradient, [coefficient])
    np.testing.assert_array_equal(program_adjoint_replay_gradient(result), [coefficient])


@pytest.mark.parametrize("payload", ["foreign", "depth", "cardinality"])
def test_loaded_helper_code_constants_refuse_before_protocols_or_execution(payload: str) -> None:
    """Unsupported loaded constants are refused without calling equality or representation.

    Parameters
    ----------
    payload
        Foreign object, excessive nesting or excessive constant cardinality.

    """
    calls: list[str] = []

    class ForeignConstant:
        def __eq__(self, other: object) -> bool:
            """Record unwanted comparison of a captured foreign constant."""
            calls.append("equality")
            return True

        def __repr__(self) -> str:
            """Record unwanted representation of a captured foreign constant."""
            calls.append("representation")
            return "foreign"

    def helper() -> float:
        return 3.0

    if payload == "foreign":
        incoming: object = ForeignConstant()
    elif payload == "depth":
        incoming = 3.0
        for _ in range(68):
            incoming = (incoming,)
    else:
        incoming = tuple(3.0 for _ in range(4097))
    function = cast(FunctionType, helper)
    function.__code__ = function.__code__.replace(co_consts=(None, incoming))

    def objective(values: TraceADArray) -> object:
        return values[0] * helper()

    tree = ast.parse(textwrap.dedent(inspect.getsource(objective)))
    baseline = active_reserved_bytes()
    findings = find_objective_effects(objective, tree)
    assert len(findings) == 1
    assert findings[0].semantic == "external_callback"
    assert findings[0].detail == "external callback source does not match captured function"
    assert getattr(findings[0].node, "lineno", 0) == 2
    assert calls == []
    assert active_reserved_bytes() == baseline
