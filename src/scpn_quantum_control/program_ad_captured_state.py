# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — captured program state binding
"""Bind an in-memory derivative tape to its callable and captured numeric state.

This companion does not modify the historical effect IR wire format. It checks
the current state at derivative access; it does not lock caller-owned containers
or detect a mutation that was completely reverted between observations.
The snapshot bounds count actual encoded UTF-8 bytes and refuse an oversized
payload before copying it into fingerprint storage.

The live binding identifies shared objects and code by their addresses, which
only mean something inside one process. A portable digest of the same traversal
numbers shared objects in first-visit order and hashes code by its content,
without its file path or line table, so a saved program can be checked again in
a later process running the same interpreter.
"""

from __future__ import annotations

import dis
import hashlib
import struct
import sys
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from types import (
    BuiltinFunctionType,
    CodeType,
    FunctionType,
    GenericAlias,
    GetSetDescriptorType,
    ModuleType,
)

import numpy as np
from numpy.typing import NDArray

from .execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from .execution_reservations import ExecutionMemoryReservation, reserve_execution_memory
from .program_ad_captured_memory import (
    _CAPTURED_STATE_MAX_NODES,
    _CapturedCodeRefusal,
    _CapturedStateMemory,
)

_MAX_NODES = _CAPTURED_STATE_MAX_NODES
_MAX_BYTES = 8 * 1024**2
_MAX_DEPTH = 64
_CAPTURED_STATE_DIGEST_BYTES = hashlib.sha256().digest_size + sys.getsizeof(b"")
_CALLABLE_ALIAS = type(Callable[[], object])
_NUMPY_SCALARS = frozenset(
    type(np.array(0, dtype=dtype)[()])
    for dtype in "?" + np.typecodes["AllFloat"] + np.typecodes["AllInteger"]
)
_NUMPY_CALLABLES = tuple(
    value
    for namespace in (vars(np), vars(np.linalg))
    for value in namespace.values()
    if callable(value)
)
_NUMPY_CALLABLE_IDS = frozenset(id(value) for value in _NUMPY_CALLABLES)
_NO_BATCHING_TRANSFORM = object()


def _portable_ordinal_bytes() -> int:
    """Measure the first-visit numbering table at its node limit, once at import."""
    address = (1 << (8 * struct.calcsize("P"))) - 1
    table = sys.getsizeof(dict.fromkeys(range(_MAX_NODES), 0))
    return table + _MAX_NODES * (sys.getsizeof(address) + sys.getsizeof(_MAX_NODES))


_PORTABLE_ORDINAL_BYTES = _portable_ordinal_bytes()


def _batching_transform() -> object:
    """Return the package's own ``vmap``, or a value nothing equals before it is loaded."""
    module = sys.modules.get(f"{__package__}.differentiable_vmap")
    return getattr(module, "vmap", _NO_BATCHING_TRANSFORM)


_PASSIVE_CLASS_FIELDS = frozenset(
    {
        "__module__",
        "__qualname__",
        "__dict__",
        "__weakref__",
        "__doc__",
        "__annotations__",
        "__firstlineno__",
        "__static_attributes__",
    }
)


def _is_passive_local_class(value: object) -> bool:
    """Admit plain local containers without constructors or attribute protocols."""
    if type(value) is not type:
        return False
    constructor = value
    if type.__getattribute__(constructor, "__bases__") != (object,):
        return False
    namespace = type.__getattribute__(constructor, "__dict__")
    for name, item in namespace.items():
        if name not in _PASSIVE_CLASS_FIELDS:
            return False
        if name in {"__dict__", "__weakref__"}:
            if type(item) is not GetSetDescriptorType or item.__objclass__ is not constructor:
                return False
        elif name == "__annotations__":
            if type(item) is not dict:
                return False
        elif name == "__firstlineno__":
            if type(item) is not int:
                return False
        elif name == "__static_attributes__":
            if type(item) is not tuple or any(type(attribute) is not str for attribute in item):
                return False
        elif item is not None and type(item) is not str:
            return False
    return True


@dataclass(frozen=True, slots=True)
class _CapturedProgramState:
    """Private live binding; an unsupported snapshot cannot admit a derivative."""

    objective: Callable[..., object] = field(repr=False, compare=False)
    digest: bytes | None = field(repr=False)
    code_references: tuple[CodeType, ...] = field(repr=False, compare=False)
    tape_digest: bytes | None = field(default=None, repr=False)

    def require_current(self, checkpoint: Callable[[], None] | None = None) -> None:
        """Refuse unsupported or changed state without invoking the objective."""
        if self.digest is None:
            raise ValueError("captured program state contains unsupported storage")
        current, _ = _snapshot(self.objective, checkpoint)
        if current != self.digest:
            raise ValueError("captured program state changed after derivative capture")


def _capture_program_state(
    objective: Callable[..., object], checkpoint: Callable[[], None] | None = None
) -> _CapturedProgramState:
    """Capture a bounded state digest before numerical execution.

    An unsupported state is recorded until a successful objective returns, so
    the objective's existing malformed-operation refusal retains its meaning.
    It can never pass ``require_current`` on a completed derivative result.
    """
    digest, code_references = _snapshot(objective, checkpoint)
    return _CapturedProgramState(objective, digest, code_references)


def _portable_program_state_digest(
    objective: Callable[..., object], checkpoint: Callable[[], None] | None = None
) -> bytes | None:
    """Return the process-independent captured-state digest, or ``None`` if unsupported.

    It covers the same callable, closure, default, global and numeric storage as
    the live binding. Shared objects are numbered in first-visit order and code
    is hashed by its instructions, names, flags and constants, so the digest is
    stable across processes of the same interpreter but changes whenever the
    captured numeric state, the aliasing between captured objects or the code
    itself changes.
    """
    digest, _ = _snapshot(objective, checkpoint, portable=True)
    return digest


def _snapshot(
    objective: Callable[..., object],
    checkpoint: Callable[[], None] | None,
    *,
    portable: bool = False,
) -> tuple[bytes | None, tuple[CodeType, ...]]:
    buffers = [
        ExecutionBuffer(
            "captured_state_digest", "intermediate", (_CAPTURED_STATE_DIGEST_BYTES,), "uint8"
        )
    ]
    if portable:
        buffers.append(
            ExecutionBuffer(
                "captured_state_portable_ordinals",
                "intermediate",
                (_PORTABLE_ORDINAL_BYTES,),
                "uint8",
            )
        )
    plan = ExecutionMemoryPlan(tuple(buffers))
    with reserve_execution_memory(plan) as reservation:
        fingerprint = _Fingerprint(checkpoint, reservation, portable=portable)
        supported = True
        try:
            fingerprint.visit(objective, 0, ())
        except (_UnsupportedState, _CapturedCodeRefusal):
            supported = False
        digest = fingerprint.hash.digest() if supported else None
        references = tuple(fingerprint.code_references)
        reservation.checkpoint()
        return digest, references


class _UnsupportedState(ValueError):
    """An exact captured storage type or bounded snapshot limit was refused."""


class _Fingerprint:
    __slots__ = (
        "hash",
        "checkpoint",
        "reservation",
        "memory",
        "seen",
        "module_scopes",
        "code_references",
        "bytes",
        "nodes",
        "ordinals",
    )

    def __init__(
        self,
        checkpoint: Callable[[], None] | None,
        reservation: ExecutionMemoryReservation,
        *,
        portable: bool = False,
    ) -> None:
        self.hash = hashlib.sha256()
        self.checkpoint = checkpoint
        self.reservation = reservation
        self.memory = _CapturedStateMemory(
            reservation,
            _CAPTURED_STATE_DIGEST_BYTES + sys.getsizeof(self.hash),
            sys.getsizeof(self),
            _FRAME_BYTES,
        )
        self.seen: set[int] = set()
        self.module_scopes: set[tuple[int, tuple[str, ...]]] = set()
        self.code_references: list[CodeType] = []
        self.bytes = 0
        self.nodes = 0
        self.ordinals: dict[int, int] | None = {} if portable else None

    def identity(self, identity: int) -> bytes:
        """Name a shared object by address, or by first-visit order when portable."""
        if self.ordinals is None:
            return str(identity).encode("ascii")
        ordinal = self.ordinals.setdefault(identity, len(self.ordinals))
        return b"#" + str(ordinal).encode("ascii")

    def code_identity(self, function: FunctionType) -> None:
        """Bind library code by address, or by its qualified name when portable."""
        if self.ordinals is None:
            self.add(str(id(function.__code__)).encode("ascii"))
        else:
            self.add(f"{function.__module__}:{function.__qualname__}".encode())

    def code_content(self, code: CodeType, depth: int) -> None:
        """Hash code instructions, names, flags and constants, without location.

        Constant nesting was already bounded when ``memory.code`` admitted this
        code; scalar constants still pass through ``visit`` and its depth limit.
        """
        self.add(b"code")
        self.add(code.co_code)
        self.add(code.co_exceptiontable)
        self.add(code.co_qualname.encode("utf-8", errors="surrogatepass"))
        self.add(
            struct.pack(
                "!4Q",
                code.co_argcount,
                code.co_posonlyargcount,
                code.co_kwonlyargcount,
                code.co_flags,
            )
        )
        for names in (code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars):
            self.add(str(len(names)).encode("ascii"))
            for name in names:
                self.add(name.encode("utf-8", errors="surrogatepass"))
        self.add(str(len(code.co_consts)).encode("ascii"))
        for value in code.co_consts:
            self.constant(value, depth + 1)

    def constant(self, value: object, depth: int) -> None:
        """Hash one compiler constant; set members are ordered by type and text."""
        kind = type(value)
        if kind is CodeType:
            self.code_content(typing.cast(CodeType, value), depth)
        elif kind is tuple or kind is frozenset:
            items = typing.cast(tuple[object, ...] | frozenset[object], value)
            ordered = (
                tuple(items)
                if kind is tuple
                else tuple(sorted(items, key=lambda item: (type(item).__name__, repr(item))))
            )
            self.add(b"const-tuple" if kind is tuple else b"const-frozenset")
            self.add(str(len(ordered)).encode("ascii"))
            for item in ordered:
                self.constant(item, depth + 1)
        else:
            self.visit(value, depth, ())

    def add(self, value: bytes) -> None:
        """Hash a length-delimited payload within the snapshot byte limit."""
        self.bytes += len(value)
        if self.bytes > _MAX_BYTES:
            raise _UnsupportedState("captured state exceeds the bounded snapshot")
        self.hash.update(struct.pack("!Q", len(value)))
        self.hash.update(value)

    def copy_bytes(self, size: int, produce: Callable[[], bytes]) -> None:
        """Admit a snapshot payload before allocating its byte copy."""
        if size > _MAX_BYTES - self.bytes:
            raise _UnsupportedState("captured state exceeds the bounded snapshot")
        plan = ExecutionMemoryPlan(
            (
                ExecutionBuffer(
                    "captured_state_byte_copy",
                    "intermediate",
                    (max(1, size) + sys.getsizeof(b""),),
                    "uint8",
                ),
            )
        )
        with reserve_execution_memory(plan) as reservation:
            self.add(produce())
            reservation.checkpoint()

    def visit(self, value: object, depth: int, module_names: tuple[str, ...]) -> None:
        """Hash an exact value within traversal and encoded-byte limits."""
        self.nodes += 1
        if self.nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise _UnsupportedState("captured state exceeds the bounded snapshot")
        self.memory.visit(depth)
        self.reservation.checkpoint()
        if self.checkpoint is not None:
            self.checkpoint()
        kind = type(value)
        if value is None:
            self.add(b"none")
        elif value is Ellipsis:
            self.add(b"ellipsis")
        elif kind is bool:
            self.add(b"true" if value else b"false")
        elif kind is int:
            integer = typing.cast(int, value)
            if integer.bit_length() > _MAX_BYTES * 8:
                raise _UnsupportedState("captured integer exceeds the bounded snapshot")
            self.add(b"int")
            size = max(1, (integer.bit_length() + 8) // 8)
            self.copy_bytes(size, lambda: integer.to_bytes(size, "big", signed=True))
        elif kind is float:
            self.add(b"float" + struct.pack("!d", typing.cast(float, value)))
        elif kind is complex:
            number = typing.cast(complex, value)
            self.add(b"complex" + struct.pack("!dd", number.real, number.imag))
        elif kind is str:
            string = typing.cast(str, value)
            if len(string) > _MAX_BYTES - self.bytes:
                raise _UnsupportedState("captured string exceeds the bounded snapshot")
            self.add(b"str")
            size = len(string)
            if not string.isascii():
                size = 0
                for index, character in enumerate(string):
                    if index % 4096 == 0:
                        self.reservation.checkpoint()
                    ordinal = ord(character)
                    size += (
                        1
                        if ordinal < 0x80
                        else 2
                        if ordinal < 0x800
                        else 3
                        if ordinal < 0x10000
                        else 4
                    )
                    if size > _MAX_BYTES - self.bytes:
                        raise _UnsupportedState("captured string exceeds the bounded snapshot")
            self.copy_bytes(size, lambda: string.encode("utf-8", errors="surrogatepass"))
        elif kind is bytes:
            self.add(b"bytes")
            self.add(typing.cast(bytes, value))
        elif any(kind is scalar_type for scalar_type in _NUMPY_SCALARS):
            scalar = typing.cast(np.generic, value)
            self.add(b"numpy-scalar")
            self.add(scalar.dtype.str.encode("ascii"))
            self.copy_bytes(scalar.dtype.itemsize, scalar.tobytes)
        else:
            self.aggregate(value, depth, module_names)

    def aggregate(self, value: object, depth: int, module_names: tuple[str, ...]) -> None:
        """Fingerprint supported storage and references at the current depth."""
        identity = id(value)
        self.add(self.identity(identity))
        kind = type(value)
        if kind is ModuleType:
            self.memory.scope(len(self.module_scopes) + 1)
            scope = (identity, module_names)
            if scope in self.module_scopes:
                self.add(b"module-reference")
                return
            self.module_scopes.add(scope)
        else:
            if identity in self.seen:
                self.add(b"reference")
                return
            self.memory.seen(len(self.seen) + 1)
            self.seen.add(identity)
        if identity in _NUMPY_CALLABLE_IDS:
            self.add(b"numpy-intrinsic")
            self.add(_intrinsic_name(value))
            if kind is FunctionType:
                function = typing.cast(FunctionType, value)
                self.memory.reference(len(self.code_references) + 1)
                self.code_references.append(function.__code__)
                self.code_identity(function)
        elif value is _batching_transform():
            # The package's own transform is bound by identity and code, like a
            # native callable; its module state is not captured program state.
            transform = typing.cast(FunctionType, value)
            self.add(b"batching-transform")
            self.memory.reference(len(self.code_references) + 1)
            self.code_references.append(transform.__code__)
            self.code_identity(transform)
        elif kind is list or kind is tuple:
            items = typing.cast(list[object] | tuple[object, ...], value)
            if len(items) > _MAX_NODES:
                raise _UnsupportedState("captured container exceeds the bounded snapshot")
            self.add(b"list" if kind is list else b"tuple")
            self.add(str(len(items)).encode("ascii"))
            for item in items:
                self.visit(item, depth + 1, module_names)
        elif kind is slice:
            # A slice is immutable; its three bounds are fingerprinted as values.
            bounds = typing.cast(slice, value)
            self.add(b"slice")
            for bound in (bounds.start, bounds.stop, bounds.step):
                self.visit(bound, depth + 1, module_names)
        elif kind is dict:
            mapping = typing.cast(dict[object, object], value)
            if len(mapping) > _MAX_NODES:
                raise _UnsupportedState("captured mapping exceeds the bounded snapshot")
            self.add(b"dict")
            self.add(str(len(mapping)).encode("ascii"))
            for key, item in mapping.items():
                self.visit(key, depth + 1, module_names)
                self.visit(item, depth + 1, module_names)
        elif kind is np.ndarray:
            array = typing.cast(NDArray[np.generic], value)
            if array.dtype.kind not in "biufc" or array.nbytes > _MAX_BYTES - self.bytes:
                raise _UnsupportedState("captured array storage is unsupported")
            self.add(b"ndarray")
            self.add(array.dtype.str.encode("ascii"))
            self.add(str(array.shape).encode("ascii"))
            self.add(str(array.strides).encode("ascii"))
            self.copy_bytes(array.nbytes, lambda: array.tobytes(order="C"))
        elif kind is FunctionType:
            self.function(typing.cast(FunctionType, value), depth)
        elif kind is ModuleType:
            module = typing.cast(ModuleType, value)
            self.add(b"module")
            namespace = vars(module)
            self.namespace(namespace)
            for name in module_names:
                if name in namespace:
                    self.add(name.encode("utf-8"))
                    self.visit(namespace[name], depth + 1, module_names)
        elif kind is BuiltinFunctionType:
            builtin = typing.cast(BuiltinFunctionType, value)
            self.add(b"builtin")
            self.add(_intrinsic_name(builtin))
            self.visit(builtin.__self__, depth + 1, module_names)
        elif _is_passive_local_class(value):
            constructor = typing.cast(type, value)
            self.add(b"passive-local-class")
            for name, item in type.__getattribute__(constructor, "__dict__").items():
                if name not in {"__dict__", "__weakref__"}:
                    self.add(name.encode("ascii"))
                    self.visit(item, depth + 1, module_names)
        elif (
            kind is np.ufunc
            or kind is type
            or kind is type(Callable)
            or kind is type(typing.Any)
            or kind is GenericAlias
            or kind is _CALLABLE_ALIAS
        ):
            self.add(b"intrinsic")
            self.add(_intrinsic_name(value))
        else:
            raise _UnsupportedState("captured state storage type is unsupported")

    def function(self, function: FunctionType, depth: int) -> None:
        """Fingerprint callable code and captures without executing the callable."""
        self.memory.reference(len(self.code_references) + 1)
        self.code_references.append(function.__code__)
        self.add(b"function")
        code_owner: FunctionType = function
        if self.ordinals is None or code_owner is typing.cast:
            self.code_identity(code_owner)
        if function is typing.cast:
            return
        if len(function.__code__.co_code) > _MAX_NODES * 16:
            raise _UnsupportedState("captured callable exceeds the bounded snapshot")
        self.visit(function.__defaults__, depth + 1, ())
        self.visit(function.__kwdefaults__, depth + 1, ())
        self.visit(function.__dict__, depth + 1, ())
        names = _attribute_names(function.__code__)
        if len(names) > _MAX_NODES:
            raise _UnsupportedState("captured callable exceeds the bounded snapshot")
        self.memory.code(function.__code__, depth)
        if self.ordinals is not None:
            self.code_content(function.__code__, depth + 1)
        for name, cell in zip(
            function.__code__.co_freevars, function.__closure__ or (), strict=True
        ):
            self.add(name.encode("utf-8"))
            try:
                captured = cell.cell_contents
            except ValueError:
                raise _UnsupportedState("captured closure cell is empty") from None
            self.visit(captured, depth + 1, names)
        global_names = _global_names(function.__code__)
        namespace = function.__globals__
        self.namespace(namespace)
        builtin_namespace: object = FunctionType.__getattribute__(function, "__builtins__")
        self.namespace(builtin_namespace)
        builtin_mapping = typing.cast(dict[str, object], builtin_namespace)
        for name in global_names:
            if name in namespace:
                self.add(name.encode("utf-8"))
                self.visit(namespace[name], depth + 1, names)
            elif name in builtin_mapping:
                self.add(name.encode("utf-8"))
                self.visit(builtin_mapping[name], depth + 1, names)

    def namespace(self, namespace: object) -> None:
        """Reject opaque dictionary protocols or keys before name lookup."""
        if type(namespace) is not dict:
            raise _UnsupportedState("captured namespace must be a plain dictionary")
        mapping = typing.cast(dict[object, object], namespace)
        if len(mapping) > _MAX_NODES or any(type(key) is not str for key in mapping):
            raise _UnsupportedState("captured namespace storage is unsupported")


def _intrinsic_name(value: object) -> bytes:
    """Name a ufunc, builtin, type or typing object without running its descriptors.

    The live digest tells such objects apart by address. The portable digest
    numbers them by first visit, so without a name ``np.add`` and
    ``np.multiply``, or ``max`` and ``min``, would fingerprint alike.
    """
    kind = type(value)
    if kind is np.ufunc:
        return f"ufunc:{typing.cast(np.ufunc, value).__name__}".encode()
    if kind is BuiltinFunctionType:
        builtin = typing.cast(BuiltinFunctionType, value)
        return f"builtin:{builtin.__module__}:{builtin.__qualname__}".encode()
    if kind is type:
        constructor = typing.cast(type, value)
        module = type.__getattribute__(constructor, "__module__")
        qualname = type.__getattribute__(constructor, "__qualname__")
        return f"type:{module}:{qualname}".encode()
    if kind is FunctionType:
        function = typing.cast(FunctionType, value)
        return f"function:{function.__module__}:{function.__qualname__}".encode()
    if kind is GenericAlias or kind is _CALLABLE_ALIAS or kind is type(typing.Any):
        # Typing objects print their full meaning and no address.
        return f"typing:{value!r}".encode()
    kind_name = f"{kind.__module__}.{kind.__qualname__}"
    # NumPy's own callables (array-function dispatchers, ABC classes) carry
    # their names; anything else is named by its type only, never by a repr
    # that could hold an address.
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if isinstance(module, str) and isinstance(name, str):
        return f"{kind_name}:{module}:{name}".encode()
    return kind_name.encode()


def _nested_code(code: CodeType) -> tuple[CodeType, ...]:
    """Return code and every code object nested in its constants, outermost first."""
    found: list[CodeType] = []
    pending = [code]
    while pending:
        current = pending.pop()
        found.append(current)
        pending.extend(constant for constant in current.co_consts if type(constant) is CodeType)
    return tuple(found)


def _attribute_names(code: CodeType) -> tuple[str, ...]:
    """Return every name read by code or its nested code, for module traversal.

    A module attribute read only inside a generator expression must still be
    part of the digest, so the module's members are selected from the names of
    the nested code as well.
    """
    return tuple(dict.fromkeys(name for nested in _nested_code(code) for name in nested.co_names))


def _global_names(code: CodeType) -> tuple[str, ...]:
    """Return the global names read by code and by the code nested in it.

    Generator expressions and comprehensions compile to nested code objects
    that read globals themselves, so scanning only the outer instructions
    would leave those reads out of the digest.
    """
    return tuple(
        dict.fromkeys(
            instruction.argval
            for nested in _nested_code(code)
            for instruction in dis.get_instructions(nested)
            if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
        )
    )


_FRAME_BYTES = sum(
    sys.getsizeof(sys._getframe())
    + (method.__code__.co_nlocals + method.__code__.co_stacksize) * struct.calcsize("P")
    for method in (_Fingerprint.visit, _Fingerprint.aggregate, _Fingerprint.function)
)
