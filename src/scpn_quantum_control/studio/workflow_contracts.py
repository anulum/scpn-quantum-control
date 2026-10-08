# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — immutable typed experiment workflow admission
"""Admit a bounded stage graph without executing a verb or granting approval.

Port descriptors express declared format, dtype, shape and unit equality.
An original handler must still validate the meaning and domain of its inputs.
Workflow connectivity is not scientific equivalence or execution evidence.
The established workspace codec owns lossless numeric transport and hashing.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..studio_workspace.canonical import canonical_bytes
from ..studio_workspace.json_transport import read_json, write_json

WORKFLOW_SCHEMA = "experiment_workflow.v1"
"""Portable graph version; existing workspace formats remain unchanged."""
MAX_WORKFLOW_BYTES = 16 * 1024 * 1024
"""Maximum portable definition size before graph admission."""
MAX_STAGES = 64
"""Maximum declared stages; stable ordering stays bounded."""
MAX_CELLS = 256
"""Maximum Cartesian coordinates including seed identities."""
MAX_EVALUATIONS = 4096
"""Maximum stage evaluations before any execution or grid allocation."""

_VERBS = frozenset(
    {
        "compile",
        "simulate",
        "analyse",
        "validate",
        "benchmark",
        "replay",
        "differentiate",
        "mitigate",
        "execute",
    }
)
_DTYPES = frozenset({"json", "float64", "int64", "uint64", "bool", "utf8"})
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_SCHEMA = re.compile(r"[a-z][a-z0-9_.-]{0,127}")


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _object(value: object, name: str, fields: set[str] | None = None) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name}: object required")
    if fields is not None and set(value) != fields:
        raise ValueError(f"{name}: incomplete or unsupported fields")
    return value


def _array(value: object, name: str, maximum: int) -> list[object]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{name}: bounded array required")
    return value


def _text(value: object, name: str, maximum: int = 128) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= maximum
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError(f"{name}: bounded nonempty text required")
    return value


def _id(value: object, name: str) -> str:
    text = _text(value, name, 64)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ValueError(f"{name}: lowercase ASCII identifier required")
    return text


def _parameter_key(value: object, name: str) -> str:
    text = _text(value, name, 64)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", text) is None:
        raise ValueError(f"{name}: original ASCII parameter key required")
    return text


def _integer(value: object, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: integer between {low} and {high} required")
    return value


@dataclass(frozen=True)
class WorkflowPortType:
    """Exact declared format of one unchanged producer value.

    Parameters
    ----------
    schema
        Explicit source-format identity; matching it does not prove science.
    dtype
        json, float64, int64, uint64, bool or utf8; no implicit conversion.
    shape
        Exact nested array dimensions; empty means one scalar or JSON datum.
    unit
        Exact declared unit string; no automatic unit conversion.

    """

    schema: str
    dtype: str
    shape: tuple[int, ...]
    unit: str


@dataclass(frozen=True)
class WorkflowInput:
    """Bind a typed original output to a handler parameter.

    Parameters
    ----------
    parameter
        Target parameter key; it cannot also have a constant or sweep value.
    source_stage
        Existing producer-stage identity.
    source_port
        Existing named producer output.
    type
        Exact consumer format, dtype, shape and unit declaration.

    """

    parameter: str
    source_stage: str
    source_port: str
    type: WorkflowPortType


@dataclass(frozen=True)
class WorkflowOutput:
    """Select an unchanged value from the original runtime output.

    Parameters
    ----------
    name
        Unique stage-local output port.
    path
        Bounded object-key path; no expressions, transformations or execution.
    type
        Exact declared output format, dtype, shape and unit.

    """

    name: str
    path: tuple[str, ...]
    type: WorkflowPortType


@dataclass(frozen=True)
class WorkflowStage:
    """One explicitly selected original operation and its dependency ports.

    Parameters
    ----------
    id
        Unique stable stage identifier.
    adapter
        executive or local-kuramoto; these models are distinct.
    verb
        Original operation name; actual availability is checked by its owner.
    backend
        Explicit backend identity; no fallback backend is substituted.
    parameters
        Deeply immutable original constant parameters.
    inputs
        Typed source bindings that add producer dependencies.
    outputs
        Named original output selections.
    depends_on
        Explicit control dependencies in addition to data dependencies.

    """

    id: str
    adapter: str
    verb: str
    backend: str
    parameters: Mapping[str, object]
    inputs: tuple[WorkflowInput, ...]
    outputs: tuple[WorkflowOutput, ...]
    depends_on: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowSweep:
    """Ordered original values, seed identities and total evaluation budget.

    Parameters
    ----------
    axes
        Stage, parameter and ordered unique original values for each axis.
    seeds
        Unique canonical uint64 decimal identities.
    seed_binding
        Explicit original seed parameter, or None for unconsumed cell seeds.
    evaluation_budget
        Declared total stage-evaluation ceiling, never an elapsed-time promise.

    """

    axes: tuple[tuple[str, str, tuple[object, ...]], ...]
    seeds: tuple[str, ...]
    seed_binding: tuple[str, str] | None
    evaluation_budget: int


@dataclass(frozen=True)
class WorkflowDefinition:
    """Completely admitted immutable graph; it grants no execution authority.

    Parameters
    ----------
    workflow_id
        Original stable workflow identity.
    stages
        Original stage order, distinct from deterministic execution order.
    sweep
        Bounded source values and declared evaluation budget.
    extensions
        Opaque immutable metadata; never imported code or approval.

    """

    workflow_id: str
    stages: tuple[WorkflowStage, ...]
    sweep: WorkflowSweep
    extensions: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return an independent exact v1 wire document.

        Returns
        -------
        dict[str, object]
            Original order and scalar types, without execution-derived fields.

        """
        return {
            "schema": WORKFLOW_SCHEMA,
            "body": {
                "workflow_id": self.workflow_id,
                "stages": [
                    {
                        "id": stage.id,
                        "adapter": stage.adapter,
                        "verb": stage.verb,
                        "backend": stage.backend,
                        "parameters": _thaw(stage.parameters),
                        "inputs": [
                            {
                                "parameter": item.parameter,
                                "source_stage": item.source_stage,
                                "source_port": item.source_port,
                                "type": _port_dict(item.type),
                            }
                            for item in stage.inputs
                        ],
                        "outputs": [
                            {
                                "name": item.name,
                                "path": list(item.path),
                                "type": _port_dict(item.type),
                            }
                            for item in stage.outputs
                        ],
                        "depends_on": list(stage.depends_on),
                    }
                    for stage in self.stages
                ],
                "sweep": {
                    "axes": [
                        {"stage_id": sid, "parameter": key, "values": _thaw(values)}
                        for sid, key, values in self.sweep.axes
                    ],
                    "seeds": list(self.sweep.seeds),
                    "seed_binding": None
                    if self.sweep.seed_binding is None
                    else {
                        "stage_id": self.sweep.seed_binding[0],
                        "parameter": self.sweep.seed_binding[1],
                    },
                    "evaluation_budget": self.sweep.evaluation_budget,
                },
            },
            "extensions": _thaw(self.extensions),
        }


def _port_dict(port: WorkflowPortType) -> dict[str, object]:
    return {
        "schema": port.schema,
        "dtype": port.dtype,
        "shape": list(port.shape),
        "unit": port.unit,
    }


def _port(value: object) -> WorkflowPortType:
    p = _object(value, "port type", {"schema", "dtype", "shape", "unit"})
    schema = _text(p["schema"], "port schema")
    dtype = _text(p["dtype"], "port dtype")
    if _SCHEMA.fullmatch(schema) is None or dtype not in _DTYPES:
        raise ValueError("unsupported port schema or dtype")
    shape = tuple(
        _integer(d, "port dimension", 0, 4096) for d in _array(p["shape"], "port shape", 4)
    )
    if math.prod(shape) > 4096 or (dtype == "json" and shape):
        raise ValueError("unsupported port shape")
    return WorkflowPortType(schema, dtype, shape, _text(p["unit"], "port unit"))


def _stage(value: object) -> WorkflowStage:
    p = _object(
        value,
        "stage",
        {"id", "adapter", "verb", "backend", "parameters", "inputs", "outputs", "depends_on"},
    )
    adapter, verb = _text(p["adapter"], "adapter"), _text(p["verb"], "verb")
    if adapter not in {"executive", "local-kuramoto"} or verb not in _VERBS:
        raise ValueError("unsupported workflow adapter or verb")
    if adapter == "local-kuramoto" and verb not in {"validate", "simulate", "analyse"}:
        raise ValueError("local classical adapter does not implement this verb")
    parameters = _object(p["parameters"], "parameters")
    inputs = []
    for item in _array(p["inputs"], "inputs", 16):
        q = _object(item, "input", {"parameter", "source_stage", "source_port", "type"})
        inputs.append(
            WorkflowInput(
                _parameter_key(q["parameter"], "input parameter"),
                _id(q["source_stage"], "source stage"),
                _id(q["source_port"], "source port"),
                _port(q["type"]),
            )
        )
    outputs = []
    for item in _array(p["outputs"], "outputs", 16):
        q = _object(item, "output", {"name", "path", "type"})
        path = tuple(
            _text(key, "output path key", 256) for key in _array(q["path"], "output path", 16)
        )
        if not path:
            raise ValueError("nonempty original output path required")
        outputs.append(WorkflowOutput(_id(q["name"], "output name"), path, _port(q["type"])))
    incoming = [item.parameter for item in inputs]
    outgoing = [item.name for item in outputs]
    dependencies = tuple(
        _id(item, "dependency") for item in _array(p["depends_on"], "dependencies", MAX_STAGES)
    )
    if (
        len(set(incoming)) != len(incoming)
        or len(set(outgoing)) != len(outgoing)
        or len(set(dependencies)) != len(dependencies)
    ):
        raise ValueError("duplicate stage port or dependency")
    if set(incoming) & set(parameters):
        raise ValueError("a bound input cannot also have a constant parameter")
    return WorkflowStage(
        _id(p["id"], "stage id"),
        adapter,
        verb,
        _text(p["backend"], "backend"),
        MappingProxyType({key: _freeze(item) for key, item in parameters.items()}),
        tuple(inputs),
        tuple(outputs),
        dependencies,
    )


def _sweep(value: object, stages: tuple[WorkflowStage, ...]) -> WorkflowSweep:
    p = _object(value, "sweep", {"axes", "seeds", "seed_binding", "evaluation_budget"})
    by_id = {stage.id: stage for stage in stages}
    axes = []
    targets: set[tuple[str, str]] = set()
    cells = 1
    for item in _array(p["axes"], "axes", 4):
        q = _object(item, "axis", {"stage_id", "parameter", "values"})
        target = (
            _id(q["stage_id"], "axis stage"),
            _parameter_key(q["parameter"], "axis parameter"),
        )
        values = _array(q["values"], "axis values", 64)
        identities = [canonical_bytes("studio.workflow-axis-value.v1", v) for v in values]
        if not values or len(set(identities)) != len(values) or target in targets:
            raise ValueError("nonempty unique sweep coordinates required")
        if target[0] not in by_id or any(
            i.parameter == target[1] for i in by_id[target[0]].inputs
        ):
            raise ValueError("sweep target is absent or already bound to a source")
        targets.add(target)
        axes.append((target[0], target[1], tuple(_freeze(v) for v in values)))
        cells *= len(values)
    seeds = tuple(_text(v, "seed", 20) for v in _array(p["seeds"], "seeds", 64))
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(re.fullmatch(r"0|[1-9][0-9]{0,19}", s) is None or int(s) > 2**64 - 1 for s in seeds)
    ):
        raise ValueError("unique canonical uint64 seeds required")
    binding = None
    if p["seed_binding"] is not None:
        q = _object(p["seed_binding"], "seed binding", {"stage_id", "parameter"})
        binding = (
            _id(q["stage_id"], "seed stage"),
            _parameter_key(q["parameter"], "seed parameter"),
        )
        if (
            binding in targets
            or binding[0] not in by_id
            or binding[1] in by_id[binding[0]].parameters
            or any(i.parameter == binding[1] for i in by_id[binding[0]].inputs)
        ):
            raise ValueError("seed target is absent or already has a value")
    budget = _integer(p["evaluation_budget"], "evaluation budget", 1, MAX_EVALUATIONS)
    cells *= len(seeds)
    if cells > MAX_CELLS or cells * len(stages) > budget:
        raise ValueError("sweep exceeds cell or stage evaluation budget")
    return WorkflowSweep(tuple(axes), seeds, binding, budget)


def topological_order(definition: WorkflowDefinition) -> tuple[str, ...]:
    """Order admitted stage dependencies by stable ASCII identifiers.

    Parameters
    ----------
    definition
        Original typed graph; data and explicit control edges both apply.

    Returns
    -------
    tuple[str, ...]
        Every stage exactly once, parents before children, lexical ties.

    Raises
    ------
    ValueError
        A dependency is missing or the graph contains a cycle.

    """
    pending = {
        stage.id: set(stage.depends_on) | {p.source_stage for p in stage.inputs}
        for stage in definition.stages
    }
    if not pending or len(pending) != len(definition.stages):
        raise ValueError("nonempty unique workflow stages required")
    if any(not dependencies <= pending.keys() for dependencies in pending.values()):
        raise ValueError("workflow dependency is absent")
    order = []
    while pending:
        ready = sorted(key for key, dependencies in pending.items() if not dependencies)
        if not ready:
            raise ValueError("workflow graph contains a cycle")
        key = ready[0]
        order.append(key)
        del pending[key]
        for dependencies in pending.values():
            dependencies.discard(key)
    return tuple(order)


def parse_workflow(payload: object) -> WorkflowDefinition:
    """Admit an exact bounded graph before a request, worker or save exists.

    Parameters
    ----------
    payload
        Lossless v1 wire object, including original parameters and extensions.

    Returns
    -------
    WorkflowDefinition
        Deeply immutable original definition with a validated acyclic graph.

    Raises
    ------
    ValueError
        Version, field, identity, port, reference, type or resource gate fails.

    """
    text = write_json(payload)
    if len(text.encode("utf-8")) > MAX_WORKFLOW_BYTES:
        raise ValueError("workflow definition exceeds byte bound")
    document = _object(read_json(text), "workflow", {"schema", "body", "extensions"})
    if document["schema"] != WORKFLOW_SCHEMA:
        raise ValueError("unsupported workflow schema")
    body = _object(document["body"], "workflow body", {"workflow_id", "stages", "sweep"})
    stages = tuple(_stage(item) for item in _array(body["stages"], "stages", MAX_STAGES))
    if not stages or len({stage.id for stage in stages}) != len(stages):
        raise ValueError("nonempty unique workflow stages required")
    by_id = {stage.id: stage for stage in stages}
    if sum(len(s.inputs) + len(s.depends_on) for s in stages) > 128:
        raise ValueError("workflow edge bound exceeded")
    for stage in stages:
        for incoming in stage.inputs:
            producer = by_id.get(incoming.source_stage)
            if producer is None:
                raise ValueError("input producer stage is absent")
            output = next((p for p in producer.outputs if p.name == incoming.source_port), None)
            if output is None or output.type != incoming.type:
                raise ValueError("input and original output port types are incompatible")
    extensions = _object(document["extensions"], "extensions")
    definition = WorkflowDefinition(
        _id(body["workflow_id"], "workflow id"),
        stages,
        _sweep(body["sweep"], stages),
        MappingProxyType({key: _freeze(value) for key, value in extensions.items()}),
    )
    topological_order(definition)
    return definition


def validate_port_value(port: WorkflowPortType, value: object) -> object:
    """Check original runtime data against its declared type without converting it.

    Parameters
    ----------
    port
        Admitted exact format, dtype, shape and unit declaration.
    value
        Original finite producer datum; nested numeric arrays retain their dtype.

    Returns
    -------
    object
        An independent immutable value with unchanged scalar types and bits.

    Raises
    ------
    ValueError
        The original value has a different shape, dtype or unsupported data.

    """
    canonical_bytes("studio.workflow-port-value.v1", value)
    _port(_port_dict(port))

    def check(item: object, shape: tuple[int, ...]) -> None:
        if shape:
            if not isinstance(item, (list, tuple)) or len(item) != shape[0]:
                raise ValueError("original port shape differs")
            for element in item:
                check(element, shape[1:])
            return
        valid = (
            port.dtype == "json"
            or (port.dtype == "utf8" and isinstance(item, str))
            or (port.dtype == "bool" and isinstance(item, bool))
            or (port.dtype == "float64" and type(item) is float and math.isfinite(item))
            or (port.dtype == "int64" and type(item) is int and -(2**63) <= item < 2**63)
            or (port.dtype == "uint64" and type(item) is int and 0 <= item < 2**64)
        )
        if not valid:
            raise ValueError("original port dtype differs")

    check(value, port.shape)
    return _freeze(read_json(write_json(value)))
