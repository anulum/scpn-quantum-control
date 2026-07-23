# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Studio executive action spine
"""Executive action spine for the standalone SCPN-QUANTUM-CONTROL studio.

The studio is an *executive* tool: it plans a verb invocation, runs it, writes a
standalone reproduction script, and seals the whole thing into an auditable
record. This module is the verb-agnostic spine every action handler plugs into.
The existing federation verb spine (:mod:`scpn_quantum_control.studio.verbs`) is
authoritative over safety: the contract a handler receives is resolved from the
declared :class:`~scpn_studio_platform.verbs.Verb`, so a handler can never widen
its own side effect, safety tier, backend set, or approval requirement.

Lifecycle
---------
``request -> plan -> (approval gate) -> execute -> generate-script -> seal``.

A ``live-hardware`` verb (QPU submission through the provider HAL) or a
``certified`` verb requires an explicit approval on the request; without it the
spine returns a ``gated`` record and never executes — the fail-closed contract
for deploying onto real endpoints. Read-only research verbs (e.g. ``differentiate``)
execute directly.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from .verbs import QUANTUM_VERBS

ActionStatus = Literal["succeeded", "failed", "gated"]
ScriptLanguage = Literal["python"]

_LIVE_HARDWARE_SIDE_EFFECT: Final[str] = "LIVE_HARDWARE"
_CERTIFIED_SAFETY_TIER: Final[str] = "CERTIFIED"


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VerbContract:
    """The safety and capability contract a verb imposes on its handler.

    Parameters
    ----------
    verb : str
        The verb name (matches a :data:`~scpn_quantum_control.studio.verbs.QUANTUM_VERBS`
        entry).
    side_effect : str
        The declared side-effect class (e.g. ``"READ_ONLY"``, ``"SIMULATED"``,
        ``"LIVE_HARDWARE"``).
    safety_tier : str
        The declared safety tier (e.g. ``"RESEARCH"``, ``"CERTIFIED"``).
    requires_approval : bool
        Whether an executive request must carry an explicit approval before the
        spine will execute. ``True`` for live-hardware or certified verbs.
    backends : tuple of str
        The backends the verb may dispatch to; a plan backend outside this set
        is rejected.
    produces : tuple of str
        The ``studio.*.v1`` evidence-schema families this verb feeds the
        informative federation layer.

    """

    verb: str
    side_effect: str
    safety_tier: str
    requires_approval: bool
    backends: tuple[str, ...]
    produces: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready contract description."""
        return {
            "verb": self.verb,
            "side_effect": self.side_effect,
            "safety_tier": self.safety_tier,
            "requires_approval": self.requires_approval,
            "backends": list(self.backends),
            "produces": list(self.produces),
        }


def resolve_verb_contract(verb: str) -> VerbContract:
    """Resolve the authoritative contract for ``verb`` from the verb spine.

    Parameters
    ----------
    verb : str
        The verb name to resolve.

    Returns
    -------
    VerbContract
        The resolved contract.

    Raises
    ------
    KeyError
        If no verb with that name is declared on the federation contract.

    """
    for declared in QUANTUM_VERBS:
        if declared.name == verb:
            side_effect = declared.side_effect.name
            safety_tier = declared.safety_tier.name
            requires_approval = (
                side_effect == _LIVE_HARDWARE_SIDE_EFFECT or safety_tier == _CERTIFIED_SAFETY_TIER
            )
            return VerbContract(
                verb=declared.name,
                side_effect=side_effect,
                safety_tier=safety_tier,
                requires_approval=requires_approval,
                backends=tuple(declared.backends),
                produces=tuple(declared.produces),
            )
    raise KeyError(f"no studio verb named {verb!r} on the federation contract")


@dataclass(frozen=True)
class ExecutiveRequest:
    """A validated request to run one studio verb executively.

    Parameters
    ----------
    verb : str
        The verb to run.
    action_id : str
        Stable identifier for this action (used in the sealed record entity id).
    parameters : Mapping
        Verb-specific, JSON-serialisable parameters interpreted by the handler.
    backend : str or None, optional
        Requested backend; ``None`` lets the handler pick its default from the
        contract's backend set.
    approved : bool, optional
        Explicit approval for a gated (live-hardware or certified) verb.

    """

    verb: str
    action_id: str
    parameters: Mapping[str, Any]
    backend: str | None = None
    approved: bool = False

    def __post_init__(self) -> None:
        """Reject malformed or ambiguous executive requests."""
        if not self.verb:
            raise ValueError("verb must be non-empty")
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a mapping")
        if self.backend is not None and not self.backend:
            raise ValueError("backend must be non-empty when provided")
        if not isinstance(self.approved, bool):
            raise ValueError("approved must be a boolean")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready request description."""
        return {
            "verb": self.verb,
            "action_id": self.action_id,
            "parameters": dict(self.parameters),
            "backend": self.backend,
            "approved": self.approved,
        }


@dataclass(frozen=True)
class ExecutionPlan:
    """An inspectable plan a handler produces before anything runs.

    Parameters
    ----------
    verb : str
        The verb being planned.
    action_id : str
        The action identifier from the request.
    backend : str
        The resolved backend (validated against the contract).
    contract : VerbContract
        The authoritative verb contract this plan was resolved under.
    claim_boundary : str
        The honest boundary of what executing this plan proves.
    steps : tuple of str
        Human-readable ordered steps the execution will take.
    parameters : Mapping
        Normalised, validated parameters the execution consumes.

    """

    verb: str
    action_id: str
    backend: str
    contract: VerbContract
    claim_boundary: str
    steps: tuple[str, ...]
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Enforce plan consistency with the resolved verb contract."""
        if self.verb != self.contract.verb:
            raise ValueError("plan verb must match its contract verb")
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if self.backend not in self.contract.backends:
            raise ValueError(f"backend {self.backend!r} is not declared for verb {self.verb!r}")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        if not self.steps:
            raise ValueError("steps must be non-empty")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a mapping")

    @property
    def requires_approval(self) -> bool:
        """Return whether executing this plan requires an explicit approval."""
        return self.contract.requires_approval

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready plan description."""
        return {
            "verb": self.verb,
            "action_id": self.action_id,
            "backend": self.backend,
            "contract": self.contract.to_dict(),
            "claim_boundary": self.claim_boundary,
            "steps": list(self.steps),
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class ExecutionResult:
    """The outcome of running a plan.

    Parameters
    ----------
    status : str
        ``"succeeded"``, ``"failed"``, or ``"gated"``.
    outputs : Mapping
        Verb-specific, JSON-serialisable result payload.
    error : str or None, optional
        Failure or gating reason when ``status`` is not ``"succeeded"``.

    """

    status: ActionStatus
    outputs: Mapping[str, Any]
    error: str | None = None

    def __post_init__(self) -> None:
        """Enforce the status, output, and error coupling contract."""
        if self.status not in ("succeeded", "failed", "gated"):
            raise ValueError("status must be succeeded, failed, or gated")
        if not isinstance(self.outputs, Mapping):
            raise ValueError("outputs must be a mapping")
        if self.status == "succeeded":
            if self.error is not None:
                raise ValueError("a succeeded result must not carry an error")
        elif not self.error:
            raise ValueError("a non-succeeded result must carry an error reason")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready result description."""
        return {
            "status": self.status,
            "outputs": dict(self.outputs),
            "error": self.error,
        }


@dataclass(frozen=True)
class GeneratedScript:
    """A standalone reproduction script the studio wrote for an action.

    Parameters
    ----------
    language : str
        The script language (``"python"``).
    filename : str
        Suggested file name for the script.
    entrypoint : str
        The shell command that runs the script.
    source : str
        The full script source.
    digest : str
        Content digest (``sha256:...``) over the source.

    """

    language: ScriptLanguage
    filename: str
    entrypoint: str
    source: str
    digest: str

    def __post_init__(self) -> None:
        """Validate the reproduction script and its content digest."""
        if self.language != "python":
            raise ValueError("language must be python")
        if not self.filename.endswith(".py"):
            raise ValueError("filename must end with .py")
        if not self.entrypoint:
            raise ValueError("entrypoint must be non-empty")
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        expected = _digest(self.source)
        if self.digest != expected:
            raise ValueError("digest must be the sha256 of source")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready script description."""
        return {
            "language": self.language,
            "filename": self.filename,
            "entrypoint": self.entrypoint,
            "source": self.source,
            "digest": self.digest,
        }


def build_generated_script(
    *, filename: str, entrypoint: str, source: str, language: ScriptLanguage = "python"
) -> GeneratedScript:
    """Build a :class:`GeneratedScript`, computing its content digest.

    Parameters
    ----------
    filename : str
        Suggested file name (must end with ``.py``).
    entrypoint : str
        Shell command that runs the script.
    source : str
        The full script source.
    language : str, optional
        The script language (``"python"``).

    Returns
    -------
    GeneratedScript
        The sealed script with its digest attached.

    """
    return GeneratedScript(
        language=language,
        filename=filename,
        entrypoint=entrypoint,
        source=source,
        digest=_digest(source),
    )


@dataclass(frozen=True)
class ExecutiveRecord:
    """The sealed, auditable record of one executive action.

    Parameters
    ----------
    request : ExecutiveRequest
        The originating request.
    plan : ExecutionPlan
        The plan that was resolved.
    result : ExecutionResult
        The execution outcome.
    script : GeneratedScript or None
        The reproduction script (``None`` when the action was gated or failed
        before a script could be written).
    digest : str
        Content digest over ``request``, ``plan``, ``result``, and ``script``.

    """

    request: ExecutiveRequest
    plan: ExecutionPlan
    result: ExecutionResult
    script: GeneratedScript | None
    digest: str

    def __post_init__(self) -> None:
        """Validate the sealed record and its script/status coupling."""
        expected = _digest(
            {
                "request": self.request.to_dict(),
                "plan": self.plan.to_dict(),
                "result": self.result.to_dict(),
                "script": None if self.script is None else self.script.to_dict(),
            }
        )
        if self.digest != expected:
            raise ValueError("digest must seal request, plan, result, and script")
        if self.result.status == "succeeded" and self.script is None:
            raise ValueError("a succeeded action must carry a reproduction script")
        if self.result.status != "succeeded" and self.script is not None:
            raise ValueError("only a succeeded action may carry a script")

    @property
    def produced_schemas(self) -> tuple[str, ...]:
        """Return the ``studio.*.v1`` families this action feeds the hub."""
        return self.plan.contract.produces

    @property
    def claim_boundary(self) -> str:
        """Return the plan's claim boundary."""
        return self.plan.claim_boundary

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready record description."""
        return {
            "request": self.request.to_dict(),
            "plan": self.plan.to_dict(),
            "result": self.result.to_dict(),
            "script": None if self.script is None else self.script.to_dict(),
            "digest": self.digest,
            "produced_schemas": list(self.produced_schemas),
        }


class ActionHandler(ABC):
    """Verb-specific executive behaviour the spine orchestrates.

    A handler declares one :attr:`verb`, plans a request under the resolved
    contract, executes the plan, and writes a standalone reproduction script.
    """

    @property
    @abstractmethod
    def verb(self) -> str:
        """Return the verb this handler implements."""

    @abstractmethod
    def plan(self, request: ExecutiveRequest, contract: VerbContract) -> ExecutionPlan:
        """Return an inspectable plan for ``request`` under ``contract``.

        Parameters
        ----------
        request : ExecutiveRequest
            The request to plan.
        contract : VerbContract
            The authoritative verb contract.

        Returns
        -------
        ExecutionPlan
            The resolved plan.

        """

    @abstractmethod
    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """Run ``plan`` and return its outcome.

        Parameters
        ----------
        plan : ExecutionPlan
            The plan to execute.

        Returns
        -------
        ExecutionResult
            The execution outcome.

        """

    @abstractmethod
    def generate_script(self, plan: ExecutionPlan, result: ExecutionResult) -> GeneratedScript:
        """Return a standalone script that reproduces the action.

        Parameters
        ----------
        plan : ExecutionPlan
            The executed plan.
        result : ExecutionResult
            The succeeded result to reproduce.

        Returns
        -------
        GeneratedScript
            The reproduction script.

        """


@dataclass
class ActionRegistry:
    """A registry of executive action handlers keyed by verb."""

    _handlers: dict[str, ActionHandler] = field(default_factory=dict)

    def register(self, handler: ActionHandler) -> None:
        """Register ``handler`` under its verb.

        Parameters
        ----------
        handler : ActionHandler
            The handler to register.

        Raises
        ------
        ValueError
            If the verb is unknown to the federation contract or already
            registered.

        """
        try:
            resolve_verb_contract(handler.verb)
        except KeyError as exc:
            raise ValueError(
                f"cannot register a handler for the unknown verb {handler.verb!r}"
            ) from exc
        if handler.verb in self._handlers:
            raise ValueError(f"a handler for verb {handler.verb!r} is already registered")
        self._handlers[handler.verb] = handler

    def resolve(self, verb: str) -> ActionHandler:
        """Return the handler registered for ``verb``.

        Parameters
        ----------
        verb : str
            The verb to resolve.

        Returns
        -------
        ActionHandler
            The registered handler.

        Raises
        ------
        KeyError
            If no handler is registered for the verb.

        """
        if verb not in self._handlers:
            raise KeyError(f"no executive handler registered for verb {verb!r}")
        return self._handlers[verb]

    def verbs(self) -> tuple[str, ...]:
        """Return the registered verbs in sorted order."""
        return tuple(sorted(self._handlers))


def preview_action(request: ExecutiveRequest, *, registry: ActionRegistry) -> ExecutionPlan:
    """Resolve and return the plan for ``request`` without executing it.

    Parameters
    ----------
    request : ExecutiveRequest
        The request to plan.
    registry : ActionRegistry
        The handler registry.

    Returns
    -------
    ExecutionPlan
        The inspectable plan.

    """
    handler = registry.resolve(request.verb)
    contract = resolve_verb_contract(request.verb)
    return handler.plan(request, contract)


def run_action(request: ExecutiveRequest, *, registry: ActionRegistry) -> ExecutiveRecord:
    """Plan, gate, execute, script, and seal ``request`` into a record.

    A plan that requires approval on an unapproved request yields a ``gated``
    record and never executes. A handler that raises during execution yields a
    ``failed`` record. A succeeded execution is scripted and sealed.

    Parameters
    ----------
    request : ExecutiveRequest
        The request to run.
    registry : ActionRegistry
        The handler registry.

    Returns
    -------
    ExecutiveRecord
        The sealed executive record.

    """
    handler = registry.resolve(request.verb)
    contract = resolve_verb_contract(request.verb)
    plan = handler.plan(request, contract)

    if plan.requires_approval and not request.approved:
        result = ExecutionResult(
            status="gated",
            outputs={},
            error=(
                f"verb {plan.verb!r} is {plan.contract.side_effect}/"
                f"{plan.contract.safety_tier} and requires an explicit approval"
            ),
        )
        return _seal(request, plan, result, script=None)

    try:
        result = handler.execute(plan)
    except Exception as exc:  # noqa: BLE001 - surfaced as a fail-closed record
        result = ExecutionResult(status="failed", outputs={}, error=f"{type(exc).__name__}: {exc}")
        return _seal(request, plan, result, script=None)

    if result.status != "succeeded":
        return _seal(request, plan, result, script=None)

    script = handler.generate_script(plan, result)
    return _seal(request, plan, result, script=script)


def _seal(
    request: ExecutiveRequest,
    plan: ExecutionPlan,
    result: ExecutionResult,
    *,
    script: GeneratedScript | None,
) -> ExecutiveRecord:
    digest = _digest(
        {
            "request": request.to_dict(),
            "plan": plan.to_dict(),
            "result": result.to_dict(),
            "script": None if script is None else script.to_dict(),
        }
    )
    return ExecutiveRecord(
        request=request,
        plan=plan,
        result=result,
        script=script,
        digest=digest,
    )


__all__ = [
    "ActionHandler",
    "ActionRegistry",
    "ActionStatus",
    "ExecutionPlan",
    "ExecutionResult",
    "ExecutiveRecord",
    "ExecutiveRequest",
    "GeneratedScript",
    "ScriptLanguage",
    "VerbContract",
    "build_generated_script",
    "preview_action",
    "resolve_verb_contract",
    "run_action",
]
