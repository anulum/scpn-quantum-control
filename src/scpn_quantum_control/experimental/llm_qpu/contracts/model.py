# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts


"""Frozen model and hidden-state tap identity."""

from __future__ import annotations

from dataclasses import dataclass

from .wire import MODEL_SCHEMA, ArtifactHeader, _digest, _positive_int, _text


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Design-only identity for a future locally probed hidden-state source."""

    model_id: str
    checkpoint_digest: str
    tokenizer_digest: str
    chat_template_digest: str
    runtime_build_digest: str
    loader_id: str
    quantization: str
    tensor_dtype: str
    block_count: int
    hidden_width: int
    tap_block_index: int
    tap_stream: str
    tap_boundary: str
    probe_evidence_digest: str
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse inferred dimensions, completion-only taps and missing lineage."""
        for name in ("model_id", "loader_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        for name in (
            "checkpoint_digest",
            "tokenizer_digest",
            "chat_template_digest",
            "runtime_build_digest",
            "probe_evidence_digest",
        ):
            _digest(getattr(self, name), name=name)
        if self.quantization not in ("none", "gguf_q8_0", "gguf_q6_k", "gguf_q4_k_m"):
            raise ValueError("unknown model quantization")
        if self.tensor_dtype not in ("float16", "float32", "bfloat16"):
            raise ValueError("unknown hidden-state dtype")
        _positive_int(self.block_count, name="model block count", maximum=1024)
        _positive_int(self.hidden_width, name="observed hidden width", maximum=65_536)
        if (
            type(self.tap_block_index) is not int
            or not 0 <= self.tap_block_index < self.block_count
        ):
            raise ValueError("tap block index must be within observed model blocks")
        if self.tap_stream != "residual_hidden_state":
            raise ValueError("completion or embedding fallback is not a hidden-state tap")
        if self.tap_boundary not in ("before_norm", "after_norm"):
            raise ValueError("tap normalization boundary must be explicit")
        if (
            type(self.header) is not ArtifactHeader
            or self.header.object_kind != "model_descriptor"
        ):
            raise ValueError("ModelDescriptor requires its exact artifact header")
        if self.header.data_origin != "owner_checkpoint":
            raise ValueError("ModelDescriptor requires owner-checkpoint origin")
        self.header.validate_content(
            self._scientific_wire(),
            parents=(
                self.checkpoint_digest,
                self.tokenizer_digest,
                self.chat_template_digest,
                self.runtime_build_digest,
                self.probe_evidence_digest,
            ),
        )

    def _scientific_wire(self) -> dict[str, object]:
        """Return only the model identity and tap semantics hashed as content."""
        return {
            "schema": MODEL_SCHEMA,
            "object_kind": "model_descriptor",
            **{
                name: getattr(self, name) for name in self.__dataclass_fields__ if name != "header"
            },
        }

    def to_wire(self) -> dict[str, object]:
        """Return detached versioned model metadata, with no weights or prompts."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> ModelDescriptor:
        """Reject unknown fields and restore the frozen model descriptor."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("ModelDescriptor fields mismatch")
        if value["schema"] != MODEL_SCHEMA or value["object_kind"] != "model_descriptor":
            raise ValueError("unknown ModelDescriptor schema")
        return cls(
            **{name: value[name] for name in cls.__dataclass_fields__ if name != "header"},
            header=ArtifactHeader.from_wire(value["header"]),
        )
