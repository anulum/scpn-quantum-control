# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts


"""Train-only fitted LLM latent compressor provenance."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

from .model import ModelDescriptor
from .task import SplitManifest
from .wire import (
    COMPRESSOR_SCHEMA,
    ArrayDescriptor,
    ArtifactHeader,
    _digest,
    _positive_int,
    canonical_bytes,
)


def _positive_float_hex(value: object, *, name: str, maximum: float) -> float:
    if type(value) is not str:
        raise ValueError(f"{name} must be canonical finite hex")
    try:
        parsed = float.fromhex(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be canonical finite hex") from exc
    if not math.isfinite(parsed) or not 0 < parsed <= maximum or parsed.hex() != value:
        raise ValueError(f"{name} must be canonical bounded positive hex")
    return parsed


@dataclass(frozen=True, slots=True)
class CompressorArtifact:
    """Bind train-only fitted float32 arrays without embedding private bytes."""

    model_digest: str
    split_digest: str
    train_group_digest: str
    software_digest: str
    fit_seed: int
    fit_method: str
    map_id: str
    scale_policy: str
    input_width: int
    output_width: int
    epsilon_hex: str
    centering: ArrayDescriptor
    projection: ArrayDescriptor
    scales: ArrayDescriptor
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Validate frozen fit metadata, dimensions and array lineage."""
        for name in ("model_digest", "split_digest", "train_group_digest", "software_digest"):
            _digest(getattr(self, name), name=name)
        if type(self.fit_seed) is not int or not 0 <= self.fit_seed < 2**63:
            raise ValueError("invalid compressor fit seed")
        if self.fit_method != "pca_train_svd_v1":
            raise ValueError("unknown compressor fit method")
        if self.map_id != "rms_normalize_center_project_tanh_v2":
            raise ValueError("unknown compressor map")
        if self.scale_policy != "block_zero_variance":
            raise ValueError("unknown compressor scale policy")
        _positive_int(self.input_width, name="compressor input width", maximum=65_536)
        if self.output_width not in (4, 8) or self.output_width > self.input_width:
            raise ValueError("compressor output width must be four or eight")
        _positive_float_hex(self.epsilon_hex, name="compressor epsilon", maximum=1.0)
        for name, shape in (
            ("centering", (self.input_width,)),
            ("projection", (self.input_width, self.output_width)),
            ("scales", (self.output_width,)),
        ):
            descriptor = getattr(self, name)
            if (
                type(descriptor) is not ArrayDescriptor
                or descriptor.dtype != "<f4"
                or descriptor.shape != shape
            ):
                raise ValueError(f"compressor {name} must be exact float32 shape")
        if (
            type(self.header) is not ArtifactHeader
            or self.header.object_kind != "compressor_artifact"
        ):
            raise ValueError("CompressorArtifact requires exact header")
        self.header.validate_content(
            self._scientific_wire(),
            parents=(
                self.model_digest,
                self.split_digest,
                self.train_group_digest,
                self.software_digest,
                self.centering.sha256,
                self.projection.sha256,
                self.scales.sha256,
            ),
        )

    def _scientific_wire(self) -> dict[str, object]:
        """Return exact compressor content hashed by the artifact header."""
        return {
            "schema": COMPRESSOR_SCHEMA,
            "object_kind": "compressor_artifact",
            "model_digest": self.model_digest,
            "split_digest": self.split_digest,
            "train_group_digest": self.train_group_digest,
            "software_digest": self.software_digest,
            "fit_seed": self.fit_seed,
            "fit_method": self.fit_method,
            "map_id": self.map_id,
            "scale_policy": self.scale_policy,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "epsilon_hex": self.epsilon_hex,
            "centering": self.centering.to_wire(),
            "projection": self.projection.to_wire(),
            "scales": self.scales.to_wire(),
        }

    def to_wire(self) -> dict[str, object]:
        """Return detached compressor descriptors and provenance."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> CompressorArtifact:
        """Reject unknown fields and revalidate fitted-array descriptors."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("CompressorArtifact fields mismatch")
        if value["schema"] != COMPRESSOR_SCHEMA or value["object_kind"] != "compressor_artifact":
            raise ValueError("unknown CompressorArtifact schema")
        return cls(
            model_digest=value["model_digest"],
            split_digest=value["split_digest"],
            train_group_digest=value["train_group_digest"],
            software_digest=value["software_digest"],
            fit_seed=value["fit_seed"],
            fit_method=value["fit_method"],
            map_id=value["map_id"],
            scale_policy=value["scale_policy"],
            input_width=value["input_width"],
            output_width=value["output_width"],
            epsilon_hex=value["epsilon_hex"],
            centering=ArrayDescriptor.from_wire(value["centering"]),
            projection=ArrayDescriptor.from_wire(value["projection"]),
            scales=ArrayDescriptor.from_wire(value["scales"]),
            header=ArtifactHeader.from_wire(value["header"]),
        )


def validate_compressor_artifact(
    split: SplitManifest,
    model: ModelDescriptor,
    compressor: CompressorArtifact,
    centering_payload: bytes,
    projection_payload: bytes,
    scales_payload: bytes,
) -> None:
    """Verify exact train lineage, float32 array bytes and positive scales."""
    if (
        type(split) is not SplitManifest
        or type(model) is not ModelDescriptor
        or type(compressor) is not CompressorArtifact
    ):
        raise ValueError("compressor validation requires frozen contracts")
    if compressor.split_digest != hashlib.sha256(canonical_bytes(split.to_wire())).hexdigest():
        raise ValueError("compressor split digest mismatch")
    if compressor.model_digest != hashlib.sha256(canonical_bytes(model.to_wire())).hexdigest():
        raise ValueError("compressor model digest mismatch")
    expected_groups = hashlib.sha256(canonical_bytes(list(split.train_groups))).hexdigest()
    if (
        compressor.train_group_digest != expected_groups
        or split.transform_fit_split != "train_only"
    ):
        raise ValueError("compressor fit groups differ from frozen train split")
    if compressor.input_width != model.hidden_width or model.tensor_dtype != "float32":
        raise ValueError("compressor input differs from model hidden state")
    if compressor.header.data_origin != split.header.data_origin:
        raise ValueError("compressor data origin mismatch")
    for descriptor, payload in (
        (compressor.centering, centering_payload),
        (compressor.projection, projection_payload),
        (compressor.scales, scales_payload),
    ):
        descriptor.validate_payload(payload)
    if any(value[0] <= 0 for value in struct.iter_unpack("<f", scales_payload)):
        raise ValueError("compressor scales must be positive")
