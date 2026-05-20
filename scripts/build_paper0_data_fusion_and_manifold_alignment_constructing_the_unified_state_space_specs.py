#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Data Fusion and Manifold Alignment: Constructing the Unified State Space spec builder
"""Promote Paper 0 Data Fusion and Manifold Alignment: Constructing the Unified State Space records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = (
    "P0R04168",
    "P0R04169",
    "P0R04170",
    "P0R04171",
    "P0R04172",
    "P0R04173",
    "P0R04174",
    "P0R04175",
    "P0R04176",
    "P0R04177",
    "P0R04178",
    "P0R04179",
    "P0R04180",
    "P0R04181",
    "P0R04182",
    "P0R04183",
    "P0R04184",
    "P0R04185",
    "P0R04186",
    "P0R04187",
    "P0R04188",
    "P0R04189",
    "P0R04190",
    "P0R04191",
    "P0R04192",
    "P0R04193",
    "P0R04194",
    "P0R04195",
    "P0R04196",
    "P0R04197",
    "P0R04198",
    "P0R04199",
    "P0R04200",
    "P0R04201",
    "P0R04202",
    "P0R04203",
    "P0R04204",
    "P0R04205",
    "P0R04206",
    "P0R04207",
    "P0R04208",
    "P0R04209",
    "P0R04210",
    "P0R04211",
    "P0R04212",
    "P0R04213",
    "P0R04214",
    "P0R04215",
)
CLAIM_BOUNDARY = "source-bounded data fusion and manifold alignment constructing the unified state space source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "data_fusion_and_manifold_alignment_constructing_the_unified_state_space.data_fusion_and_manifold_alignment_constructing_the_unified_state_space": {
        "context_id": "data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
        "validation_protocol": "paper0.data_fusion_and_manifold_alignment_constructing_the_unified_state_space.data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
        "canonical_statement": "The source-bounded component 'Data Fusion and Manifold Alignment: Constructing the Unified State Space' preserves Paper 0 records P0R04168-P0R04215 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04168:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04169:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04170:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04171:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04172:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04173:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04174:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04175:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04176:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04177:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04178:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04179:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04180:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04181:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04182:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04183:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04184:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04185:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04186:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04187:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04188:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04189:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04190:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04191:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04192:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04193:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04194:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04195:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04196:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04197:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04198:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04199:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04200:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04201:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04202:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04203:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04204:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04205:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04206:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04207:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04208:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04209:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04210:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04211:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04212:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04213:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04214:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
            "P0R04215:data_fusion_and_manifold_alignment_constructing_the_unified_state_space",
        ),
        "source_formulae": (
            "P0R04168: Data Fusion and Manifold Alignment: Constructing the Unified State Space",
            "P0R04169: P0R04169",
            "P0R04170: A critical methodological challenge in operationalizing Qualia Capacity ($Q$) is the dimensional and temporal mismatch of the requisite neurophysiological data. To capture both the fast, high-frequency intentional dynamics (e.g., EEG at $\\sim 1000$ Hz) and the slow, deeply integrated affective/metabolic states (e.g., fMRI at $\\sim 0.5$ Hz and HRV), we must fuse distinct data streams. However, these streams cannot be naively concatenated into a single point cloud, as they exist in entirely different physical coordinate systems and timescales.",
            "P0R04171: To rigorously construct the unified Consciousness Manifold ($\\mathcal{M}$), the SCPN framework employs a formal two-step geometric fusion process: Takens' Delay Embedding followed by Diffeomorphic Manifold Alignment.",
            "P0R04172: 1. Phase Space Reconstruction (Takens' Theorem)",
            "P0R04173: First, the fast temporal dynamics (EEG and HRV) must be lifted from 1D time series into fully realized geometric objects. According to Takens' Delay Embedding Theorem, the hidden dynamics of a chaotic system can be reconstructed from a sequence of observations of a single state variable. For a given time series $x(t)$, we construct a $d$-dimensional delay vector:",
            "P0R04174: $$\\mathbf{v}(t) = [x(t), x(t-\\tau), \\dots, x(t-(d-1)\\tau)]$$",
            "P0R04175: where $\\tau$ is the optimal time delay (determined via the first minimum of the average mutual information) and $d$ is the embedding dimension (determined via the False Nearest Neighbors method). Applying this to the multichannel electrophysiological data yields a high-dimensional, fast-temporal manifold, $\\mathcal{M}_{fast}$.",
            "P0R04176: 2. Diffeomorphic Manifold Alignment",
            "P0R04177: Simultaneously, the fMRI BOLD signals define a distinct, sluggish spatial manifold, $\\mathcal{M}_{slow}$, representing the large-scale metabolic and neurovascular network state. To fuse $\\mathcal{M}_{fast}$ and $\\mathcal{M}_{slow}$, we project both into a shared, lower-dimensional latent space $\\mathcal{Y}$.",
            "P0R04178: This is achieved via Semi-Supervised Manifold Alignment, utilizing Laplacian Eigenmaps to preserve the local geometry of both original manifolds while minimizing the Procrustes distance between their temporally synchronized anchor points (the discrete stimulus onset times or heartbeat R-R peaks). We find projection matrices $W_{fast}$ and $W_{slow}$ that minimize the alignment cost function:",
            "P0R04179: $$\\mathcal{C}(W_{fast}, W_{slow}) = \\mu \\sum_{i} ||W_{fast}^T \\mathbf{v}_{fast}^{(i)} - W_{slow}^T \\mathbf{v}_{slow}^{(i)}||^2 + \\text{Geometry Preservation Terms}$$",
            "P0R04180: The resulting shared latent space $\\mathcal{Y}$ is the rigorously defined Consciousness Manifold ($\\mathcal{M}$). It is strictly upon this geometrically fused, timescale-invariant manifold that the Vietoris-Rips filtration is applied and the persistent homology (the Betti numbers, $b_k$) is calculated to derive the final Qualia Capacity, $Q(t)$.",
            "P0R04181: P0R04181",
            "P0R04182: We formalize the Qualia Capacity Q(t)Q(t)Q(t) using Topological Data Analysis (TDA) applied to momentary neural population activity (Layer 5). For a time window centered at ttt, let",
            "P0R04183: $\\mathbf{X}\\left( \\mathbf{t} \\right)\\mathbf{= xi}\\left( \\mathbf{\\tau} \\right)\\mathbf{i = 1}\\mathbf{Nc}\\mathbf{\\subset}\\mathbf{Rp}\\mathcal{X}\\left( \\mathbf{t} \\right)\\mathbf{=}\\textit{\\textbf{\\{}}\\mathbf{x}_{\\mathbf{i}}\\left( \\mathbf{\\tau} \\right)\\textit{\\textbf{\\}}}_{\\mathbf{i = 1}}^{\\mathbf{N}_{\\mathbf{c}}}\\mathbf{\\subset}\\mathbf{R}^{\\mathbf{p}}\\mathbf{X}\\left( \\mathbf{t} \\right)\\mathbf{= xi}\\left( \\mathbf{\\tau} \\right)\\mathbf{i = 1}\\mathbf{Nc}\\mathbf{\\subset}\\mathbf{Rp}$",
            "P0R04184: denote the point cloud of neural states (channels or parcels i=1,,Nci=1,\\dots,N_ci=1,,Nc; ppp features per state) extracted from band-limited activity, embeddings, or microstate centroids. Using a metric dij(t)d_{ij}(t)dij(t) (e.g., correlation distance dij=2(1rhoij)d_{ij}=\\sqrt{2(1-\\rho_{ij})}dij=2(1rhoij)), we build a filtered simplicial complex (e.g., Vietoris-Rips) and compute persistent homology {Hk}k=0D\\{H_k\\}_{k=0}^{D}{Hk}k=0D, yielding birth-death pairs Dk(t)={(bi(k),di(k))}\\mathcal{D}_k(t)=\\{(b_i^{(k)},d_i^{(k)})\\}Dk(t)={(bi(k),di(k))} and Betti numbers k(t)\\beta_k(t)k(t) that count connected components (k=0k=0k=0), loops (k=1k=1k=1), voids (k=2k=2k=2), etc.",
            "P0R04185: We define Q(t)Q(t)Q(t) as a weighted, persistence-aware topological richness that balances integration (low 0\\beta_00) and differentiation (nontrivial higher-dimensional structure for k>0k>0k>0):",
            "P0R04186: $Q(t)\\ \\mspace{2mu} = \\ \\mspace{2mu} 11 + \\beta 0(t)\\text{/}Nc\\overset{i}{}ntegration\\ factor\\ I(t)\\ \\mspace{2mu} \\cdot \\ \\mspace{2mu}\\sum_{}^{}k = 1Dwk\\ \\mspace{2mu}\\Pi k(t;\\alpha)\\overset{d}{}ifferentiation\\ via\\ persistence$",
            "P0R04187: $\\boxed{\\ Q(t)\\ = \\ \\underset{\\text{integration factor }I(t)}{\\overset{\\frac{1}{1 + \\beta_{0}(t)\\text{/}N_{c}}}{}}\\ \\cdot \\ \\underset{\\text{differentiation via persistence}}{\\overset{\\sum_{k = 1}^{D}w_{k}\\ \\Pi_{k}(t;\\alpha)}{}}}$",
            "P0R04188: $Q(t) = integration\\ factor\\ I(t)1 + \\beta 0(t)\\text{/}Nc 1 \\cdot differentiation\\ via\\ persistencek = 1\\sum_{}^{}{D wk\\Pi k(t;\\alpha)}$",
            "P0R04189: with",
            "P0R04190: $\\Pi k(t;\\alpha)\\ \\mspace{2mu} = \\ \\mspace{2mu}\\sum_{}^{}(b,d) \\in Dk(t)\\,\\,(d - b\\Lambda)\\,\\alpha,\\alpha \\geq 1,\\Pi_{k}(t;\\alpha)\\ = \\ \\sum_{(b,d) \\in \\mathcal{D}_{\\mathcal{k}}(t)}^{}{\\text{!!}\\left( \\frac{d - b}{\\Lambda} \\right)^{\\text{!}\\alpha}},\\quad\\quad$",
            "P0R04191: $\\alpha \\geq 1,\\Pi k(t;\\alpha) = (b,d) \\in Dk(t)\\sum_{}^{}{(\\Lambda d - b)\\alpha},\\alpha \\geq 1$,",
            "P0R04192: where \\Lambda is a scale normalizer (e.g., the 95th percentile of lifetimes across the session) ensuring QQQ is dimensionless and robust to overall signal scale. The integration factor I(t)(0,1]I(t)\\in(0,1]I(t)(0,1] penalises fragmentation, while k\\Pi_kk sums lifetimes (persistence) of kkk-dimensional features, suppressing spurious, short-lived holes. The weights wk0w_k\\ge 0wk0 prioritize topological dimensions (e.g., default w1=1, w2=[0,1], wk>2=0w_1=1,\\; w_2=\\eta\\in[0,1],\\; w_{k>2}=0w1=1,w2=[0,1],wk>2=0 unless data support higher kkk), and \\alpha (default =1\\alpha=1=1) controls emphasis on longer-lived features.",
            "P0R04193: For transparency and compatibility with simpler summaries, we also report the Betti-sum proxy",
            "P0R04194: $QBetti(t)\\ \\mspace{2mu} = \\ \\mspace{2mu}\\sum_{}^{}k = 0Dw\\sim k\\,\\beta k(t),Q_{\\text{Betti}}(t)\\ = \\ \\sum_{k = 0}^{D}\\widetilde{w_{k}}\\,\\beta_{k}(t),QBetti(t) = k = 0\\sum_{}^{}{D w}\\sim k\\beta k(t),$",
            "P0R04195: with w~0<0\\tilde{w}_0<0w~0<0 (penalizing disintegration) and w~k>0>0\\tilde{w}_{k>0}>0w~k>0>0. In practice Q(t)Q(t)Q(t) (persistence-aware) is the primary metric; QBetti(t)Q_{\\text{Betti}}(t)QBetti(t) serves as a fast diagnostic.",
            "P0R04196: Properties and rationale.",
            "P0R04197: (i) Computability: Q(t)Q(t)Q(t) follows a standard TDA pipeline (metric selection, filtration, persistent homology, lifetime aggregation).",
            "P0R04198: (ii) Stability: by the stability of persistent diagrams under the bottleneck/Wasserstein metrics, Q(t)Q(t)Q(t) is Lipschitz in the input geometry (with constants set by ,,wk\\alpha,\\Lambda,w_k,,wk), ensuring robustness to moderate noise.",
            "P0R04199: (iii) Scale-free structure: normalising by \\Lambda and window-wise z-scoring features mitigates dependence on absolute amplitudes. (iv) Interpretation: high QQQ occurs when neural states are globally integrated (few components) yet richly differentiated (many persistent loops/voids), aligning with phenomenology of unified yet vivid experience.",
            "P0R04200: Link to the Ethical Lagrangian. With",
            "P0R04201: $LEthical = WC\\, C + WK\\, K + WQ\\, QL_{\\text{Ethical}} = W_{C}\\, C + W_{K}\\, K + W_{Q}\\, QLEthical = WC C + WK K + WQ Q,$",
            'P0R04202: this construction makes the third term computable and calibratable. Moreover, maximizing QQQ is consistent with the Causal Entropic Principle: topologies that support many long-lived meso-scale motifs (large k\\Pi_kk) generically admit more accessible future histories (greater reconfigurability), thus increasing causal efficacy and adaptability-our operational reframing of "ethical" drive.',
            "P0R04203: Implementation notes (default choices). Window length 1 51\\!-\\!515 s (task-dependent), overlap 50%50\\%50%; metric dijd_{ij}dij from debiased correlation or phase-synchrony; Vietoris-Rips filtration up to D=2D=2D=2 unless high-SNR data support k=3k=3k=3; persistence thresholding by lifetime or statistical bootstraps; w1=1, w2=0.5, =1, w_1=1,\\; w_2=0.5,\\; \\alpha=1,\\; \\Lambdaw1=1,w2=0.5,=1, as session-level lifetime scale. Report Q(t)Q(t)Q(t), its distribution over time, and DeltaQ\\Delta QDeltaQ under interventions relevant to Sustainable Ethical Coherence.",
            "P0R04204: Steps",
            "P0R04205: Define neural state cloud X(t)\\mathcal{X}(t)X(t) in a sliding window; choose metric dijd_{ij}dij. | Build a filtered complex (e.g., Vietoris-Rips); compute Dk(t)\\mathcal{D}_k(t)Dk(t) and k(t)\\beta_k(t)k(t). | Aggregate lifetimes via k(t;)\\Pi_k(t;\\alpha)k(t;); compute",
            "P0R04206: I$(t) = 1\\text{/}\\left( 1 + \\beta 0(t)\\text{/}Nc \\right)I(t) = 1\\text{/}\\left( 1 + \\beta_{0}(t)\\text{/}N_{c} \\right)I(t) = 1\\text{/}\\left( 1 + \\beta 0(t)\\text{/}Nc \\right).$",
            "P0R04207: Evaluate $Q(t) = I(t)\\sum_{}^{}k \\geq 1wk\\Pi k(t;\\alpha)Q(t) = I(t)\\sum_{k \\geq 1}^{}{w_{k}\\Pi_{k}(t;\\alpha)Q(t)} = I(t)\\sum_{}^{}k \\geq 1 wk\\Pi k(t;\\alpha)$",
            "P0R04208: (and optional QBetti(t)Q_{\\text{Betti}}(t)QBetti(t)).",
            "P0R04209: Normalize and report QQQ across conditions; plug into LEthicalL_{\\text{Ethical}}LEthical via WQW_QWQ.",
            "P0R04210: Index",
            "P0R04211: Qualia Capacity QQQ; persistent homology; Betti numbers k\\beta_kk; lifetimes (db)(d-b)(db); Vietoris-Rips; integration factor I(t)I(t)I(t); k(t;)\\Pi_k(t;\\alpha)k(t;); Ethical Lagrangian LEthicalL_{\\text{Ethical}}LEthical; Causal Entropic Principle; Sustainable Ethical Coherence.",
            'P0R04212: This TDA-based construction turns Qualia Capacity into a rigorous, noise-stable, and computable scalar that operationalises experiential richness as "integrated differentiation." It slots cleanly into the Ethical Lagrangian, aligning the drive to maximise SEC with the physical imperative to find adaptable, causally potent, and topologically rich states of being.',
            "P0R04213: The Principle of Ethical Least Action (PELA)",
            "P0R04214: Source Material: The description of the variational principle that governs the universe's trajectory through state space. The system evolves along the path that extremises the Ethical Functional, analogous to the Principle of Least Action in classical mechanics.",
            "P0R04215: P0R04215",
        ),
        "test_protocols": (
            "preserve Data Fusion and Manifold Alignment: Constructing the Unified State Space source-accounting boundary",
        ),
        "null_results": (
            "Data Fusion and Manifold Alignment: Constructing the Unified State Space is not empirical validation evidence",
        ),
        "variables": ("data_fusion_and_manifold_alignment_constructing_the_unified_state_space",),
        "validation_targets": ("preserve records P0R04168-P0R04215",),
        "null_controls": (
            "data_fusion_and_manifold_alignment_constructing_the_unified_state_space must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class DataFusionAndManifoldAlignmentConstructingTheUnifiedStateSpaceSpec:
    """Spec promoted from Paper 0 source records."""

    key: str
    context_id: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    source_formulae: tuple[str, ...]
    test_protocols: tuple[str, ...]
    null_results: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class DataFusionAndManifoldAlignmentConstructingTheUnifiedStateSpaceSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[DataFusionAndManifoldAlignmentConstructingTheUnifiedStateSpaceSpec, ...]
    summary: dict[str, Any]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL ledger into dictionaries."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def build_data_fusion_and_manifold_alignment_constructing_the_unified_state_space_specs(
    source_records: list[dict[str, Any]],
) -> DataFusionAndManifoldAlignmentConstructingTheUnifiedStateSpaceSpecBundle:
    """Build source-covered specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[DataFusionAndManifoldAlignmentConstructingTheUnifiedStateSpaceSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            DataFusionAndManifoldAlignmentConstructingTheUnifiedStateSpaceSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 foundational extraction",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(
                    int(record["source_block_index"]) for record in anchors
                ),
                source_formulae=tuple(metadata["source_formulae"]),
                test_protocols=tuple(metadata["test_protocols"]),
                null_results=tuple(metadata["null_results"]),
                variables=tuple(metadata["variables"]),
                validation_targets=tuple(metadata["validation_targets"]),
                executable_validation_targets=tuple(metadata["validation_targets"]),
                null_controls=tuple(metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="promoted_source_accounting_fixture",
                domain_review_status="source_bounded_no_empirical_validation",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 "
        + "Data Fusion and Manifold Alignment: Constructing the Unified State Space"
        + " Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": sorted(set(SOURCE_LEDGER_IDS) - set(consumed)),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "category_counts": dict(sorted(category_counts.items())),
        "block_type_counts": dict(sorted(block_counts.items())),
        "math_ids": sorted(
            {math_id for record in anchors for math_id in record.get("math_ids", [])}
        ),
        "image_ids": sorted(
            {image_id for record in anchors for image_id in record.get("image_ids", [])}
        ),
        "table_ids": sorted(
            {str(record["table_id"]) for record in anchors if record.get("table_id") is not None}
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R04216",
    }
    return DataFusionAndManifoldAlignmentConstructingTheUnifiedStateSpaceSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> DataFusionAndManifoldAlignmentConstructingTheUnifiedStateSpaceSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_data_fusion_and_manifold_alignment_constructing_the_unified_state_space_specs(
        load_jsonl(ledger_path)
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(
    bundle: DataFusionAndManifoldAlignmentConstructingTheUnifiedStateSpaceSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Data Fusion and Manifold Alignment: Constructing the Unified State Space"
        + " Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Next source boundary: {bundle.summary['next_source_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### `{spec.key}`",
                "",
                spec.canonical_statement,
                "",
                f"- Context: `{spec.context_id}`",
                f"- Protocol: `{spec.validation_protocol}`",
                f"- Source equations: {', '.join(spec.source_equation_ids)}",
                f"- Null controls: {', '.join(spec.null_controls)}",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(
    bundle: DataFusionAndManifoldAlignmentConstructingTheUnifiedStateSpaceSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_data_fusion_and_manifold_alignment_constructing_the_unified_state_space_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_data_fusion_and_manifold_alignment_constructing_the_unified_state_space_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 data-fusion manifold-alignment specs from the ledger."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()
    outputs = write_outputs(
        build_from_ledger(args.ledger), output_dir=args.output_dir, date_tag=args.date_tag
    )
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
