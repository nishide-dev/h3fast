"""Named generation profiles for the measured speed/quality ladder.

Each profile is a pinned protocol that was adopted from a Tier 2 evaluation
(ADR 0014). Selecting a profile never changes the task family, model
revision, or precision; it selects the sampling and attention configuration
whose measured speed and blind-pairwise verdict are recorded below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from h3fast.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class GenerationProfile:
    """A pinned protocol plus the evidence that justified adopting it."""

    name: str
    protocol_id: str
    protocol_path: Path
    speedup_versus_quality: float
    pairwise_score: float
    evidence: str
    summary: str

    def to_dict(self) -> dict[str, object]:
        """Return profile metadata without local paths."""
        return {
            "schema_version": "1.0",
            "name": self.name,
            "protocol_id": self.protocol_id,
            "protocol": self.protocol_path.name,
            "speedup_versus_quality": self.speedup_versus_quality,
            "pairwise_score": self.pairwise_score,
            "evidence": self.evidence,
            "summary": self.summary,
        }


_SAGE_EXPERIMENT = "docs/experiments/0011-sage-attention-tier2-adoption.md"
_TURBO_EXPERIMENT = "docs/experiments/0012-turbo-lora-tier2-evaluation.md"
_TURBO12_EXPERIMENT = "docs/experiments/0013-turbo-lora-step-sweep.md"

GENERATION_PROFILES: dict[str, GenerationProfile] = {
    "quality": GenerationProfile(
        name="quality",
        protocol_id="h3fast-phase1b-sage-attn-v1",
        protocol_path=Path("benchmarks/protocol-sage.yaml"),
        speedup_versus_quality=1.0,
        pairwise_score=0.20,
        evidence=_SAGE_EXPERIMENT,
        summary=(
            "50-step sampling with Sage attention. The reference for quality "
            "comparisons: no degradation against the FlashAttention baseline."
        ),
    ),
    "balanced": GenerationProfile(
        name="balanced",
        protocol_id="h3fast-phase1b-turbo-lora-12-v1",
        protocol_path=Path("benchmarks/protocol-turbo12.yaml"),
        speedup_versus_quality=3.86,
        pairwise_score=0.0,
        evidence=_TURBO12_EXPERIMENT,
        summary=(
            "Turbo LoRA at 12 sigma points (11 effective steps). Blind "
            "pairwise against quality tied at 6/6/8, so no degradation was "
            "detected at 3.86x the throughput."
        ),
    ),
    "speed": GenerationProfile(
        name="speed",
        protocol_id="h3fast-phase1b-turbo-lora-v1",
        protocol_path=Path("benchmarks/protocol-turbo.yaml"),
        speedup_versus_quality=4.94,
        pairwise_score=-0.25,
        evidence=_TURBO_EXPERIMENT,
        summary=(
            "Turbo LoRA at 9 sigma points (8 effective steps). Fastest, but "
            "blind pairwise preferred quality 10/5/5; use where iteration "
            "speed outweighs motion fidelity."
        ),
    ),
}

DEFAULT_GENERATION_PROFILE = "balanced"


def resolve_generation_profile(name: str) -> GenerationProfile:
    """Return the named profile, failing closed on an unknown name."""
    profile = GENERATION_PROFILES.get(name)
    if profile is None:
        supported = ", ".join(sorted(GENERATION_PROFILES))
        message = f"unsupported generation profile {name!r}; expected {supported}"
        raise ValidationError(message)
    return profile
