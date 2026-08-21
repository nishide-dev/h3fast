"""Tests for the named generation profile registry."""

import json
from pathlib import Path

import pytest

from h3fast.benchmarks import (
    DEFAULT_GENERATION_PROFILE,
    GENERATION_PROFILES,
    resolve_generation_profile,
)
from h3fast.exceptions import ValidationError


def test_default_profile_is_the_adopted_turbo12_configuration() -> None:
    """ADR 0014 adopted turbo12 as the default after a tied pairwise."""
    assert DEFAULT_GENERATION_PROFILE == "balanced"

    profile = resolve_generation_profile(DEFAULT_GENERATION_PROFILE)

    assert profile.name == "balanced"
    assert profile.protocol_path == Path("benchmarks/protocol-turbo12-fp8.yaml")
    assert profile.protocol_path.is_file()


def test_every_profile_points_at_a_committed_protocol() -> None:
    for name in GENERATION_PROFILES:
        profile = resolve_generation_profile(name)

        assert profile.name == name
        assert profile.protocol_path.is_file()
        protocol = json.loads(profile.protocol_path.read_text(encoding="utf-8"))
        assert protocol["protocol_id"] == profile.protocol_id


def test_profiles_cover_the_measured_speed_quality_ladder() -> None:
    """Each profile records why it exists, newest evidence included."""
    assert set(GENERATION_PROFILES) == {
        "quality",
        "balanced",
        "bf16-balanced",
        "speed",
    }

    quality = resolve_generation_profile("quality")
    balanced = resolve_generation_profile("balanced")
    bf16 = resolve_generation_profile("bf16-balanced")
    speed = resolve_generation_profile("speed")

    assert quality.protocol_id == "h3fast-phase1b-sage-attn-v1"
    assert balanced.protocol_id == "h3fast-phase1b-turbo12-fp8-v1"
    assert bf16.protocol_id == "h3fast-phase1b-turbo-lora-12-v1"
    assert speed.protocol_id == "h3fast-phase1b-turbo-lora-v1"
    # The quality profile is the pairwise reference, so faster profiles are
    # only admissible while their measured verdict stays non-negative; the
    # speed profile is the one exception recorded as degraded.
    assert balanced.pairwise_score >= 0.0
    assert bf16.pairwise_score >= 0.0
    assert speed.pairwise_score < 0.0
    assert (
        speed.speedup_versus_quality
        > balanced.speedup_versus_quality
        > bf16.speedup_versus_quality
        > quality.speedup_versus_quality
        == 1.0
    )
    for profile in (quality, balanced, bf16, speed):
        assert profile.evidence.startswith("docs/experiments/")


def test_unknown_profile_fails_closed() -> None:
    with pytest.raises(ValidationError, match="unsupported generation profile"):
        resolve_generation_profile("fastest")

    with pytest.raises(ValidationError, match="unsupported generation profile"):
        resolve_generation_profile("")


def test_profile_report_excludes_local_paths() -> None:
    payload = resolve_generation_profile("balanced").to_dict()

    assert payload["name"] == "balanced"
    assert payload["protocol_id"] == "h3fast-phase1b-turbo12-fp8-v1"
    assert "/grouper/" not in json.dumps(payload)


def test_cli_lists_profiles_and_resolves_the_default(capsys) -> None:
    """`h3fast benchmark profiles` exposes the ladder to operators."""
    from h3fast.cli import main

    assert main(["benchmark", "profiles"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["default"] == "balanced"
    names = [entry["name"] for entry in payload["profiles"]]
    assert names == ["quality", "bf16-balanced", "balanced", "speed"]
    balanced = next(e for e in payload["profiles"] if e["name"] == "balanced")
    assert balanced["protocol"] == "protocol-turbo12-fp8.yaml"
    assert balanced["pairwise_score"] == 0.20


def test_quantized_profiles_pin_dynamic_lora_merge() -> None:
    """Static merge cannot handle FP8 runtime layout, so dynamic is required."""
    for name in GENERATION_PROFILES:
        protocol = json.loads(
            resolve_generation_profile(name).protocol_path.read_text(encoding="utf-8")
        )
        runtime = protocol["runtime"]
        if runtime.get("quantization") is None or "lora" not in runtime:
            continue
        assert runtime["lora"]["merge_mode"] == "dynamic"
