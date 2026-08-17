"""Tests for fail-closed attention backend verification."""

from pathlib import Path

import pytest

from h3fast.benchmarks import verify_attention_backend
from h3fast.exceptions import ValidationError

_LOADED = "Loading required modules: 100%"


def _log(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "server.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_auto_backend_accepts_any_resolved_backend(tmp_path: Path) -> None:
    log = _log(
        tmp_path,
        [
            "[08-18 05:33:33] Using fa attention backend",
            "[08-18 05:34:11] Attention backends for text_encoder: fa",
        ],
    )

    report = verify_attention_backend(log, requested="auto")

    assert report.requested == "auto"
    assert report.resolved == "fa"
    assert report.verified is True
    payload = report.to_dict()
    assert str(tmp_path) not in str(payload)


def test_global_sage_run_is_verified(tmp_path: Path) -> None:
    log = _log(
        tmp_path,
        [
            "[08-18 08:14:18] Using sage_attn attention backend",
            "[08-18 08:14:44] Using torch_sdpa backend for component: text_encoder",
            "[08-18 08:15:02] Using sage_attn attention backend",
        ],
    )

    report = verify_attention_backend(log, requested="sage_attn")

    assert report.resolved == "sage_attn"
    assert report.verified is True


def test_lost_component_override_fails_closed(tmp_path: Path) -> None:
    """The measured component-only failure mode must be rejected."""
    log = _log(
        tmp_path,
        [
            "[08-18 07:36:48] Using sage_attn attention backend",
            "[08-18 07:37:26] Using sage_attn backend for component: transformer",
            "[08-18 07:37:33] Using fa attention backend",
        ],
    )

    with pytest.raises(ValidationError, match="resolved to 'fa'"):
        verify_attention_backend(log, requested="sage_attn")


def test_missing_backend_evidence_fails_closed(tmp_path: Path) -> None:
    log = _log(tmp_path, ["[08-18 05:33:33] server started", _LOADED])

    with pytest.raises(ValidationError, match="no attention backend"):
        verify_attention_backend(log, requested="sage_attn")

    with pytest.raises(ValidationError, match="no attention backend"):
        verify_attention_backend(log, requested="auto")


def test_unreadable_log_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="could not be read"):
        verify_attention_backend(tmp_path / "missing.log", requested="sage_attn")


def test_report_records_all_observed_backends(tmp_path: Path) -> None:
    log = _log(
        tmp_path,
        [
            "[08-18 08:14:18] Using sage_attn attention backend",
            "[08-18 08:14:44] Using torch_sdpa backend for component: text_encoder",
            "[08-18 08:15:02] Using sage_attn attention backend",
        ],
    )

    report = verify_attention_backend(log, requested="sage_attn")

    assert report.observed == ("sage_attn", "sage_attn")
    assert report.components == {"text_encoder": "torch_sdpa"}
    assert report.to_dict() == {
        "schema_version": "1.0",
        "requested": "sage_attn",
        "resolved": "sage_attn",
        "observed": ["sage_attn", "sage_attn"],
        "components": {"text_encoder": "torch_sdpa"},
        "verified": True,
    }
