"""Fail-closed verification that a requested attention backend was used.

SGLang reports a component-scoped override when the component loader
accepts it, but MiniMax H3 resolves its attention backend lazily on the
first forward, after that override context has closed. A run can therefore
log the requested backend and still execute a different one.

This module reads the guarded server log and treats the *last* resolved
backend as the one the DiT actually used, so a benchmark that silently fell
back is rejected instead of being compared against a baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from h3fast.exceptions import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

_BACKEND_PATTERN = re.compile(r"Using ([A-Za-z0-9_]+) attention backend")
_COMPONENT_PATTERN = re.compile(
    r"Using ([A-Za-z0-9_]+) backend for component: ([A-Za-z0-9_]+)"
)


@dataclass(frozen=True, slots=True)
class BackendVerificationReport:
    """Evidence that the requested attention backend was resolved."""

    requested: str
    resolved: str
    observed: tuple[str, ...]
    components: dict[str, str]
    verified: bool

    def to_dict(self) -> dict[str, object]:
        """Return backend evidence without local paths."""
        return {
            "schema_version": "1.0",
            "requested": self.requested,
            "resolved": self.resolved,
            "observed": list(self.observed),
            "components": dict(self.components),
            "verified": self.verified,
        }


def verify_attention_backend(
    server_log_path: Path, *, requested: str
) -> BackendVerificationReport:
    """Verify the resolved attention backend, failing closed on mismatch."""
    try:
        text = server_log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        message = f"guarded server log could not be read: {error}"
        raise ValidationError(message) from error

    observed = tuple(_BACKEND_PATTERN.findall(text))
    components = {
        component: backend for backend, component in _COMPONENT_PATTERN.findall(text)
    }
    if not observed:
        message = (
            "guarded server log contains no attention backend resolution; "
            "cannot verify which backend executed"
        )
        raise ValidationError(message)

    # H3 resolves lazily, so the final resolution is the one the DiT used.
    resolved = observed[-1]
    if requested not in ("auto", resolved):
        message = (
            f"requested attention backend {requested!r} but the server "
            f"resolved to {resolved!r}; the benchmark is not comparable"
        )
        raise ValidationError(message)

    return BackendVerificationReport(
        requested=requested,
        resolved=resolved,
        observed=observed,
        components=components,
        verified=True,
    )
