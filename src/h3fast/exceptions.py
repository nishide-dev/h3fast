"""Project-specific exceptions."""


class H3FastError(Exception):
    """Base exception for user-facing H3Fast failures."""


class ValidationError(H3FastError):
    """Raised when an artifact or protocol fails validation."""
