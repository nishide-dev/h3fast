"""Safe parsing and verification for SHA-256 inventories."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath

from h3fast.exceptions import ValidationError

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 without loading the full file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_relative_path(value: str) -> PurePosixPath:
    """Validate an artifact-relative POSIX path."""
    if "\\" in value:
        msg = f"checksum path must use POSIX separators: {value!r}"
        raise ValidationError(msg)
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        msg = f"unsafe artifact-relative path: {value!r}"
        raise ValidationError(msg)
    return path


def parse_checksum_file(path: Path) -> dict[PurePosixPath, str]:
    """Parse a GNU-style SHA-256 file while rejecting ambiguity."""
    if not path.is_file():
        msg = f"checksum inventory is missing: {path}"
        raise ValidationError(msg)

    entries: dict[PurePosixPath, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        digest, separator, relative_name = line.partition("  ")
        if not separator or not SHA256_PATTERN.fullmatch(digest):
            msg = f"invalid checksum line {line_number} in {path}"
            raise ValidationError(msg)
        relative = validate_relative_path(relative_name)
        if relative in entries:
            msg = f"duplicate checksum path: {relative}"
            raise ValidationError(msg)
        entries[relative] = digest

    if not entries:
        msg = f"checksum inventory has no entries: {path}"
        raise ValidationError(msg)
    return entries


def verify_checksums(root: Path, inventory: Path) -> tuple[str, ...]:
    """Verify every listed file and reject symlink escapes."""
    root_resolved = root.resolve(strict=True)
    entries = parse_checksum_file(inventory)
    verified: list[str] = []

    for relative, expected in entries.items():
        candidate = root.joinpath(*relative.parts)
        if candidate.is_symlink():
            msg = f"symlinks are not allowed in verified artifacts: {relative}"
            raise ValidationError(msg)
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            msg = f"checksummed file is missing: {relative}"
            raise ValidationError(msg) from error
        if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
            msg = f"checksummed path is outside the artifact or not a file: {relative}"
            raise ValidationError(msg)
        actual = sha256_file(resolved)
        if actual != expected:
            msg = f"checksum mismatch for {relative}: expected {expected}, got {actual}"
            raise ValidationError(msg)
        verified.append(relative.as_posix())

    return tuple(verified)
