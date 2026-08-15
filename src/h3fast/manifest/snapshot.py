"""Local MiniMax H3 snapshot inspection without downloads or imports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from h3fast.exceptions import ValidationError
from h3fast.manifest.checksums import sha256_file

if TYPE_CHECKING:
    from pathlib import Path

BaseVariant = Literal["fl2va", "ref2va"]
SUPPORTED_BASE_MODEL = "MiniMaxAI/MiniMax-H3"
IMMUTABLE_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
REQUIRED_COMPONENTS = (
    "processor",
    "tokenizer",
    "text_encoder",
    "transformer",
    "visual_vae",
    "audio_vae",
)


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    """One local snapshot file."""

    path: str
    size: int
    sha256: str | None

    def to_dict(self) -> dict[str, str | int | None]:
        """Return JSON-serializable file data."""
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class SnapshotReport:
    """Validated local H3 snapshot structure and provenance input."""

    base_model: str
    base_revision: str
    variant: BaseVariant
    files: tuple[SnapshotFile, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable report data."""
        return {
            "valid": True,
            "base_model": self.base_model,
            "base_revision": self.base_revision,
            "variant": self.variant,
            "files": [file.to_dict() for file in self.files],
        }


def _validate_snapshot_structure(root: Path, variant: BaseVariant) -> None:
    if not root.is_dir():
        msg = f"snapshot directory does not exist: {root}"
        raise ValidationError(msg)
    if not (root / "model_index.json").is_file():
        msg = "snapshot root is missing model_index.json"
        raise ValidationError(msg)

    variant_root = root / variant.upper()
    if not (variant_root / "model_index.json").is_file():
        msg = f"snapshot is missing {variant.upper()}/model_index.json"
        raise ValidationError(msg)
    missing = [
        component
        for component in REQUIRED_COMPONENTS
        if not (variant_root / component).is_dir()
    ]
    if missing:
        msg = f"snapshot {variant.upper()} is missing components: {', '.join(missing)}"
        raise ValidationError(msg)


def inspect_snapshot(
    root: Path,
    *,
    variant: BaseVariant,
    base_revision: str,
    base_model: str = SUPPORTED_BASE_MODEL,
    include_hashes: bool = False,
) -> SnapshotReport:
    """Inspect an explicitly supplied local snapshot without network access."""
    if base_model != SUPPORTED_BASE_MODEL:
        msg = f"unsupported base model: {base_model!r}"
        raise ValidationError(msg)
    if not IMMUTABLE_REVISION_PATTERN.fullmatch(base_revision):
        msg = "base revision must be a lowercase 40-character commit SHA"
        raise ValidationError(msg)
    _validate_snapshot_structure(root, variant)

    files: list[SnapshotFile] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            msg = f"snapshot symlinks are not accepted: {path.relative_to(root)}"
            raise ValidationError(msg)
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            SnapshotFile(
                path=relative,
                size=path.stat().st_size,
                sha256=sha256_file(path) if include_hashes else None,
            )
        )

    return SnapshotReport(
        base_model=base_model,
        base_revision=base_revision,
        variant=variant,
        files=tuple(files),
    )
