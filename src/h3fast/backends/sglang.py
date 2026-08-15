"""SGLang compatibility metadata without importing or initializing SGLang."""

from dataclasses import dataclass
from importlib import metadata

REFERENCE_SGLANG_VERSION = "0.5.15.post1"


@dataclass(frozen=True, slots=True)
class SGLangStatus:
    """Installed SGLang version and reference compatibility."""

    installed_version: str | None
    reference_version: str
    compatible: bool

    def to_dict(self) -> dict[str, str | bool | None]:
        """Return JSON-serializable status data."""
        return {
            "installed_version": self.installed_version,
            "reference_version": self.reference_version,
            "compatible": self.compatible,
        }


def inspect_sglang() -> SGLangStatus:
    """Inspect distribution metadata without importing the optional package."""
    try:
        installed = metadata.version("sglang")
    except metadata.PackageNotFoundError:
        installed = None
    return SGLangStatus(
        installed_version=installed,
        reference_version=REFERENCE_SGLANG_VERSION,
        compatible=installed == REFERENCE_SGLANG_VERSION,
    )
