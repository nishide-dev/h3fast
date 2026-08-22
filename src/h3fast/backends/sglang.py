"""SGLang compatibility metadata without importing or initializing SGLang."""

from dataclasses import dataclass
from importlib import metadata

# Adopted 2026-08-23 (experiment 0021). Produces a bit-identical artifact to
# the previous pin while cutting peak VRAM by 8%, so protocols measured on
# 6eb941a3 keep their own revision rather than being rewritten to this one.
REFERENCE_SGLANG_COMMIT = "7d22b7a8750f53a04e41a5a5671f9a56ab6cd001"
PREVIOUS_SGLANG_COMMIT = "6eb941a34cb100b708a42ed1d26d2bdefafbd01e"
REFERENCE_SGLANG_VERSION = f"git:{REFERENCE_SGLANG_COMMIT}"


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
