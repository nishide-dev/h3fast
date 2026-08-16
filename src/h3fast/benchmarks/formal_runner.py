"""Formal quality-set generation runner over the private case registry.

Binds the private reviewed registry to the committed formal quality set
(fail-closed on prompt digest or metadata drift), builds per-case
payloads from the pinned protocol's fixed generation parameters, runs
each case against a guarded local server, and records a redacted
run manifest with per-artifact digests. Prompts never reach stdout or
the manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from h3fast.benchmarks.human_pairwise_runner import _require_private_file
from h3fast.benchmarks.quality_sets import check_formal_quality_set
from h3fast.exceptions import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

    from h3fast.benchmarks.client import BenchmarkResult

_ASPECT_RATIO_MAP = {"landscape": "16:9", "portrait": "9:16", "square": "1:1"}
_SPLITS = frozenset({"smoke", "regression"})
_REPETITION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_TEMPLATE_FIELDS = ("short_edge", "sigma_points", "flow_shift", "audio_flow_shift")
_MATCHED_FIELDS = ("seed", "task", "duration_seconds", "aspect_ratio")


@dataclass(frozen=True, slots=True)
class FormalRunReport:
    """Aggregate outcome of one formal generation repetition."""

    protocol_id: str
    repetition_id: str
    case_count: int
    generated_count: int
    skipped_count: int
    manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return run metadata without prompts or local paths."""
        return {
            "schema_version": "1.0",
            "protocol_id": self.protocol_id,
            "repetition_id": self.repetition_id,
            "case_count": self.case_count,
            "generated_count": self.generated_count,
            "skipped_count": self.skipped_count,
            "manifest_sha256": self.manifest_sha256,
        }


def _run_single_case(
    protocol_path: Path,
    case: dict[str, object],
    *,
    endpoint: str,
    output_dir: Path,
    poll_interval: float,
    timeout: float,
) -> BenchmarkResult:
    from h3fast.benchmarks.client import run_supplied_case

    return run_supplied_case(
        protocol_path,
        case,
        endpoint=endpoint,
        output_dir=output_dir,
        poll_interval=poll_interval,
        timeout=timeout,
    )


def _load_registry_cases(registry_path: Path) -> dict[str, dict[str, object]]:
    _require_private_file(registry_path, "private quality registry")
    try:
        value = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        message = f"private quality registry could not be read: {error}"
        raise ValidationError(message) from error
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        message = "private quality registry must contain a cases array"
        raise ValidationError(message)
    cases: dict[str, dict[str, object]] = {}
    for index, case in enumerate(value["cases"]):
        if not isinstance(case, dict):
            message = f"private registry case {index} must be an object"
            raise ValidationError(message)
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            message = f"private registry case {index} must define an id"
            raise ValidationError(message)
        if case_id in cases:
            message = f"duplicate private registry case: {case_id}"
            raise ValidationError(message)
        cases[case_id] = case
    return cases


def _bind_registry_to_formal_set(
    formal_cases: list[dict[str, object]],
    registry_cases: dict[str, dict[str, object]],
) -> None:
    formal_ids = [str(case["id"]) for case in formal_cases]
    if list(registry_cases) != formal_ids:
        message = "private registry must cover every formal case in fixed order"
        raise ValidationError(message)
    for formal_case in formal_cases:
        case_id = str(formal_case["id"])
        registry_case = registry_cases[case_id]
        prompt = registry_case.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            message = f"private registry case {case_id} must define a prompt"
            raise ValidationError(message)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if digest != formal_case["prompt_sha256"]:
            message = f"registry prompt digest mismatch for {case_id}"
            raise ValidationError(message)
        for field in _MATCHED_FIELDS:
            if registry_case.get(field) != formal_case[field]:
                message = (
                    f"registry {field} does not match the formal case for {case_id}"
                )
                raise ValidationError(message)


def _protocol_template(protocol_path: Path) -> tuple[str, dict[str, object]]:
    from h3fast.benchmarks.client import _protocol_identity

    protocol_id, _sglang_commit = _protocol_identity(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cases = protocol.get("cases")
    if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], dict):
        message = "benchmark protocol must define exactly one template case"
        raise ValidationError(message)
    template = cases[0]
    fixed: dict[str, object] = {"conditions": template.get("conditions", [])}
    for field in _TEMPLATE_FIELDS:
        if field not in template:
            message = f"benchmark protocol template case is missing {field}"
            raise ValidationError(message)
        fixed[field] = template[field]
    return protocol_id, fixed


def _record_artifact(record: dict[str, object]) -> dict[str, object] | None:
    artifact = record.get("artifact")
    if not isinstance(artifact, dict):
        return None
    if not isinstance(artifact.get("path"), str) or not isinstance(
        artifact.get("sha256"), str
    ):
        return None
    return artifact


def _write_atomic_json(path: Path, value: dict[str, object]) -> None:
    data = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)


def _reusable_result(
    result_path: Path, formal_case: dict[str, object], protocol_id: str
) -> dict[str, object] | None:
    from pathlib import Path as _Path

    if not result_path.is_file():
        return None
    try:
        value = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    # A reused result must be bound to the same prompt digest and pinned
    # protocol as the run being resumed, not merely self-consistent.
    if value.get("prompt_sha256") != formal_case["prompt_sha256"]:
        return None
    if value.get("protocol_id") != protocol_id:
        return None
    artifact = _record_artifact(value)
    if artifact is None:
        return None
    target = _Path(str(artifact["path"]))
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    if digest.hexdigest() != artifact["sha256"]:
        return None
    return value


def _manifest_entry(record: dict[str, object]) -> dict[str, object]:
    from pathlib import Path as _Path

    artifact = _record_artifact(record)
    if artifact is None:
        message = "benchmark result record is missing artifact metadata"
        raise ValidationError(message)
    return {
        "case_id": record["case_id"],
        "artifact_name": _Path(str(artifact["path"])).name,
        "artifact_sha256": artifact["sha256"],
        "artifact_size": artifact["size"],
        "elapsed_seconds": record["elapsed_seconds"],
        "job_id": record["job_id"],
    }


def run_formal_cases(
    protocol_path: Path,
    registry_path: Path,
    formal_set_path: Path,
    *,
    endpoint: str,
    output_dir: Path,
    repetition_id: str,
    split: str | None = None,
    task: str = "t2va",
    poll_interval: float = 1.0,
    timeout: float = 7200.0,
) -> FormalRunReport:
    """Generate formal-case media for one repetition with digest binding."""
    if not _REPETITION_ID_PATTERN.fullmatch(repetition_id):
        message = "repetition_id must be a simple alphanumeric identifier"
        raise ValidationError(message)
    if split is not None and split not in _SPLITS:
        message = f"unknown formal split: {split}"
        raise ValidationError(message)
    if task != "t2va":
        # The pinned payload contract cannot express reference-conditioned
        # generation yet; silently degrading fl2va/ref2va to t2va is the
        # one thing this runner must never do.
        message = f"unsupported formal task family for generation: {task}"
        raise ValidationError(message)
    check_formal_quality_set(formal_set_path)
    formal_raw = formal_set_path.read_bytes()
    formal_set = json.loads(formal_raw)
    formal_cases = list(formal_set["cases"])
    registry_cases = _load_registry_cases(registry_path)
    _bind_registry_to_formal_set(formal_cases, registry_cases)
    protocol_id, fixed = _protocol_template(protocol_path)

    selected = [
        case
        for case in formal_cases
        if (split is None or case["split"] == split) and case["task"] == task
    ]
    if not selected:
        message = "no formal cases match the selected split and task"
        raise ValidationError(message)
    for case in selected:
        if case["reference_asset_sha256s"]:
            message = (
                "reference-conditioned formal cases are not supported yet: "
                f"{case['id']}"
            )
            raise ValidationError(message)
    repetition_dir = output_dir / repetition_id
    repetition_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    skipped_count = 0
    for formal_case in selected:
        case_id = str(formal_case["id"])
        registry_case = registry_cases[case_id]
        result_path = repetition_dir / f"{case_id}.result.json"
        reused = _reusable_result(result_path, formal_case, protocol_id)
        if reused is not None:
            skipped_count += 1
            entries.append(_manifest_entry(reused))
            continue
        payload_case: dict[str, object] = {
            "id": case_id,
            "prompt": registry_case["prompt"],
            "seed": formal_case["seed"],
            "aspect_ratio": _ASPECT_RATIO_MAP[str(formal_case["aspect_ratio"])],
            "duration_seconds": formal_case["duration_seconds"],
            **fixed,
        }
        result = _run_single_case(
            protocol_path,
            payload_case,
            endpoint=endpoint,
            output_dir=repetition_dir,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        if result.prompt_sha256 != formal_case["prompt_sha256"]:
            message = f"executed prompt digest mismatch for {case_id}"
            raise ValidationError(message)
        record = result.to_dict()
        _write_atomic_json(result_path, record)
        entries.append(_manifest_entry(record))

    manifest = {
        "schema_version": "1.0",
        "protocol_id": protocol_id,
        "repetition_id": repetition_id,
        "split": split,
        "task": task,
        "formal_set_sha256": hashlib.sha256(formal_raw).hexdigest(),
        "cases": entries,
    }
    manifest_data = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (repetition_dir / "run-manifest.json").write_text(manifest_data, encoding="utf-8")

    return FormalRunReport(
        protocol_id=protocol_id,
        repetition_id=repetition_id,
        case_count=len(selected),
        generated_count=len(entries) - skipped_count,
        skipped_count=skipped_count,
        manifest_sha256=hashlib.sha256(manifest_data.encode("utf-8")).hexdigest(),
    )
