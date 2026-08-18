"""Tests for formal quality metric-plan readiness."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from h3fast.benchmarks import check_quality_metric_plan
from h3fast.exceptions import ValidationError

PLAN_PATH = Path("benchmarks/quality/formal-quality-metric-plan.json")


def _plan() -> dict[str, object]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _write_plan(tmp_path: Path, plan: object) -> Path:
    path = tmp_path / "quality-metric-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _metrics(plan: dict[str, object]) -> list[dict[str, object]]:
    metrics = plan["metrics"]
    assert isinstance(metrics, list)
    assert all(isinstance(metric, dict) for metric in metrics)
    return metrics  # type: ignore[return-value]


def _reset_unassigned(metric: dict[str, object]) -> dict[str, object]:
    """Return the metric reset to the unassigned baseline disposition."""
    metric.update(
        {
            "state": "unassigned",
            "owner": None,
            "implementation": None,
            "budget": None,
            "evidence": [],
        }
    )
    return metric


def _approve(plan: dict[str, object]) -> dict[str, object]:
    approved = copy.deepcopy(plan)
    approved["status"] = "approved"
    for metric in _metrics(approved):
        family = str(metric["family"])
        metric.update(
            {
                "state": "approved",
                "owner": "quality-owner",
                "implementation": {
                    "name": f"h3fast-{family}",
                    "version": "1.0.0",
                    "revision": "a" * 40,
                    "entrypoint": f"h3fast.metrics.{family}",
                    "dependencies": ["python==3.12.7"],
                    "inputs": [
                        "prompt",
                        "reference-media",
                        "baseline-video",
                        "baseline-audio",
                        "candidate-video",
                        "candidate-audio",
                        "human-ballot",
                    ],
                    "score_direction": "higher-is-better",
                },
                "budget": {
                    "method": "baseline-self-envelope-v1",
                    "unit": "normalized-score",
                    "absolute_tolerance": 0,
                    "relative_tolerance": 0,
                    "minimum_case_coverage": 1.0,
                    "aggregation": "per-case-all-runs",
                    "failure_policy": "any-family-fails",
                    "notes": "Candidate observations must remain inside the baseline envelope.",
                },
                "evidence": [f"https://example.test/metrics/{family}/v1"],
            }
        )
    return approved


def test_committed_metric_plan_is_valid_draft() -> None:
    report = check_quality_metric_plan(PLAN_PATH)

    assert report.status == "draft"
    assert report.ready is False
    assert report.approved_metrics == 0
    # Three families are planned per ADR 0013; none carries an approved
    # budget, so the plan stays a draft and every family remains a blocker.
    assert report.planned_metrics == 3
    assert report.unassigned_metrics == 3
    assert len(report.blockers) == 6


def test_approved_metric_plan_is_ready(tmp_path: Path) -> None:
    approved = _approve(_plan())
    schema = json.loads(
        Path("schemas/quality-metric-plan.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    ).validate(approved)
    report = check_quality_metric_plan(_write_plan(tmp_path, approved))

    assert report.status == "approved"
    assert report.ready is True
    assert report.approved_metrics == 6
    assert report.blockers == ()


def test_metric_plan_rejects_invalid_json_and_root(tmp_path: Path) -> None:
    path = tmp_path / "quality-metric-plan.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValidationError, match="not valid JSON"):
        check_quality_metric_plan(path)

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValidationError, match="root must be an object"):
        check_quality_metric_plan(path)


def test_metric_plan_rejects_unknown_and_missing_fields(tmp_path: Path) -> None:
    plan = _plan()
    plan["unexpected"] = True
    with pytest.raises(ValidationError, match="unknown fields: unexpected"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))

    plan = _plan()
    plan.pop("updated_at")
    with pytest.raises(ValidationError, match="missing required fields: updated_at"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "2.0", "unsupported quality metric-plan"),
        ("plan_id", "other", "unsupported quality metric plan_id"),
        ("status", "pending", "status has unsupported value"),
        ("updated_at", "tomorrow", "must be an ISO 8601 date"),
        ("source_issue", "http://example.test/16", "must be an HTTPS URL"),
        ("quality_profile", "fast", "supports only the exact profile"),
    ],
)
def test_metric_plan_rejects_invalid_top_level_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    plan = _plan()
    plan[field] = value

    with pytest.raises(ValidationError, match=message):
        check_quality_metric_plan(_write_plan(tmp_path, plan))


def test_metric_plan_rejects_changed_evaluation_contract(tmp_path: Path) -> None:
    plan = _plan()
    evaluation = plan["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["baseline_repetitions"] = 1
    with pytest.raises(ValidationError, match="baseline_repetitions must equal 3"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))

    plan = _plan()
    evaluation = plan["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["statistics"] = ["p50", "p5", "p95", "worst-case"]
    with pytest.raises(ValidationError, match="must use fixed order"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))


def _exemption() -> dict[str, object]:
    return {
        "policy": "bit-exact-digest-match-v1",
        "verified_repetitions": 2,
        "owner": "nishide-dev",
        "verified_at": "2026-08-17",
        "evidence": [
            "docs/experiments/0008-formal-generation-determinism.md",
        ],
    }


def test_committed_plan_has_no_determinism_exemption() -> None:
    plan = _plan()
    evaluation = plan["evaluation"]
    assert isinstance(evaluation, dict)
    assert "deterministic_generation_exemption" not in evaluation


def test_metric_plan_accepts_verified_determinism_exemption(tmp_path: Path) -> None:
    plan = _plan()
    evaluation = plan["evaluation"]
    assert isinstance(evaluation, dict)
    evaluation["deterministic_generation_exemption"] = _exemption()

    report = check_quality_metric_plan(_write_plan(tmp_path, plan))

    assert report.plan_id == "h3fast-phase0-formal-quality-metrics-v1"


def test_metric_plan_rejects_incomplete_determinism_exemption(tmp_path: Path) -> None:
    for field, value, message in (
        ("policy", "trust-me-v1", "policy must equal"),
        ("verified_repetitions", 1, "verified_repetitions must be at least 2"),
        ("owner", "", "owner"),
        ("verified_at", "17-08-2026", "verified_at"),
        ("evidence", [], "evidence"),
    ):
        plan = _plan()
        evaluation = plan["evaluation"]
        assert isinstance(evaluation, dict)
        exemption = _exemption()
        exemption[field] = value
        evaluation["deterministic_generation_exemption"] = exemption
        with pytest.raises(ValidationError, match=message):
            check_quality_metric_plan(_write_plan(tmp_path, plan))

    plan = _plan()
    evaluation = plan["evaluation"]
    assert isinstance(evaluation, dict)
    exemption = _exemption()
    del exemption["owner"]
    evaluation["deterministic_generation_exemption"] = exemption
    with pytest.raises(ValidationError, match="missing required fields"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))


def test_metric_plan_rejects_missing_or_duplicate_families(tmp_path: Path) -> None:
    plan = _plan()
    _metrics(plan).pop()
    with pytest.raises(ValidationError, match="missing: human-pairwise"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))

    plan = _plan()
    metrics = _metrics(plan)
    _reset_unassigned(metrics[0])
    _reset_unassigned(metrics[1])
    metrics[1]["family"] = metrics[0]["family"]
    with pytest.raises(ValidationError, match="duplicate quality metric family"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))


def test_metric_plan_rejects_invalid_state_disposition(tmp_path: Path) -> None:
    plan = _plan()
    metric = _reset_unassigned(_metrics(plan)[0])
    metric["owner"] = "quality-owner"
    with pytest.raises(ValidationError, match=r"unassigned metric .* disposition"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))

    plan = _plan()
    metric = _reset_unassigned(_metrics(plan)[0])
    metric["state"] = "planned"
    with pytest.raises(ValidationError, match=r"planned metric .* requires an owner"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))


def test_metric_plan_rejects_moving_or_unpinned_implementation(
    tmp_path: Path,
) -> None:
    plan = _approve(_plan())
    implementation = _metrics(plan)[0]["implementation"]
    assert isinstance(implementation, dict)
    implementation["version"] = "latest"
    with pytest.raises(ValidationError, match="exact non-moving identifier"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))

    plan = _approve(_plan())
    implementation = _metrics(plan)[0]["implementation"]
    assert isinstance(implementation, dict)
    implementation["inputs"] = ["candidate-audio"]
    with pytest.raises(ValidationError, match="missing required values"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))

    plan = _approve(_plan())
    implementation = _metrics(plan)[0]["implementation"]
    assert isinstance(implementation, dict)
    implementation["dependencies"] = ["python>=3.12"]
    with pytest.raises(ValidationError, match="exact name==version"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))

    plan = _approve(_plan())
    implementation = _metrics(plan)[0]["implementation"]
    assert isinstance(implementation, dict)
    implementation["dependencies"] = ["python==*"]
    with pytest.raises(ValidationError, match="exact name==version"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))

    plan = _approve(_plan())
    implementation = _metrics(plan)[0]["implementation"]
    assert isinstance(implementation, dict)
    implementation["dependencies"] = ["metric-lib@main"]
    with pytest.raises(ValidationError, match="40/64-character-revision"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))


def test_metric_plan_rejects_nonexact_budget_and_insecure_evidence(
    tmp_path: Path,
) -> None:
    plan = _approve(_plan())
    budget = _metrics(plan)[0]["budget"]
    assert isinstance(budget, dict)
    budget["absolute_tolerance"] = 0.01
    with pytest.raises(ValidationError, match="must equal zero"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))

    plan = _approve(_plan())
    budget = _metrics(plan)[0]["budget"]
    assert isinstance(budget, dict)
    budget["minimum_case_coverage"] = 0.9
    with pytest.raises(ValidationError, match="minimum_case_coverage must equal 1"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))

    plan = _approve(_plan())
    _metrics(plan)[0]["evidence"] = ["http://example.test/metric"]
    with pytest.raises(ValidationError, match="must be an HTTPS URL"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))


def test_metric_plan_rejects_status_readiness_mismatch(tmp_path: Path) -> None:
    plan = _plan()
    plan["status"] = "approved"
    with pytest.raises(ValidationError, match="claims approval while blockers remain"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))

    plan = _approve(_plan())
    plan["status"] = "draft"
    with pytest.raises(ValidationError, match="draft even though every family"):
        check_quality_metric_plan(_write_plan(tmp_path, plan))
