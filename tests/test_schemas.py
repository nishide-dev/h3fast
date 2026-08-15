"""Tests for all committed JSON Schema documents."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

SCHEMAS = tuple(sorted(Path("schemas").glob("*.schema.json")))
COMPLIANCE_RECORDS = (
    Path("compliance/release-gates/initial-runtime.json"),
    Path("compliance/territories/initial-runtime.json"),
)
FORMAL_QUALITY_SET = Path("benchmarks/quality/formal-quality-set.json")


def _evidence_values(value: object) -> tuple[str, ...]:
    found: list[str] = []
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, item in current.items():
                if key == "evidence" and isinstance(item, list):
                    found.extend(value for value in item if isinstance(value, str))
                else:
                    pending.append(item)
        elif isinstance(current, list):
            pending.extend(current)
    return tuple(found)


@pytest.mark.parametrize("schema_path", SCHEMAS, ids=lambda path: path.name)
def test_schema_matches_draft_2020_12(schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)


def test_committed_quality_reference_matches_schema() -> None:
    schema = json.loads(
        Path("schemas/quality-reference.schema.json").read_text(encoding="utf-8")
    )
    reference = json.loads(
        Path("benchmarks/quality/exact-smoke-001-reference.json").read_text(
            encoding="utf-8"
        )
    )

    Draft202012Validator(schema).validate(reference)


def test_committed_release_gate_matches_schema() -> None:
    schema = json.loads(
        Path("schemas/release-gate.schema.json").read_text(encoding="utf-8")
    )
    record = json.loads(
        Path("compliance/release-gates/initial-runtime.json").read_text(
            encoding="utf-8"
        )
    )

    Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    ).validate(record)


def test_committed_territory_inventory_matches_schema() -> None:
    schema = json.loads(
        Path("schemas/territory-inventory.schema.json").read_text(encoding="utf-8")
    )
    inventory = json.loads(
        Path("compliance/territories/initial-runtime.json").read_text(encoding="utf-8")
    )

    Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    ).validate(inventory)


def test_committed_formal_quality_set_matches_schema() -> None:
    schema = json.loads(
        Path("schemas/formal-quality-set.schema.json").read_text(encoding="utf-8")
    )
    record = json.loads(FORMAL_QUALITY_SET.read_text(encoding="utf-8"))

    Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    ).validate(record)


@pytest.mark.parametrize("record_path", COMPLIANCE_RECORDS, ids=lambda path: path.name)
def test_committed_compliance_local_evidence_exists(record_path: Path) -> None:
    record = json.loads(record_path.read_text(encoding="utf-8"))

    missing = sorted(
        evidence
        for evidence in _evidence_values(record)
        if not evidence.startswith("https://")
        and not Path(evidence.split("#", 1)[0]).is_file()
    )

    assert missing == []
