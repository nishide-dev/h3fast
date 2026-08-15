"""Tests for all committed JSON Schema documents."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

SCHEMAS = tuple(sorted(Path("schemas").glob("*.schema.json")))


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
