"""The committed JSON Schema files must match the Pydantic contract exactly."""

from __future__ import annotations

from pathlib import Path

from impactlens.cli import render_schemas

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def test_committed_schemas_are_current():
    for name, text in render_schemas().items():
        committed = SCHEMA_DIR / name
        assert committed.is_file(), f"{name} missing; run `impactlens schema`"
        assert committed.read_text(encoding="utf-8") == text, (
            f"{name} is stale; run `impactlens schema`"
        )


def test_contract_pins_unvalidated_fields():
    schema = (SCHEMA_DIR / "predict-response-1.0.0.schema.json").read_text(encoding="utf-8")
    assert '"part"' in schema and '"severity"' in schema
    assert '"unknown"' in schema
