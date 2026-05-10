"""Integration: screening pipeline with keyword config + enriched PMC row (mock LLM)."""

from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from two_stage_screen.cli import cmd_run
from two_stage_screen.csv_io import read_csv_rows
from two_stage_screen.models import Stage1ModelResponse, Stage2ModelResponse

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
ENRICHED = FIXTURES / "enriched-34212274.csv"
CONFIG = FIXTURES / "screen_keywords_rain_soil_water.yaml"


class TestKeywordScreeningIntegration(unittest.TestCase):
    def setUp(self) -> None:
        if not ENRICHED.is_file():
            self.skipTest(f"Missing {ENRICHED}; run enrich-pubmed first.")

    def test_run_invokes_llm_with_keyword_lists(self) -> None:
        calls: list[tuple[str, str]] = []

        def complete_json_schema(_self, *, system: str, user: str, schema_model):  # noqa: ANN001
            calls.append((system, user))
            name = schema_model.__name__
            if name == "Stage1ModelResponse":
                m = Stage1ModelResponse(
                    decision="include",
                    rationale="Abstract centers on water-based emulsions; no cows/forests focus.",
                    triggered_criteria=["water"],
                )
                return m, m.model_dump_json()
            if name == "Stage2ModelResponse":
                m = Stage2ModelResponse(
                    decision="include",
                    rationale="Full text discusses aqueous phases and water-in-oil systems.",
                    supporting_excerpts=["Water (aqueous) phase is commonly used"],
                )
                return m, m.model_dump_json()
            raise AssertionError(f"Unexpected schema {name}")

        fd, tmp_out = tempfile.mkstemp(prefix="screened_", suffix=".csv")
        os.close(fd)
        fd2, tmp_jsonl = tempfile.mkstemp(prefix="audit_", suffix=".jsonl")
        os.close(fd2)
        prev_key = os.environ.get("OPENAI_API_KEY")
        try:
            os.environ["OPENAI_API_KEY"] = "sk-mock-test-key"
            with patch(
                "two_stage_screen.llm.LLMClient.complete_json_schema",
                new=complete_json_schema,
            ):
                args = argparse.Namespace(
                    config=str(CONFIG),
                    input=str(ENRICHED),
                    csv_out=tmp_out,
                    jsonl_out=tmp_jsonl,
                    jsonl_stage1=None,
                    limit=1,
                )
                rc = cmd_run(args)
            self.assertEqual(rc, 0)

            _, user_s1 = calls[0]
            for kw in ("rain", "soil", "water", "cows", "forests"):
                with self.subTest(kw=kw):
                    self.assertIn(kw, user_s1.lower())

            fn, rows = read_csv_rows(Path(tmp_out))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row.get("stage1_decision"), "include")
            self.assertEqual(row.get("stage2_decision"), "include")
        finally:
            Path(tmp_out).unlink(missing_ok=True)
            Path(tmp_jsonl).unlink(missing_ok=True)
            if prev_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = prev_key


if __name__ == "__main__":
    unittest.main()
