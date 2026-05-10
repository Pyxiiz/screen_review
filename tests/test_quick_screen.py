from __future__ import annotations

import unittest

from two_stage_screen.quick_screen import (
    build_keyword_screen_config,
    parse_comma_separated_keywords,
)


class TestParseCommaSeparated(unittest.TestCase):
    def test_splits_and_trims(self) -> None:
        self.assertEqual(parse_comma_separated_keywords("a, b , c"), ["a", "b", "c"])

    def test_empty(self) -> None:
        self.assertEqual(parse_comma_separated_keywords(""), [])
        self.assertEqual(parse_comma_separated_keywords(None), [])


class TestBuildKeywordConfig(unittest.TestCase):
    def test_keywords_in_rubric(self) -> None:
        cfg = build_keyword_screen_config(["rain", "water"], ["cows"])
        self.assertEqual(cfg.inclusion_keywords, ["rain", "water"])
        self.assertEqual(cfg.exclusion_keywords, ["cows"])
        self.assertIn("rain", cfg.research_question)
        self.assertIn("cows", cfg.eligibility_text)


if __name__ == "__main__":
    unittest.main()
