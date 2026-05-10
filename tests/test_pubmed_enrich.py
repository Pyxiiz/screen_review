from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import httpx

from two_stage_screen.pubmed_enrich import (
    enrich_row,
    normalize_pmcid,
    parse_pmc_body_xml,
    parse_pubmed_abstract_xml,
)


PUBMED_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">34212274</PMID>
      <Article>
        <ArticleTitle>Example</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Background line.</AbstractText>
          <AbstractText>Main conclusion.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

PMC_XML = b"""<?xml version="1.0"?>
<article xmlns="http://jats.nlm.nih.gov">
  <body>
    <p>First paragraph.</p>
    <p>Second paragraph.</p>
  </body>
</article>
"""


class TestNormalizePmcid(unittest.TestCase):
    def test_strip_prefix(self) -> None:
        self.assertEqual(normalize_pmcid("PMC8249250"), "8249250")
        self.assertEqual(normalize_pmcid("pmc8249250"), "8249250")

    def test_digits(self) -> None:
        self.assertEqual(normalize_pmcid("8249250"), "8249250")

    def test_empty(self) -> None:
        self.assertIsNone(normalize_pmcid(None))
        self.assertIsNone(normalize_pmcid(""))


class TestParseXml(unittest.TestCase):
    def test_pubmed_structured_abstract(self) -> None:
        text = parse_pubmed_abstract_xml(PUBMED_XML)
        self.assertIn("BACKGROUND: Background line.", text)
        self.assertIn("Main conclusion.", text)

    def test_pmc_body(self) -> None:
        text = parse_pmc_body_xml(PMC_XML)
        self.assertIn("First paragraph.", text)
        self.assertIn("Second paragraph.", text)


class TestEnrichRow(unittest.TestCase):
    def setUp(self) -> None:
        self.sleep_patcher = patch("two_stage_screen.pubmed_enrich.time.sleep")
        self.sleep_patcher.start()

    def tearDown(self) -> None:
        self.sleep_patcher.stop()

    def test_enrich_row_ok(self) -> None:
        client = MagicMock()

        def side_effect(url: str, timeout: float | None = None) -> httpx.Response:
            if "db=pubmed" in url:
                return httpx.Response(200, content=PUBMED_XML)
            if "db=pmc" in url:
                return httpx.Response(200, content=PMC_XML)
            return httpx.Response(404, content=b"")

        client.get.side_effect = side_effect
        out = enrich_row(client, "34212274", "T", "PMC8249250", None)
        self.assertEqual(out["id"], "34212274")
        self.assertEqual(out["title"], "T")
        self.assertIn("Main conclusion.", out["abstract"])
        self.assertIn("First paragraph.", out["full_text"])
        self.assertEqual(out["enrich_abstract_ok"], "1")
        self.assertEqual(out["enrich_fulltext_ok"], "1")
        self.assertEqual(out["enrich_message"], "")
        self.assertEqual(client.get.call_count, 2)

    def test_enrich_row_no_pmcid(self) -> None:
        client = MagicMock()

        def side_effect(url: str, timeout: float | None = None) -> httpx.Response:
            if "db=pubmed" in url:
                return httpx.Response(200, content=PUBMED_XML)
            return httpx.Response(404, content=b"")

        client.get.side_effect = side_effect
        out = enrich_row(client, "34212274", "T", None, None)
        self.assertEqual(out["enrich_fulltext_ok"], "0")
        self.assertEqual(out["full_text"], "")
        self.assertIn("fulltext:no_pmcid", out["enrich_message"])
        self.assertEqual(client.get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
