"""
Tests for the ingestion pipeline.

Run: pytest tests/test_ingestion.py -v
Uses real S2 API (integration tests) — requires SEMANTIC_SCHOLAR_API_KEY in env.
"""

import os
import tempfile
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Storage tests (no network) ────────────────────────────────────────────────

class TestPaperStore:
    def _make_store(self, tmp_path):
        from ingestion.papers.storage import PaperStore
        store = PaperStore(db_path=tmp_path / "test.db")
        store.connect()
        return store

    def _sample_paper(self, pid="abc123", year=2022, has_pdf=True):
        return {
            "paperId": pid,
            "title": f"Test paper {pid}",
            "abstract": "A study of bone mechanics.",
            "year": year,
            "venue": "Journal of Bone Research",
            "citationCount": 10,
            "referenceCount": 5,
            "openAccessPdf": {"url": "https://example.com/paper.pdf"} if has_pdf else None,
            "authors": [{"authorId": "1", "name": "Jane Doe"}],
            "externalIds": {"DOI": "10.1234/bone.2022"},
            "s2FieldsOfStudy": [{"category": "Medicine"}],
            "publicationDate": "2022-06-01",
            "publicationTypes": ["JournalArticle"],
        }

    def test_upsert_new_paper(self, tmp_path):
        store = self._make_store(tmp_path)
        paper = self._sample_paper()
        is_new = store.upsert_paper(paper, keyword="bone mechanics")
        assert is_new is True
        assert store.total_papers() == 1
        store.close()

    def test_upsert_duplicate_merges_keywords(self, tmp_path):
        store = self._make_store(tmp_path)
        paper = self._sample_paper()
        store.upsert_paper(paper, keyword="bone mechanics")
        is_new = store.upsert_paper(paper, keyword="cortical bone")
        assert is_new is False  # not new
        row = store.get_paper("abc123")
        import json
        matched = json.loads(row["keywords_matched"])
        assert "bone mechanics" in matched
        assert "cortical bone" in matched
        store.close()

    def test_stats(self, tmp_path):
        store = self._make_store(tmp_path)
        store.upsert_paper(self._sample_paper("p1", has_pdf=True), "kw1")
        store.upsert_paper(self._sample_paper("p2", has_pdf=False), "kw2")
        stats = store.stats()
        assert stats["total"] == 2
        assert stats["with_open_pdf"] == 1
        store.close()

    def test_set_local_pdf_path(self, tmp_path):
        store = self._make_store(tmp_path)
        store.upsert_paper(self._sample_paper(), "kw")
        store.set_local_pdf_path("abc123", "/data/raw/papers/2022/abc123.pdf")
        row = store.get_paper("abc123")
        assert row["pdf_local_path"] == "/data/raw/papers/2022/abc123.pdf"
        store.close()


# ── API client tests (mocked) ────────────────────────────────────────────────

class TestSemanticScholarClient:
    def _mock_response(self, papers, total=None):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": papers,
            "total": total or len(papers),
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_search_returns_papers(self):
        from ingestion.papers.semantic_scholar import SemanticScholarClient
        client = SemanticScholarClient(api_key="test-key")

        fake_papers = [
            {"paperId": f"id{i}", "title": f"Paper {i}", "abstract": ""}
            for i in range(5)
        ]
        with patch.object(client.session, "get", return_value=self._mock_response(fake_papers, total=5)):
            results = list(client.search_papers("bone mechanics", max_papers=5))
        assert len(results) == 5
        assert results[0]["paperId"] == "id0"

    def test_search_respects_max_papers(self):
        from ingestion.papers.semantic_scholar import SemanticScholarClient
        client = SemanticScholarClient(api_key="test-key")

        fake_papers = [{"paperId": f"id{i}", "title": f"P{i}"} for i in range(100)]
        with patch.object(client.session, "get", return_value=self._mock_response(fake_papers, total=1000)):
            results = list(client.search_papers("bone", max_papers=50))
        assert len(results) == 50

    def test_search_empty_result(self):
        from ingestion.papers.semantic_scholar import SemanticScholarClient
        client = SemanticScholarClient(api_key="test-key")

        with patch.object(client.session, "get", return_value=self._mock_response([], total=0)):
            results = list(client.search_papers("xyznonexistent"))
        assert results == []


# ── Integration test (skipped without API key) ───────────────────────────────

@pytest.mark.skipif(
    not os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
    reason="SEMANTIC_SCHOLAR_API_KEY not set",
)
def test_live_search_bone():
    from ingestion.papers.semantic_scholar import SemanticScholarClient
    client = SemanticScholarClient()
    results = list(client.search_papers("cortical bone fracture", max_papers=5))
    assert len(results) > 0
    assert all("title" in p for p in results)
    assert all("paperId" in p for p in results)
