"""
tests/test_ingestion.py
-----------------------
Unit tests for src/data_ingestion.py.

These tests use mocking so they never hit the real FEWS NET API.
They verify that the fetcher functions handle responses, pagination,
HTTP errors, and file saving correctly.

Run:
    pytest tests/test_ingestion.py -v
"""

import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion import (
    _get,
    _get_all_pages,
    _save,
    fetch_fewsnet_classifications,
    fetch_fewsnet_population,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_CLASSIFICATION = {
    "fnid": "MG001",
    "country_code": "MG",
    "geographic_unit_name": "Androy",
    "scenario": "CS",
    "value": 3,
    "projection_start": "2024-02-01",
    "projection_end": "2024-04-30",
}

SAMPLE_POPULATION = {
    "fnid": "MG001",
    "country_code": "MG",
    "geographic_unit_name": "Androy",
    "scenario": "CS",
    "phase3_pop": 120000,
    "total_pop": 450000,
}


# ── _get: basic success ───────────────────────────────────────────────────────

class TestGet:
    def test_returns_json_on_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": [SAMPLE_CLASSIFICATION]}
        mock_resp.raise_for_status = MagicMock()

        with patch("src.data_ingestion.requests.get", return_value=mock_resp):
            result = _get("https://fdw.fews.net/api/ipcclassification/", {}, "test")

        assert result == {"results": [SAMPLE_CLASSIFICATION]}

    def test_returns_none_after_three_failures(self):
        import requests as req
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = req.exceptions.HTTPError("500 Server Error")

        with patch("src.data_ingestion.requests.get", return_value=mock_resp):
            with patch("src.data_ingestion.time.sleep"):
                result = _get("https://fdw.fews.net/api/test/", {}, "test")

        assert result is None

    def test_retries_on_request_exception(self):
        import requests as req
        with patch("src.data_ingestion.requests.get",
                   side_effect=req.exceptions.ConnectionError("timeout")):
            with patch("src.data_ingestion.time.sleep"):
                result = _get("https://fdw.fews.net/api/test/", {}, "test")

        assert result is None


# ── _get_all_pages: pagination ────────────────────────────────────────────────

class TestGetAllPages:
    def test_single_page_list_response(self):
        """API returns a plain list (no pagination envelope)."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [SAMPLE_CLASSIFICATION, SAMPLE_CLASSIFICATION]

        with patch("src.data_ingestion.requests.get", return_value=mock_resp):
            results = _get_all_pages(
                "https://fdw.fews.net/api/ipcclassification/", {}, "test"
            )

        assert len(results) == 2

    def test_paginated_response_follows_next(self):
        """API returns paginated envelope with 'next' URL."""
        page1 = MagicMock()
        page1.raise_for_status = MagicMock()
        page1.json.return_value = {
            "results": [SAMPLE_CLASSIFICATION],
            "next": "https://fdw.fews.net/api/ipcclassification/?page=2",
        }

        page2 = MagicMock()
        page2.raise_for_status = MagicMock()
        page2.json.return_value = {
            "results": [SAMPLE_POPULATION],
            "next": None,
        }

        with patch("src.data_ingestion.requests.get", side_effect=[page1, page2]):
            results = _get_all_pages(
                "https://fdw.fews.net/api/ipcclassification/", {}, "test"
            )

        assert len(results) == 2

    def test_empty_results_stops_pagination(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": [], "next": None}

        with patch("src.data_ingestion.requests.get", return_value=mock_resp):
            results = _get_all_pages(
                "https://fdw.fews.net/api/ipcclassification/", {}, "test"
            )

        assert results == []

    def test_returns_empty_list_on_api_failure(self):
        with patch("src.data_ingestion._get", return_value=None):
            results = _get_all_pages(
                "https://fdw.fews.net/api/ipcclassification/", {}, "test"
            )

        assert results == []


# ── _save: file writing ───────────────────────────────────────────────────────

class TestSave:
    def test_saves_json_file(self, tmp_path):
        data = [SAMPLE_CLASSIFICATION]

        with patch("src.data_ingestion.DATA_RAW", str(tmp_path)):
            path = _save(data, "test_output.json")

        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_creates_directory_if_missing(self, tmp_path):
        target_dir = str(tmp_path / "new_subdir")
        data = [{"key": "value"}]

        with patch("src.data_ingestion.DATA_RAW", target_dir):
            _save(data, "test.json")

        assert os.path.isdir(target_dir)


# ── fetch functions: integration with mocked API ──────────────────────────────

class TestFetchFunctions:
    def test_fetch_classifications_returns_list(self):
        mock_data = [SAMPLE_CLASSIFICATION] * 5

        with patch("src.data_ingestion._get_all_pages", return_value=mock_data):
            result = fetch_fewsnet_classifications()

        assert isinstance(result, list)
        assert len(result) == 5

    def test_fetch_population_returns_list(self):
        mock_data = [SAMPLE_POPULATION] * 3

        with patch("src.data_ingestion._get_all_pages", return_value=mock_data):
            result = fetch_fewsnet_population()

        assert isinstance(result, list)
        assert len(result) == 3

    def test_fetch_classifications_uses_mg_country_code(self):
        """Verify the MG country code is passed to the API."""
        with patch("src.data_ingestion._get_all_pages", return_value=[]) as mock_pages:
            fetch_fewsnet_classifications()
            call_args = mock_pages.call_args
            params = call_args[0][1]  # second positional arg is params dict
            assert params.get("country") == "MG"

    def test_fetch_population_uses_mg_country_code(self):
        with patch("src.data_ingestion._get_all_pages", return_value=[]) as mock_pages:
            fetch_fewsnet_population()
            call_args = mock_pages.call_args
            params = call_args[0][1]
            assert params.get("country") == "MG"