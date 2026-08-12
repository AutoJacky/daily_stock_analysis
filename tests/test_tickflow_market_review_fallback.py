# -*- coding: utf-8 -*-
"""Regression tests for TickFlow market-review manager fallback."""

import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.base import DataFetcherManager


class _DummyFetcher:
    def __init__(self, name, indices=None, stats=None):
        self.name = name
        self.priority = 1
        self.indices = indices
        self.stats = stats
        self.index_calls = 0
        self.stats_calls = 0

    def get_main_indices(self, region="cn"):
        self.index_calls += 1
        return self.indices

    def get_market_stats(self):
        self.stats_calls += 1
        return self.stats


class _DummyTickFlowFetcher:
    def __init__(self, indices=None, stats=None, error=None):
        self.indices = indices
        self.stats = stats
        self.error = error
        self.closed = False

    def get_main_indices(self, region="cn"):
        if self.error is not None:
            raise self.error
        return self.indices

    def get_market_stats(self):
        if self.error is not None:
            raise self.error
        return self.stats

    def close(self):
        self.closed = True


class TestTickFlowMarketReviewFallback(unittest.TestCase):
    def test_manager_prefers_tickflow_indices_when_available(self):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        fallback = _DummyFetcher("AkshareFetcher", indices=[{"code": "fallback"}])
        manager._fetchers = [fallback]
        manager._get_tickflow_fetcher = lambda: _DummyTickFlowFetcher(
            indices=[{"code": "000001"}]
        )

        data = DataFetcherManager.get_main_indices(manager, region="cn")

        self.assertEqual(data[0]["code"], "000001")
        self.assertEqual(data[0]["source"], "TickFlowFetcher")
        self.assertTrue(data[0]["fetched_at"])
        self.assertEqual(fallback.index_calls, 0)

    def test_manager_falls_back_when_tickflow_indices_fail(self):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        fallback = _DummyFetcher("AkshareFetcher", indices=[{"code": "fallback"}])
        manager._fetchers = [fallback]
        manager._get_tickflow_fetcher = lambda: _DummyTickFlowFetcher(
            error=RuntimeError("tickflow down")
        )

        data = DataFetcherManager.get_main_indices(manager, region="cn")

        self.assertEqual(data[0]["code"], "fallback")
        self.assertEqual(data[0]["source"], "AkshareFetcher")
        self.assertTrue(data[0]["fetched_at"])
        self.assertEqual(fallback.index_calls, 1)

    def test_manager_falls_back_when_tickflow_indices_missing(self):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        fallback = _DummyFetcher("AkshareFetcher", indices=[{"code": "fallback"}])
        manager._fetchers = [fallback]
        manager._get_tickflow_fetcher = lambda: _DummyTickFlowFetcher(
            indices=None
        )

        data = DataFetcherManager.get_main_indices(manager, region="cn")

        self.assertEqual(data[0]["code"], "fallback")
        self.assertEqual(data[0]["source"], "AkshareFetcher")
        self.assertTrue(data[0]["fetched_at"])
        self.assertEqual(fallback.index_calls, 1)

    def test_manager_skips_tickflow_for_non_cn_indices(self):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        fallback = _DummyFetcher("YfinanceFetcher", indices=[{"code": "^GSPC"}])
        manager._fetchers = [fallback]
        manager._get_tickflow_fetcher = lambda: self.fail(
            "TickFlow should not be called for non-CN indices"
        )

        data = DataFetcherManager.get_main_indices(manager, region="us")

        self.assertEqual(data[0]["code"], "^GSPC")
        self.assertEqual(data[0]["source"], "YfinanceFetcher")
        self.assertTrue(data[0]["fetched_at"])
        self.assertEqual(fallback.index_calls, 1)

    def test_manager_merges_same_session_sources_to_restore_prior_turnover(self):
        codes = [
            ("sh000001", "上证指数"),
            ("sz399001", "深证成指"),
            ("sz399006", "创业板指"),
            ("sh000688", "科创50"),
            ("sh000016", "上证50"),
            ("sh000300", "沪深300"),
        ]

        def row(code, name, *, with_prior=False):
            item = {
                "code": code,
                "name": name,
                "current": 101.0,
                "prev_close": 100.0,
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "amount": 12_000_000_000.0,
                "trade_date": "2026-08-10",
            }
            if with_prior:
                item.update({
                    "previous_amount": 10_000_000_000.0,
                    "previous_trade_date": "2026-08-07",
                })
            return item

        realtime = _DummyFetcher(
            "AkshareFetcher",
            indices=[row(code, name) for code, name in codes],
        )
        historical = _DummyFetcher(
            "BaostockFetcher",
            # Baostock does not expose STAR 50; the manager must retain that
            # same-session quote from the realtime provider.
            indices=[
                row(code, name, with_prior=True)
                for code, name in codes
                if code != "sh000688"
            ],
        )
        manager = DataFetcherManager.__new__(DataFetcherManager)
        manager._fetchers = [realtime, historical]
        manager._get_tickflow_fetcher = lambda: None

        data = DataFetcherManager.get_main_indices(manager, region="cn")

        assert len(data) == 6
        by_code = {item["code"]: item for item in data}
        assert by_code["sh000001"]["previous_amount"] == 10_000_000_000.0
        assert by_code["sz399001"]["previous_amount"] == 10_000_000_000.0
        assert by_code["sh000688"]["source"] == "AkshareFetcher"
        assert by_code["sh000001"]["source"] == "BaostockFetcher"

    def test_manager_keeps_latest_close_and_uses_lagged_exchange_amount_as_prior(self):
        codes = [
            ("sh000001", "上证指数"),
            ("sz399001", "深证成指"),
            ("sz399006", "创业板指"),
            ("sh000688", "科创50"),
            ("sh000016", "上证50"),
            ("sh000300", "沪深300"),
        ]

        def row(code, name, date, current, amount):
            return {
                "code": code,
                "name": name,
                "current": current,
                "prev_close": current - 1,
                "open": current - 0.5,
                "high": current + 1,
                "low": current - 2,
                "amount": amount,
                "trade_date": date,
            }

        close_rows = [
            row(code, name, "2026-08-12", 102.0, 12_000_000_000.0)
            for code, name in codes
        ]
        lagged_daily_rows = [
            row(code, name, "2026-08-11", 101.0, 10_000_000_000.0)
            for code, name in codes
            if code != "sh000688"
        ]
        manager = DataFetcherManager.__new__(DataFetcherManager)
        manager._fetchers = [
            _DummyFetcher("TencentFetcher", indices=close_rows),
            _DummyFetcher("BaostockFetcher", indices=lagged_daily_rows),
        ]
        manager._get_tickflow_fetcher = lambda: None

        data = DataFetcherManager.get_main_indices(manager, region="cn")

        assert len(data) == 6
        assert {item["trade_date"] for item in data} == {"2026-08-12"}
        by_code = {item["code"]: item for item in data}
        assert by_code["sh000001"]["current"] == 102.0
        assert by_code["sh000001"]["previous_amount"] == 10_000_000_000.0
        assert by_code["sh000001"]["previous_trade_date"] == "2026-08-11"
        assert by_code["sz399001"]["previous_amount"] == 10_000_000_000.0

    def test_manager_falls_back_when_tickflow_market_stats_fails(self):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        fallback = _DummyFetcher(
            "AkshareFetcher",
            stats={"up_count": 1, "down_count": 2, "flat_count": 3},
        )
        manager._fetchers = [fallback]
        manager._get_tickflow_fetcher = lambda: _DummyTickFlowFetcher(
            error=RuntimeError("tickflow down")
        )

        data = DataFetcherManager.get_market_stats(manager, purpose="market_review:cn")

        self.assertEqual(data["up_count"], 1)
        self.assertEqual(fallback.stats_calls, 1)

    @patch("src.config.get_config")
    def test_manager_times_out_hung_market_stats_provider_and_falls_back(
        self, mock_get_config
    ):
        mock_get_config.return_value = SimpleNamespace(
            tickflow_api_key=None,
            market_stats_provider_timeout_seconds=0.01,
        )
        manager = DataFetcherManager.__new__(DataFetcherManager)

        class _HungFetcher(_DummyFetcher):
            def get_market_stats(self):
                self.stats_calls += 1
                time.sleep(0.2)
                return None

        hung = _HungFetcher("AkshareFetcher")
        fallback = _DummyFetcher(
            "TencentFetcher",
            stats={"up_count": 3, "down_count": 1, "flat_count": 0},
        )
        manager._fetchers = [hung, fallback]

        started_at = time.monotonic()
        data = DataFetcherManager.get_market_stats(
            manager,
            purpose="market_review:cn",
        )

        self.assertEqual(data["up_count"], 3)
        self.assertEqual(hung.stats_calls, 1)
        self.assertEqual(fallback.stats_calls, 1)
        self.assertLess(time.monotonic() - started_at, 0.15)

    @patch("src.config.get_config")
    def test_manager_skips_tickflow_without_api_key(self, mock_get_config):
        mock_get_config.return_value = SimpleNamespace(tickflow_api_key=None)

        manager = DataFetcherManager.__new__(DataFetcherManager)
        fallback = _DummyFetcher(
            "AkshareFetcher",
            stats={"up_count": 2, "down_count": 1, "flat_count": 0},
        )
        manager._fetchers = [fallback]

        data = DataFetcherManager.get_market_stats(manager)

        self.assertEqual(data["up_count"], 2)
        self.assertEqual(fallback.stats_calls, 1)

    def test_manager_close_releases_tickflow_fetcher(self):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        tickflow_fetcher = _DummyTickFlowFetcher(indices=[{"code": "000001"}])
        manager._tickflow_fetcher = tickflow_fetcher
        manager._tickflow_api_key = "tf-secret"
        manager._tickflow_lock = None

        DataFetcherManager.close(manager)

        self.assertTrue(tickflow_fetcher.closed)
        self.assertIsNone(manager._tickflow_fetcher)
        self.assertIsNone(manager._tickflow_api_key)


if __name__ == "__main__":
    unittest.main()
