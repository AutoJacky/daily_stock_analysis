# -*- coding: utf-8 -*-
"""Regression tests for the A-share completed-session index contract."""

from unittest.mock import patch

import pandas as pd

from data_provider.akshare_fetcher import AkshareFetcher


def _daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-08-07", "2026-08-10"],
            "open": [99.0, 100.5],
            "close": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [98.0, 100.0],
            "volume": [1_000.0, 1_200.0],
            "amount": [10_000_000_000.0, 12_000_000_000.0],
        }
    )


def test_cn_daily_indices_retry_only_failed_symbol_and_preserve_prior_turnover():
    """某个指数短暂断连时，补抓后仍应返回六指数完整合约。"""

    fetcher = AkshareFetcher(sleep_min=0, sleep_max=0)
    attempts = {}

    def fetch_daily(*, symbol, start_date, end_date):
        attempts[symbol] = attempts.get(symbol, 0) + 1
        if symbol == "sh000001" and attempts[symbol] == 1:
            raise ConnectionError("transient disconnect")
        return _daily_frame()

    with patch(
        "akshare.stock_zh_index_daily_em",
        side_effect=fetch_daily,
    ), patch.object(fetcher, "_set_random_user_agent"), patch.object(
        fetcher, "_enforce_rate_limit"
    ), patch("data_provider.akshare_fetcher.time.sleep"):
        rows = fetcher.get_main_indices("cn")

    assert rows is not None
    assert len(rows) == 6
    assert attempts["sh000001"] == 2
    assert all(row["trade_date"] == "2026-08-10" for row in rows)
    assert all(row["previous_trade_date"] == "2026-08-07" for row in rows)
    assert all(row["previous_amount"] == 10_000_000_000.0 for row in rows)
    assert all(row["source"] == "Eastmoney via AkShare (daily)" for row in rows)
