# -*- coding: utf-8 -*-
"""Regression tests for Baostock's A-share index close-review fallback."""

from contextlib import contextmanager

from data_provider.baostock_fetcher import BaostockFetcher


class _Result:
    error_code = "0"
    error_msg = "success"
    fields = [
        "date", "open", "high", "low", "close", "preclose",
        "volume", "amount", "pctChg",
    ]

    def __init__(self):
        self._rows = iter([
            [
                "2026-08-07", "99", "101", "98", "100", "98",
                "1000", "10000000000", "2.04",
            ],
            [
                "2026-08-10", "100", "102", "99", "101", "100",
                "1200", "12000000000", "1.00",
            ],
        ])
        self._current = None

    def next(self):
        try:
            self._current = next(self._rows)
            return True
        except StopIteration:
            return False

    def get_row_data(self):
        return self._current


class _Client:
    def __init__(self):
        self.calls = []

    def query_history_k_data_plus(self, **kwargs):
        self.calls.append(kwargs)
        return _Result()


def test_baostock_returns_six_completed_indices_with_prior_turnover():
    fetcher = BaostockFetcher()
    client = _Client()

    @contextmanager
    def session():
        yield client

    fetcher._baostock_session = session
    rows = fetcher.get_main_indices("cn")

    assert rows is not None
    assert len(rows) == 6
    assert {row["code"] for row in rows} == {
        "sh000001", "sz399001", "sz399006",
        "sh000688", "sh000016", "sh000300",
    }
    assert all(row["current"] == 101.0 for row in rows)
    assert all(row["prev_close"] == 100.0 for row in rows)
    assert all(row["previous_amount"] == 10_000_000_000.0 for row in rows)
    assert all(row["trade_date"] == "2026-08-10" for row in rows)
    assert all(row["previous_trade_date"] == "2026-08-07" for row in rows)
    assert all(row["source"] == "Baostock (daily)" for row in rows)
    assert all(call["adjustflag"] == "3" for call in client.calls)
