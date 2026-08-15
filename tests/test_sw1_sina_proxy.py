# -*- coding: utf-8 -*-
"""Deterministic tests for the strict Sina SW1 proxy fallback."""

from unittest.mock import patch

import pytest

from data_provider.akshare_fetcher import AkshareFetcher
from data_provider.sw1_sina_proxy import (
    SW1_2021_NODES,
    SinaSw1ProxyFetcher,
    _IndustrySnapshot,
)


def _constituent_rows(change_pct: float):
    previous = 100.0
    current = previous * (1.0 + change_pct / 100.0)
    return [
        {
            "symbol": f"sh6000{index:02d}",
            "trade": current,
            "settlement": previous,
            "nmc": 1000.0 + index,
            "volume": 10000.0 + index,
        }
        for index in range(4)
    ]


def test_complete_sw1_proxy_is_ranked_and_fully_labelled():
    fetcher = SinaSw1ProxyFetcher(workers=1)
    changes = {code: float(index - 15) / 10.0 for index, (code, _) in enumerate(SW1_2021_NODES)}

    def fake_rows(code):
        return _constituent_rows(changes[code])

    with patch.object(fetcher, "_fetch_node_rows", side_effect=fake_rows), patch.object(
        fetcher,
        "_fetch_quote_dates",
        return_value={f"sh6000{index:02d}": "2026-08-14" for index in range(4)},
    ):
        top, bottom = fetcher.get_rankings(n=5)

    assert len(top) == len(bottom) == 5
    assert top[0]["change_pct"] > top[-1]["change_pct"]
    assert bottom[0]["change_pct"] < bottom[-1]["change_pct"]
    assert all(item["classification"] == "SW1_PROXY" for item in top + bottom)
    assert all(item["quote_date"] == "2026-08-14" for item in top + bottom)
    assert all(item["industry_count"] == 31 for item in top + bottom)
    assert all(item["minimum_universe_coverage"] == 1.0 for item in top + bottom)


def test_sw1_proxy_rejects_cross_session_representatives():
    fetcher = SinaSw1ProxyFetcher(workers=1)
    node_index = {code: index for index, (code, _) in enumerate(SW1_2021_NODES)}

    def fake_snapshot(code, name):
        index = node_index[code]
        return _IndustrySnapshot(
            code=code,
            name=name,
            change_pct=1.0,
            valid_count=4,
            raw_count=4,
            representatives=(f"sh{600000 + index:06d}",),
        )

    dates = {
        f"sh{600000 + index:06d}": "2026-08-14" if index < 30 else "2026-08-13"
        for index in range(31)
    }
    with patch.object(fetcher, "_build_snapshot", side_effect=fake_snapshot), patch.object(
        fetcher,
        "_fetch_quote_dates",
        return_value=dates,
    ):
        with pytest.raises(RuntimeError, match="not aligned"):
            fetcher.get_rankings(n=5)


def test_akshare_sector_rankings_falls_back_to_strict_sw1_proxy():
    expected = ([{"name": "电子"}], [{"name": "煤炭"}])
    fetcher = AkshareFetcher(sleep_min=0, sleep_max=0)

    with patch(
        "data_provider.akshare_fetcher._akshare_call_with_timeout",
        side_effect=TimeoutError("official endpoint timeout"),
    ), patch(
        "data_provider.akshare_fetcher.SinaSw1ProxyFetcher.get_rankings",
        return_value=expected,
    ) as proxy:
        assert fetcher.get_sector_rankings(5) == expected

    proxy.assert_called_once_with(n=5)
