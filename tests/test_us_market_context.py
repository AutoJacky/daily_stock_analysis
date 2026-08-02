# -*- coding: utf-8 -*-
"""Regression tests for the strict US market-recap evidence contract."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_provider.yfinance_fetcher import YfinanceFetcher
from src.core.market_profile import US_PROFILE
from src.market_analyzer import MarketAnalyzer, MarketIndex, MarketOverview


def _price_metric(symbol: str, name: str, change_pct: float, *, as_of: str = "2026-07-30"):
    return {
        "symbol": symbol,
        "name": name,
        "last": 100.0,
        "previous_close": 100.0 / (1 + change_pct / 100),
        "change_pct": change_pct,
        "volume": 120.0,
        "average_volume_20": 100.0,
        "volume_ratio_20d": 1.2,
        "as_of": as_of,
        "source": "Yahoo Finance/yfinance",
    }


def _fred_metric(series_id: str):
    unit = "%" if series_id in {"DGS2", "DGS10"} else "index"
    return {
        "series_id": series_id,
        "name": series_id,
        "value": 4.25 if unit == "%" else 120.0,
        "previous_value": 4.20 if unit == "%" else 119.8,
        "change": 0.05 if unit == "%" else 0.2,
        "unit": unit,
        "as_of": "2026-07-30",
        "previous_as_of": "2026-07-29",
        "source": "Federal Reserve Bank of St. Louis (FRED)",
    }


def _treasury_metric(series_id: str):
    metric = _fred_metric(series_id)
    metric["source"] = "U.S. Department of the Treasury"
    metric["url"] = (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        "TextView"
    )
    return metric


def _complete_context(as_of: str = "2026-07-30"):
    proxies = {
        "SPY": _price_metric("SPY", "标普500 ETF", 0.8, as_of=as_of),
        "RSP": _price_metric("RSP", "标普500等权 ETF", 0.6, as_of=as_of),
        "IWM": _price_metric("IWM", "罗素2000 ETF", 1.0, as_of=as_of),
        "QQQ": _price_metric("QQQ", "纳斯达克100 ETF", 1.1, as_of=as_of),
    }
    sectors = [
        _price_metric(symbol, name, 1.5 - index * 0.25, as_of=as_of)
        for index, (symbol, name) in enumerate(YfinanceFetcher._US_SECTOR_ETFS.items())
    ]
    return {
        "as_of": as_of,
        "proxies": proxies,
        "participation": {
            "equal_weight_vs_cap_weight_pct": -0.2,
            "small_cap_vs_large_cap_pct": 0.2,
            "nasdaq100_vs_large_cap_pct": 0.3,
            "sector_advancers": 7,
            "sector_decliners": 4,
            "sector_flat": 0,
            "sector_coverage": 11,
            "spy_volume_ratio_20d": 1.2,
            "as_of": as_of,
        },
        "sector_rankings": {
            "top": sectors[:5],
            "bottom": list(reversed(sectors[-5:])),
            "coverage": 11,
            "universe": 11,
        },
        "macro": {
            "DGS2": _fred_metric("DGS2"),
            "DGS10": _fred_metric("DGS10"),
            "DTWEXBGS": _fred_metric("DTWEXBGS"),
        },
        "quality": {
            "status": "ok",
            "core_ready": True,
            "proxy_ready": True,
            "liquidity_ready": True,
            "sector_ready": True,
            "macro_ready": True,
            "missing_core_fields": [],
        },
        "sources": [],
    }


def _us_analyzer() -> MarketAnalyzer:
    analyzer = MarketAnalyzer.__new__(MarketAnalyzer)
    analyzer.config = SimpleNamespace(
        report_language="zh",
        market_review_color_scheme="red_up",
        market_review_strict_data_only=False,
    )
    analyzer.region = "us"
    analyzer.profile = US_PROFILE
    return analyzer


def _overview(context=None) -> MarketOverview:
    date = "2026-07-30"
    return MarketOverview(
        date=date,
        indices=[
            MarketIndex(code="SPX", name="标普500", current=6400, change_pct=0.7, trade_date=date),
            MarketIndex(code="IXIC", name="纳斯达克", current=23000, change_pct=1.0, trade_date=date),
            MarketIndex(code="DJI", name="道琼斯", current=46000, change_pct=0.3, trade_date=date),
            MarketIndex(code="RUT", name="罗素2000", current=2500, change_pct=0.9, trade_date=date),
            MarketIndex(code="VIX", name="VIX", current=17, change_pct=-3.0, trade_date=date),
        ],
        indices_attempted=True,
        us_context_attempted=True,
        us_market_context=context or _complete_context(),
    )


def test_build_us_price_metric_calculates_return_volume_ratio_and_date():
    dates = pd.date_range("2026-07-01", periods=22, freq="B")
    frame = pd.DataFrame(
        {
            "Close": [100.0] * 21 + [102.0],
            "Volume": [100.0] * 21 + [150.0],
        },
        index=dates,
    )

    metric = YfinanceFetcher._build_us_price_metric(
        frame,
        symbol="SPY",
        name="标普500 ETF",
    )

    assert metric is not None
    assert metric["change_pct"] == pytest.approx(2.0)
    assert metric["volume_ratio_20d"] == pytest.approx(1.5)
    assert metric["as_of"] == dates[-1].date().isoformat()


def test_get_us_market_context_builds_complete_verified_contract():
    fetcher = YfinanceFetcher()
    metrics = {}
    for symbol, name in {
        **fetcher._US_MARKET_PROXIES,
        **fetcher._US_SECTOR_ETFS,
    }.items():
        metrics[symbol] = _price_metric(symbol, name, 1.0)

    with patch.object(fetcher, "_fetch_us_etf_metrics", return_value=metrics), patch.object(
        fetcher,
        "_fetch_fred_metric",
        side_effect=lambda series_id: _fred_metric(series_id),
    ):
        context = fetcher.get_us_market_context()

    assert context is not None
    assert context["quality"]["core_ready"] is True
    assert context["quality"]["liquidity_ready"] is True
    assert context["sector_rankings"]["coverage"] == 11
    assert context["participation"]["sector_advancers"] == 11
    assert set(context["macro"]) == {"DGS2", "DGS10", "DTWEXBGS"}


def test_fred_metric_parses_official_observation_date_header():
    response = MagicMock()
    response.read.return_value = (
        b"observation_date,DGS2\n"
        b"2026-07-28,4.20\n"
        b"2026-07-29,\n"
        b"2026-07-30,4.25\n"
    )
    response.__enter__.return_value = response
    response.__exit__.return_value = False

    with patch("data_provider.yfinance_fetcher.urlopen", return_value=response):
        metric = YfinanceFetcher._fetch_fred_metric("DGS2")

    assert metric is not None
    assert metric["as_of"] == "2026-07-30"
    assert metric["previous_as_of"] == "2026-07-28"
    assert metric["value"] == pytest.approx(4.25)
    assert metric["change"] == pytest.approx(0.05)


def test_treasury_fallback_parses_official_xml_and_excludes_future_rows():
    response = MagicMock()
    response.read.return_value = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
 xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"
 xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata">
 <entry><content type="application/xml"><m:properties>
  <d:NEW_DATE m:type="Edm.DateTime">2026-07-29T00:00:00</d:NEW_DATE>
  <d:BC_2YEAR m:type="Edm.Double">4.20</d:BC_2YEAR>
  <d:BC_10YEAR m:type="Edm.Double">4.50</d:BC_10YEAR>
 </m:properties></content></entry>
 <entry><content type="application/xml"><m:properties>
  <d:NEW_DATE m:type="Edm.DateTime">2026-07-30T00:00:00</d:NEW_DATE>
  <d:BC_2YEAR m:type="Edm.Double">4.25</d:BC_2YEAR>
  <d:BC_10YEAR m:type="Edm.Double">4.55</d:BC_10YEAR>
 </m:properties></content></entry>
 <entry><content type="application/xml"><m:properties>
  <d:NEW_DATE m:type="Edm.DateTime">2026-07-31T00:00:00</d:NEW_DATE>
  <d:BC_2YEAR m:type="Edm.Double">9.99</d:BC_2YEAR>
  <d:BC_10YEAR m:type="Edm.Double">9.99</d:BC_10YEAR>
 </m:properties></content></entry>
</feed>"""
    response.__enter__.return_value = response
    response.__exit__.return_value = False

    with patch("data_provider.yfinance_fetcher.urlopen", return_value=response):
        metrics = YfinanceFetcher._fetch_treasury_yield_metrics("2026-07-30")

    assert set(metrics) == {"DGS2", "DGS10"}
    assert metrics["DGS2"]["value"] == pytest.approx(4.25)
    assert metrics["DGS2"]["previous_value"] == pytest.approx(4.20)
    assert metrics["DGS10"]["value"] == pytest.approx(4.55)
    assert metrics["DGS10"]["as_of"] == "2026-07-30"
    assert metrics["DGS10"]["source"] == "U.S. Department of the Treasury"


def test_us_context_uses_treasury_fallback_when_fred_yields_are_unavailable():
    fetcher = YfinanceFetcher()
    metrics = {}
    for symbol, name in {
        **fetcher._US_MARKET_PROXIES,
        **fetcher._US_SECTOR_ETFS,
    }.items():
        metrics[symbol] = _price_metric(symbol, name, 1.0)

    with patch.object(fetcher, "_fetch_us_etf_metrics", return_value=metrics), patch.object(
        fetcher,
        "_fetch_fred_metric",
        return_value=None,
    ), patch.object(
        fetcher,
        "_fetch_treasury_yield_metrics",
        return_value={
            "DGS2": _treasury_metric("DGS2"),
            "DGS10": _treasury_metric("DGS10"),
        },
    ) as treasury_fetch:
        context = fetcher.get_us_market_context()

    treasury_fetch.assert_called_once_with("2026-07-30")
    assert context is not None
    assert context["quality"]["core_ready"] is True
    assert context["quality"]["macro_ready"] is True
    assert context["macro"]["DGS2"]["source"] == "U.S. Department of the Treasury"
    assert context["macro"]["DGS10"]["source"] == "U.S. Department of the Treasury"
    assert {item["name"] for item in context["sources"]} == {
        "Yahoo Finance/yfinance",
        "U.S. Department of the Treasury",
    }


def test_us_quality_requires_same_trade_date_and_all_core_layers():
    analyzer = _us_analyzer()
    quality = analyzer._assess_market_data_quality(_overview())

    assert quality["core_data_ready"] is True
    assert quality["trade_date_aligned"] is True
    assert quality["valid_index_count"] == 4
    assert quality["macro_available"] is True


def test_us_review_is_always_deterministic_and_does_not_allow_llm_fact_rewrites():
    analyzer = _us_analyzer()
    analyzer.analyzer = MagicMock()
    analyzer.analyzer.is_available.return_value = True
    analyzer.analyzer.generate_text.return_value = (
        "微软财报超预期，制造业PMI改善，标普关注7450点压力。"
    )

    report = analyzer.generate_market_review(_overview(), [])

    analyzer.analyzer.generate_text.assert_not_called()
    assert "美股大盘复盘（严格数据版）" in report
    assert "本次未取得带来源的有效新闻" in report
    assert "微软财报" not in report
    assert "PMI" not in report
    assert "7450" not in report
    assert "官方来源：FRED" in report


def test_us_quality_fails_closed_when_etf_date_does_not_match_indices():
    analyzer = _us_analyzer()
    context = _complete_context(as_of="2026-07-29")
    quality = analyzer._assess_market_data_quality(_overview(context))

    assert quality["core_data_ready"] is False
    assert "us_trade_date_alignment" in quality["missing_core_fields"]
    report = analyzer.generate_market_review(_overview(context), [])
    assert "数据校验未通过" in report
    assert "禁止生成方向、仓位和买卖计划" in report


def test_us_prompt_requires_policy_earnings_sources_and_fact_inference_separation():
    analyzer = _us_analyzer()
    prompt = analyzer._build_review_prompt(_overview(), [])

    assert "联储、财政、监管、关税、政治/地缘" in prompt
    assert "财报与管理层指引" in prompt
    assert "行情事实、带来源新闻、分析推断" in prompt
    assert "禁止写成交易所全市场涨跌家数" in prompt


def test_workflow_has_separate_us_close_full_market_filtered_schedule():
    # pathlib keeps this assertion independent from a YAML parser dependency.
    from pathlib import Path

    text = Path(".github/workflows/00-daily-analysis.yml").read_text(encoding="utf-8")
    assert "cron: '15 21 * * 1-5'" in text
    assert 'MODE="full"' in text
    assert 'export MARKET_REVIEW_REGION="us"' in text
    assert 'export STOCK_MARKET_FILTER="us"' in text
    assert 'export STOCK_MARKET_FILTER="cn"' in text
    assert "cron: '15 8 * * 1-5'" in text
    assert 'export STOCK_MARKET_FILTER="hk,jp"' in text
