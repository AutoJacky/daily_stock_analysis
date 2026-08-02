# -*- coding: utf-8 -*-
"""Regression tests for evidence-directed report self-healing."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from data_provider.base import DataFetcherManager
from src.analyzer import AnalysisResult
from src.core.pipeline import StockAnalysisPipeline
from src.enums import ReportType
from src.search_service import SearchService


def _result(code: str, name: str = "测试股票") -> AnalysisResult:
    return AnalysisResult(
        code=code,
        name=name,
        sentiment_score=60,
        trend_prediction="震荡",
        operation_advice="观望",
        analysis_summary="测试",
        success=True,
    )


def _pipeline() -> StockAnalysisPipeline:
    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    pipeline.config = SimpleNamespace(
        push_report_self_heal_enabled=True,
        push_report_self_heal_max_attempts=2,
        push_report_self_heal_delay_seconds=0,
    )
    pipeline.fetcher_manager = MagicMock()
    pipeline.fetcher_manager.invalidate_stock_caches.return_value = {
        "fundamental": 1,
        "stock_name": 1,
    }
    pipeline.fetcher_manager.get_daily_data.return_value = (
        pd.DataFrame({"date": ["2026-07-31"], "close": [100.0]}),
        "FallbackFetcher",
    )
    pipeline.db = MagicMock()
    pipeline.db.save_daily_data.return_value = 1
    pipeline.search_service = MagicMock()
    pipeline.search_service.invalidate_stock_search_cache.return_value = 2
    pipeline.notifier = MagicMock()
    pipeline._emit_progress = MagicMock()
    return pipeline


@patch("src.core.pipeline.IntelligenceService")
def test_self_heal_refetches_targeted_inputs_and_returns_repaired_result(mock_intel):
    pipeline = _pipeline()
    initial = _result("SNDK", "闪迪")
    repaired = _result("SNDK", "闪迪")
    issues = [
        "当日行情 OHLC/前收盘/成交量不完整",
        "最新报告期及核心财务指标不完整",
        "近期新闻/公告缺少日期、来源和可访问链接",
    ]
    pipeline.notifier.evaluate_single_stock_push_readiness.side_effect = [
        (False, issues),
        (True, []),
    ]
    pipeline.analyze_stock = MagicMock(return_value=repaired)
    mock_intel.return_value.refresh_auto_sources.return_value = {"status": "ok"}

    actual = pipeline._self_heal_incomplete_report(
        result=initial,
        code="SNDK",
        report_type=ReportType.SIMPLE,
        query_id="query-heal",
        current_time=None,
    )

    assert actual is repaired
    assert actual.report_self_heal["success"] is True
    assert actual.report_self_heal["attempts"] == 1
    pipeline.fetcher_manager.invalidate_stock_caches.assert_called_once_with("SNDK")
    pipeline.fetcher_manager.get_daily_data.assert_called_once_with("SNDK", days=120)
    pipeline.db.save_daily_data.assert_called_once()
    pipeline.search_service.invalidate_stock_search_cache.assert_called_once_with(
        "SNDK",
        "闪迪",
    )
    mock_intel.return_value.refresh_auto_sources.assert_called_once_with(force=True)
    pipeline.analyze_stock.assert_called_once_with(
        "SNDK",
        ReportType.SIMPLE,
        query_id="query-heal",
    )
    pipeline.notifier.send.assert_not_called()


def test_self_heal_is_bounded_and_preserves_unresolved_evidence_gaps():
    pipeline = _pipeline()
    initial = _result("600519", "贵州茅台")
    retry_one = _result("600519", "贵州茅台")
    retry_two = _result("600519", "贵州茅台")
    issue = "最新报告期及核心财务指标不完整"
    pipeline.notifier.evaluate_single_stock_push_readiness.side_effect = [
        (False, [issue]),
        (False, [issue]),
        (False, [issue]),
    ]
    pipeline.analyze_stock = MagicMock(side_effect=[retry_one, retry_two])

    actual = pipeline._self_heal_incomplete_report(
        result=initial,
        code="600519",
        report_type=ReportType.SIMPLE,
        query_id="query-bounded",
        current_time=None,
    )

    assert actual is retry_two
    assert actual.report_self_heal["success"] is False
    assert actual.report_self_heal["attempts"] == 2
    assert actual.report_self_heal["remaining_issues"] == [issue]
    assert pipeline.analyze_stock.call_count == 2
    assert pipeline.fetcher_manager.invalidate_stock_caches.call_count == 2
    pipeline.notifier.send.assert_not_called()


def test_data_manager_invalidation_is_stock_scoped():
    manager = DataFetcherManager.__new__(DataFetcherManager)
    manager._fundamental_cache = {
        "SNDK|budget=100": {"context": {}},
        "AAPL|budget=100": {"context": {}},
    }
    manager._stock_name_cache = {"SNDK": "闪迪", "AAPL": "苹果"}

    removed = manager.invalidate_stock_caches("SNDK")

    assert removed == {"fundamental": 1, "stock_name": 1}
    assert set(manager._fundamental_cache) == {"AAPL|budget=100"}
    assert manager._stock_name_cache == {"AAPL": "苹果"}


def test_search_cache_invalidation_is_stock_scoped():
    service = SearchService.__new__(SearchService)
    service._cache = {
        "SNDK earnings|5|30": (0.0, object()),
        "闪迪 公告|5|30": (0.0, object()),
        "AAPL earnings|5|30": (0.0, object()),
    }
    import threading

    service._cache_lock = threading.RLock()

    removed = service.invalidate_stock_search_cache("SNDK", "闪迪")

    assert removed == 2
    assert set(service._cache) == {"AAPL earnings|5|30"}
