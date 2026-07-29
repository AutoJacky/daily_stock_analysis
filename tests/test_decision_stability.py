# -*- coding: utf-8 -*-
"""Tests for structure-aware decision stability calibration."""

from types import SimpleNamespace

from src.analyzer import (
    AnalysisResult,
    _capital_flow_bias,
    enforce_evidence_consistency,
    fill_price_position_if_needed,
    stabilize_decision_with_structure,
)


def _result(
    *,
    decision_type: str,
    operation_advice: str,
    score: int,
    current_price: float,
    change_pct: float = 0.0,
) -> AnalysisResult:
    return AnalysisResult(
        code="002812",
        name="恩捷股份",
        sentiment_score=score,
        trend_prediction="看多" if decision_type == "buy" else "看空",
        operation_advice=operation_advice,
        decision_type=decision_type,
        report_language="zh",
        current_price=current_price,
        change_pct=change_pct,
        dashboard={
            "core_conclusion": {"one_sentence": "原始结论"},
            "data_perspective": {
                "price_position": {
                    "current_price": current_price,
                    "support_level": 30.0,
                    "resistance_level": 34.0,
                }
            },
        },
    )


def _fund_flow(main: float, five_day: float = 0.0, ten_day: float = 0.0) -> dict:
    return {
        "capital_flow": {
            "status": "ok",
            "data": {
                "stock_flow": {
                    "main_net_inflow": main,
                    "inflow_5d": five_day,
                    "inflow_10d": ten_day,
                }
            },
        }
    }


def _unsupported_fund_flow() -> dict:
    return {"capital_flow": {"status": "not_supported", "data": {}}}


def _unsupported_fund_flow_caps() -> dict:
    return {"capital_flow": {"status": "NOT_SUPPORTED", "data": {"stock_flow": {"main_net_inflow": 0}}}}


def test_capital_flow_bias_is_unavailable_when_stock_flow_data_is_missing() -> None:
    assert _capital_flow_bias(_unsupported_fund_flow()) == "unavailable"
    assert _capital_flow_bias({"capital_flow": {"status": "ok", "data": {}}}) == "unavailable"


def test_capital_flow_bias_is_neutral_when_missing_main_windows_conflict() -> None:
    context = {
        "capital_flow": {
            "data": {
                "stock_flow": {
                    "inflow_5d": 2_000_000,
                    "inflow_10d": -1_000_000,
                }
            }
        }
    }

    assert _capital_flow_bias(context) == "neutral"


def test_capital_flow_bias_is_neutral_when_main_conflicts_with_windows() -> None:
    context = _fund_flow(main=-500_000, five_day=1_200_000, ten_day=2_000_000)

    assert _capital_flow_bias(context) == "neutral"


def test_downgrades_buy_near_resistance_without_fund_confirmation() -> None:
    result = _result(
        decision_type="buy",
        operation_advice="买入",
        score=65,
        current_price=33.4,
    )

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=-1_000_000, five_day=-2_000_000),
    )

    assert result.decision_type == "hold"
    assert result.sentiment_score <= 59
    assert result.operation_advice == "震荡观望"
    assert result.dashboard["decision_stability"]["applied"] is True
    assert "不宜仅因短线反弹追买" in result.risk_warning
    assert result.dashboard["core_conclusion"]["signal_type"] == "🟡持有观望"


def test_downgrades_buy_mid_range_with_neutral_fund_flow() -> None:
    result = _result(
        decision_type="buy",
        operation_advice="买入",
        score=66,
        current_price=32.0,
    )

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=0, five_day=0, ten_day=0),
    )

    assert result.decision_type == "hold"
    assert result.sentiment_score <= 59
    assert result.operation_advice == "震荡观望"
    assert "资金流不明确" in result.risk_warning


def test_downgrades_buy_when_capital_flow_is_unavailable() -> None:
    buy_result = _result(
        decision_type="buy",
        operation_advice="买入",
        score=66,
        current_price=32.0,
    )
    sell_result = _result(
        decision_type="sell",
        operation_advice="卖出",
        score=30,
        current_price=30.4,
        change_pct=-2.1,
    )

    stabilize_decision_with_structure(
        buy_result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _unsupported_fund_flow(),
    )
    stabilize_decision_with_structure(
        sell_result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _unsupported_fund_flow(),
    )

    assert buy_result.decision_type == "hold"
    assert buy_result.operation_advice == "持有观察"
    assert buy_result.confidence_level == "低"
    assert buy_result.sentiment_score <= 59
    assert buy_result.dashboard["decision_stability"]["applied"] is True
    assert "买入结论缺少资金面确认" in buy_result.dashboard["decision_stability"]["reason"]
    assert buy_result.dashboard["core_conclusion"]["signal_type"] == "🟡持有观望"
    assert sell_result.decision_type == "sell"
    assert sell_result.operation_advice == "卖出"
    assert sell_result.dashboard["decision_stability"]["applied"] is False
    assert "未使用资金流校准" in sell_result.dashboard["decision_stability"]["reason"]


def test_missing_evidence_rewrites_all_action_fields_and_hallucination_prone_text() -> None:
    result = _result(
        decision_type="buy",
        operation_advice="买入",
        score=80,
        current_price=1321.0,
    )
    result.trend_prediction = "强烈看多"
    result.confidence_level = "中"
    result.risk_warning = "汇率波动影响出口业务"
    result.buy_reason = "PE处于近5年20%分位"
    result.dashboard.update(
        {
            "core_conclusion": {
                "one_sentence": "立即建仓",
                "time_sensitivity": "立即行动",
                "position_advice": {},
            },
            "data_perspective": {
                "price_position": {
                    "current_price": 1321.0,
                    "ma5": 1309.78,
                    "ma10": 1303.44,
                    "ma20": 1257.35,
                    "support_level": 1309.78,
                    "resistance_level": "1325（模型猜测）",
                }
            },
            "intelligence": {
                "sentiment_summary": "无近期重大事件",
                "risk_alerts": [],
                "positive_catalysts": ["利好"],
            },
            "battle_plan": {
                "sniper_points": {
                    "ideal_buy": "1310-1315",
                    "secondary_buy": "1303-1308",
                    "stop_loss": "1295",
                    "take_profit": "1350（MA10上轨）",
                },
                "action_checklist": [
                    "✅ PE 19.96处于合理区间（近5年百分位20%）",
                    "✅ 无重大利空公告",
                ],
            },
            "phase_decision": {
                "immediate_action": "立即建仓",
                "watch_conditions": ["竞价量能>600万手"],
                "data_limitations": [],
            },
        }
    )
    trend = SimpleNamespace(
        current_price=1321.0,
        ma5=1309.78,
        ma10=1303.44,
        ma20=1257.35,
        bias_ma5=0.86,
        volume_ratio_5d=1.17,
        support_levels=[1309.78, 1303.44, 1257.35],
        resistance_levels=[1343.48],
    )
    missing_fundamentals = {
        "growth": {"status": "failed", "data": {}},
        "earnings": {"status": "failed", "data": {}},
        "capital_flow": {"status": "failed", "data": {}},
    }

    fill_price_position_if_needed(result, trend)
    stabilize_decision_with_structure(result, trend, missing_fundamentals)
    enforce_evidence_consistency(
        result,
        trend,
        missing_fundamentals,
        news_result_count=0,
    )

    assert result.decision_type == "hold"
    assert result.trend_prediction == "震荡偏多（等待确认）"
    assert result.dashboard["trend_prediction"] == result.trend_prediction
    assert result.dashboard["core_conclusion"]["time_sensitivity"].startswith("不急")
    assert result.dashboard["phase_decision"]["immediate_action"].startswith("等待")
    assert "600万手" not in " ".join(result.dashboard["phase_decision"]["watch_conditions"])
    assert result.dashboard["data_perspective"]["price_position"]["resistance_level"] == 1343.48
    sniper = result.dashboard["battle_plan"]["sniper_points"]
    assert "1309.78" in sniper["ideal_buy"]
    assert "1303.44" in sniper["stop_loss"]
    assert "1343.48" in sniper["take_profit"]
    checklist = " ".join(result.dashboard["battle_plan"]["action_checklist"])
    assert "近5年百分位" not in checklist
    assert "无重大利空" in checklist
    assert "不能表述" in checklist
    assert "汇率波动" not in result.risk_warning
    assert result.dashboard["intelligence"]["positive_catalysts"] == []
    assert "无法排除近期事件风险" in result.dashboard["intelligence"]["latest_news"]


def test_verified_event_fields_are_rebuilt_from_dated_source_link_facts() -> None:
    result = _result(
        decision_type="hold",
        operation_advice="持有观察",
        score=55,
        current_price=32.0,
    )
    result.dashboard["intelligence"] = {
        "latest_news": "模型编造消息",
        "risk_alerts": ["无来源风险"],
        "positive_catalysts": ["无来源利好"],
    }
    fundamentals = {
        "growth": {"status": "ok", "data": {"revenue_yoy": 8.2}},
        "earnings": {"status": "ok", "data": {"report_period": "2026Q2"}},
        "capital_flow": {"status": "ok", "data": {"stock_flow": {"main_net_inflow": 1}}},
    }

    enforce_evidence_consistency(
        result,
        SimpleNamespace(ma5=31.8, ma10=31.0, ma20=30.0, bias_ma5=0.6),
        fundamentals,
        news_result_count=2,
        verified_event_evidence=[
            {
                "title": "公司发布回购公告",
                "published_date": "2026-07-30",
                "source": "上交所",
                "url": "https://www.sse.com.cn/example",
            },
            {
                "title": "股东披露减持计划",
                "published_date": "2026-07-29",
                "source": "巨潮资讯",
                "url": "https://www.cninfo.com.cn/example",
            },
        ],
    )

    intel = result.dashboard["intelligence"]
    assert "模型编造消息" not in intel["latest_news"]
    assert "[2026-07-30][上交所] 公司发布回购公告" in intel["latest_news"]
    assert intel["positive_catalysts"] == [
        "[2026-07-30][上交所] 公司发布回购公告（https://www.sse.com.cn/example）"
    ]
    assert intel["risk_alerts"][0] == (
        "[2026-07-29][巨潮资讯] 股东披露减持计划（https://www.cninfo.com.cn/example）"
    )
    assert "不等同交易所/公司公告全量核验" in intel["risk_alerts"][1]


def test_downgrades_buy_when_capital_flow_values_are_na() -> None:
    result = _result(
        decision_type="buy",
        operation_advice="买入",
        score=66,
        current_price=33.0,
    )

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        {
            "capital_flow": {
                "status": "ok",
                "data": {
                    "stock_flow": {
                        "main_net_inflow": "N/A",
                        "inflow_5d": "N/A",
                        "inflow_10d": "N/A",
                    }
                },
            }
        },
    )

    assert result.decision_type == "hold"
    assert result.operation_advice == "持有观察"
    assert result.dashboard["decision_stability"]["applied"] is True
    assert "资金流数据缺失" in result.dashboard["decision_stability"]["capital_flow_status"]


def test_downgrades_buy_advice_when_decision_type_is_hold_and_capital_flow_unavailable() -> None:
    result = _result(
        decision_type="hold",
        operation_advice="建议买入",
        score=68,
        current_price=32.0,
    )

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _unsupported_fund_flow(),
    )

    assert result.decision_type == "hold"
    assert result.operation_advice == "持有观察"
    assert result.sentiment_score <= 59
    assert result.dashboard["decision_stability"]["applied"] is True
    assert "买入结论缺少资金面确认" in result.dashboard["decision_stability"]["reason"]


def test_downgrades_buy_when_capital_flow_status_is_unavailable_case_insensitive() -> None:
    buy_result = _result(
        decision_type="buy",
        operation_advice="买入",
        score=66,
        current_price=32.0,
    )

    stabilize_decision_with_structure(
        buy_result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _unsupported_fund_flow_caps(),
    )

    assert buy_result.decision_type == "hold"
    assert buy_result.operation_advice == "持有观察"
    assert buy_result.dashboard["decision_stability"]["applied"] is True
    assert "暂不支持" in str(buy_result.dashboard["decision_stability"]["capital_flow_status"])


def test_skips_downgrade_when_only_generic_risk_warning_and_sell_near_support() -> None:
    result = _result(
        decision_type="sell",
        operation_advice="卖出",
        score=30,
        current_price=30.4,
        change_pct=1.0,
    )
    result.risk_warning = "注意常见回撤风险，建议关注仓位。"

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=500_000, five_day=300_000),
    )

    assert result.decision_type == "hold"
    assert result.operation_advice == "洗盘观察"
    assert "价格贴近支撑且未见资金持续流出" in result.risk_warning


def test_stability_can_infer_decision_from_natural_chinese_phrases_in_analyzer_path() -> None:
    result = _result(
        decision_type="建议卖出",
        operation_advice="建议卖出",
        score=30,
        current_price=30.4,
        change_pct=1.0,
    )

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=500_000, five_day=300_000),
    )

    assert result.decision_type == "hold"
    assert result.operation_advice == "洗盘观察"
    assert result.dashboard["decision_stability"]["applied"] is True


def test_downgrades_sell_near_support_without_sustained_outflow() -> None:
    result = _result(
        decision_type="sell",
        operation_advice="卖出",
        score=30,
        current_price=30.4,
        change_pct=-2.1,
    )

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=800_000, five_day=1_200_000),
    )

    assert result.decision_type == "hold"
    assert result.sentiment_score >= 45
    assert result.operation_advice == "洗盘观察"
    assert "不宜仅因单日下跌直接卖出" in result.risk_warning


def test_preserves_sell_signal_when_significant_risk_exists_near_support() -> None:
    result = _result(
        decision_type="sell",
        operation_advice="卖出",
        score=30,
        current_price=30.4,
        change_pct=-2.1,
    )
    result.risk_warning = "重大利空消息：公司发布重大减持计划"
    result.dashboard["intelligence"] = {"risk_alerts": ["股东高位减持预告"]}

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=800_000, five_day=1_200_000),
    )

    assert result.decision_type == "sell"
    assert result.operation_advice == "卖出"


def test_refines_hold_pullback_near_support_as_shakeout_watch() -> None:
    result = _result(
        decision_type="hold",
        operation_advice="持有",
        score=52,
        current_price=30.5,
        change_pct=-1.6,
    )

    stabilize_decision_with_structure(
        result,
        SimpleNamespace(support_levels=[30.0], resistance_levels=[34.0]),
        _fund_flow(main=0, five_day=500_000),
    )

    assert result.decision_type == "hold"
    assert result.operation_advice == "洗盘观察"
    assert "更适合按洗盘观察处理" in result.risk_warning
