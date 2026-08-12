from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.services.qwen_free_report_fusion import (
    MODELSCOPE_API_URL,
    MODELSCOPE_FREE_MODEL,
    FusionSources,
    QwenFreeFusionError,
    call_free_qwen_review,
    render_fused_report,
    stock_prompt_excerpt,
)
from scripts.fuse_qwen_market_report import _market_section
from scripts.fuse_qwen_market_report import _today_report


MARKET_REPORT = """# A股大盘复盘

## 2026-08-10 大盘复盘（严格数据版）

### 一、数据校验
- 市场宽度覆盖：5537 只证券

### 二、市场宽度与成交
- 上涨 / 下跌 / 平盘：4067 / 1391 / 79
- 上涨占比：74.5%

### 六、次日量化计划
- 规则姿态：偏进攻；模型组合仓位区间 40%-60%。
"""

STOCK_REPORT = """# 2026-08-10 决策仪表盘

## 分析结果摘要

🟡 **贵州茅台(600519)**: 持有 | 评分 59

---

## 贵州茅台 (600519)

### 核心结论
一句话决策：等待确认。
"""


def _response(payload: dict, status_code: int = 200) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.text = "response"
    response.json.return_value = payload
    return response


def test_call_uses_only_fixed_modelscope_free_endpoint_and_scrubs_new_numbers():
    qwen_json = {
        "summary": "上涨占比74.5%，未来7天保证上涨",
        "consensus": [{"point": "市场偏暖", "evidence": "上涨家数4067"}],
        "disagreements": [],
        "risk_actions": ["仓位不超过60%"],
        "opportunity_watch": [],
        "data_gaps": [],
    }
    response = _response(
        {"choices": [{"message": {"content": f"```json\n{__import__('json').dumps(qwen_json, ensure_ascii=False)}\n```"}}]}
    )
    sources = FusionSources(MARKET_REPORT, STOCK_REPORT)

    with patch("src.services.qwen_free_report_fusion.requests.post", return_value=response) as post:
        review = call_free_qwen_review("cn", sources, token="secret")

    assert review["summary"] == (
        "上涨占比74.5%，未来（千问新增数值已省略，具体数值见下方权威底稿）天保证上涨"
    )
    assert review["risk_actions"] == ["仓位不超过60%"]
    assert post.call_args.args[0] == MODELSCOPE_API_URL
    assert post.call_args.kwargs["json"]["model"] == MODELSCOPE_FREE_MODEL
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer secret"


def test_free_quota_error_stops_without_fallback():
    response = _response({}, status_code=429)
    with patch("src.services.qwen_free_report_fusion.requests.post", return_value=response) as post:
        with pytest.raises(QwenFreeFusionError, match="未切换到收费服务"):
            call_free_qwen_review(
                "cn", FusionSources(MARKET_REPORT, STOCK_REPORT), token="secret"
            )
    assert post.call_count == 1


def test_render_keeps_authoritative_source_facts_and_labels_qwen_role():
    report = render_fused_report(
        "cn",
        FusionSources(MARKET_REPORT, STOCK_REPORT),
        {
            "summary": "宽度偏强，但按条件执行。",
            "consensus": [{"point": "偏暖", "evidence": "上涨占比74.5%"}],
            "disagreements": [],
            "risk_actions": ["跌破条件后降低风险。"],
            "opportunity_watch": ["等待确认。"],
            "data_gaps": [],
        },
        generated_at=datetime(2026, 8, 10, 18, 0),
    )

    assert "A股双AI融合复盘 · 2026-08-10" in report
    assert "免费千问负责独立复核" in report
    assert "5537 只证券" in report
    assert "4067 / 1391 / 79" in report
    assert "贵州茅台(600519)" in report
    assert "不构成投资建议" in report


def test_stock_prompt_keeps_late_core_conclusion():
    filler = "\n".join(f"普通描述 {index}" for index in range(120))
    report = f"# 报告\n{filler}\n### 核心结论\n一句话决策：持有观察。"
    excerpt = stock_prompt_excerpt(report)
    assert "核心结论" in excerpt
    assert "持有观察" in excerpt


def test_stock_prompt_keeps_complete_report_when_within_budget():
    report = "# 报告\n## 股票一\n当日行情完整\n## 股票二\n成交量完整"
    assert stock_prompt_excerpt(report) == report


def test_market_section_rejects_wrong_market_and_isolates_combined_report():
    combined = f"{MARKET_REPORT}\n\n---\n\n# 美股大盘复盘\n美股内容"
    assert "美股内容" not in _market_section(combined, "cn")
    with pytest.raises(ValueError, match="美股"):
        _market_section(MARKET_REPORT, "us")


def test_market_section_accepts_legacy_generic_a_share_title_only_for_cn():
    legacy = "# 🎯 大盘复盘\n\n上涨 / 下跌：100 / 50"
    assert _market_section(legacy, "cn") == legacy
    with pytest.raises(ValueError, match="美股"):
        _market_section(legacy, "us")


def test_today_report_never_falls_back_to_stale_file(tmp_path, monkeypatch):
    stale = tmp_path / "report_20260729.md"
    stale.write_text("old", encoding="utf-8")
    assert _today_report(tmp_path, "report") is None


def test_source_report_date_is_not_kept_as_a_spurious_future_gap():
    qwen_json = {
        "summary": "按源报告执行。",
        "consensus": [],
        "disagreements": [],
        "risk_actions": [],
        "opportunity_watch": [],
        "data_gaps": [
            "未来日期 2026-08-10 无法实时验证",
            "缺少带来源标识的市场新闻",
        ],
    }
    response = _response(
        {"choices": [{"message": {"content": __import__('json').dumps(qwen_json, ensure_ascii=False)}}]}
    )
    with patch("src.services.qwen_free_report_fusion.requests.post", return_value=response):
        review = call_free_qwen_review(
            "cn", FusionSources(MARKET_REPORT, STOCK_REPORT), token="secret"
        )

    assert review["data_gaps"] == ["缺少带来源标识的市场新闻"]
