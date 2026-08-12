#!/usr/bin/env python3
"""Create and optionally push one free-Qwen/Codex fused market report."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config  # noqa: E402
from src.notification_sender.pushplus_sender import PushplusSender  # noqa: E402
from src.services.qwen_free_report_fusion import (  # noqa: E402
    FusionSources,
    QwenFreeFusionError,
    call_free_qwen_review,
    render_fused_report,
)
from src.services.institutional_market_context import (  # noqa: E402
    InstitutionalMarketContextCollector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse a strict market report with a free ModelScope Qwen review"
    )
    parser.add_argument("--market", choices=("cn", "us"), required=True)
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--market-report", type=Path)
    parser.add_argument("--stock-report", type=Path)
    parser.add_argument("--native-qwen-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    return parser.parse_args()


def _native_report_from_environment(market: str, today: date) -> str:
    """Read a same-day report carried in an encrypted GitHub secret."""

    encoded = os.getenv(f"QWEN_NATIVE_REPORT_{market.upper()}_B64", "").strip()
    if not encoded:
        return ""
    try:
        payload = json.loads(base64.b64decode(encoded, validate=True).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("千问原生报告密文载荷无法解析") from exc
    if not isinstance(payload, dict):
        raise ValueError("千问原生报告载荷不是对象")
    if payload.get("market") != market:
        raise ValueError("千问原生报告市场不匹配")
    if payload.get("report_date") != today.isoformat():
        raise ValueError("千问原生报告不是当天内容，已拒绝融合")
    report = str(payload.get("content") or "").strip()
    if not report:
        raise ValueError("千问原生报告正文为空")
    return report


def _market_report_date(market: str, now: datetime | None = None) -> date:
    """Canonical session date shared by local bridge and GitHub runner."""

    now = now or datetime.now().astimezone()
    timezone = ZoneInfo("Asia/Shanghai" if market == "cn" else "America/New_York")
    return now.astimezone(timezone).date()


def _strict_report_session_date(report: str, market: str) -> date:
    """Extract the complete session validated by the strict market report."""

    marker = "美股" if market == "us" else "大盘复盘"
    match = re.search(
        rf"(?m)^##\s+(20\d{{2}}-\d{{2}}-\d{{2}})\s+[^\n]*{marker}[^\n]*严格数据版",
        report,
    )
    if match is None:
        raise ValueError("严格市场报告缺少可验证的交易日标题")
    return date.fromisoformat(match.group(1))


def _today_report(directory: Path, prefix: str) -> Path | None:
    """Select only a report generated for today's workflow run."""

    path = directory / f"{prefix}_{datetime.now().strftime('%Y%m%d')}.md"
    return path if path.is_file() else None


def _market_section(report: str, market: str) -> str:
    """Reject or isolate a combined report before sending it to Qwen."""

    expected = "A股" if market == "cn" else "美股"
    other = "美股" if market == "cn" else "A股"
    if expected not in report:
        # Legacy A-share reports used the generic title ``大盘复盘`` before
        # multi-market output existed.  Accept that shape only when the caller
        # explicitly selects ``cn`` and no US-market marker is present.  US
        # reports always require an explicit title to prevent cross-market
        # leakage.
        if market == "cn" and "大盘复盘" in report and "美股" not in report:
            return report.strip()
        raise ValueError(f"严格市场报告不包含{expected}标题")
    expected_match = re.search(rf"(?m)^#\s+{expected}.*$", report)
    if expected_match is None:
        return report
    tail = report[expected_match.start() :]
    other_match = re.search(rf"(?m)^#\s+{other}.*$", tail)
    return tail[: other_match.start()].strip() if other_match else tail.strip()


def main() -> int:
    args = parse_args()
    market_path = args.market_report or _today_report(args.reports_dir, "market_review")
    if market_path is None or not market_path.is_file():
        print("错误：未找到严格市场复盘文件；融合已停止。", file=sys.stderr)
        return 2
    stock_path = args.stock_report or _today_report(args.reports_dir, "report")
    try:
        market_report = _market_section(
            market_path.read_text(encoding="utf-8"), args.market
        )
    except ValueError as exc:
        print(f"错误：{exc}；融合已停止。", file=sys.stderr)
        return 2
    try:
        native_qwen_report = (
            args.native_qwen_report.read_text(encoding="utf-8")
            if args.native_qwen_report is not None
            else _native_report_from_environment(
                args.market, _market_report_date(args.market)
            )
        )
    except OSError as exc:
        print(f"错误：无法读取显式千问原生报告：{exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"警告：{exc}；本轮忽略该原生报告并继续免费复核。", file=sys.stderr)
        native_qwen_report = ""
    try:
        report_session_date = _strict_report_session_date(market_report, args.market)
    except ValueError as exc:
        print(f"错误：{exc}；融合已停止。", file=sys.stderr)
        return 2
    institutional_context = InstitutionalMarketContextCollector().collect(
        args.market, report_session_date
    )
    sources = FusionSources(
        market_report=market_report,
        stock_report=(
            stock_path.read_text(encoding="utf-8")
            if stock_path is not None and stock_path.is_file()
            else ""
        ),
        native_qwen_report=native_qwen_report,
        institutional_context=institutional_context,
    )
    try:
        review = call_free_qwen_review(
            args.market,
            sources,
            timeout_seconds=args.timeout,
        )
    except QwenFreeFusionError as exc:
        print(f"警告：{exc} 将继续生成程序校验版，不调用任何收费模型。", file=sys.stderr)
        review = {
            "_audit_status": "unavailable",
            "_audit_note": str(exc),
            "summary": "免费千问审计暂不可用；本报告仅保留程序校验事实和规则，不采用未经交叉审计的模型观点。",
            "consensus": [],
            "disagreements": [],
            "risk_actions": [],
            "opportunity_watch": [],
            "data_gaps": ["魔搭免费千问审计本轮未完成；未调用任何收费回退。"],
        }

    content = render_fused_report(args.market, sources, review)
    output = args.output or (
        args.reports_dir
        / f"fused_review_{args.market}_{datetime.now().strftime('%Y%m%d')}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"融合报告已保存：{output}")

    if args.send:
        market_name = "A股" if args.market == "cn" else "美股"
        sender = PushplusSender(get_config())
        sent = sender.send_to_pushplus(
            content,
            title=f"🏛️ {market_name}多源机构框架复盘 · {datetime.now():%m-%d}",
            timeout_seconds=20,
        )
        if not sent:
            print("融合报告已生成，但 PushPlus 微信推送失败。", file=sys.stderr)
            return 4
        print("融合报告已由 PushPlus 受理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
