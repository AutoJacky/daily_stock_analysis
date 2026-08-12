#!/usr/bin/env python3
"""Create and optionally push one free-Qwen/Codex fused market report."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse a strict market report with a free ModelScope Qwen review"
    )
    parser.add_argument("--market", choices=("cn", "us"), required=True)
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--market-report", type=Path)
    parser.add_argument("--stock-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    return parser.parse_args()


def _latest(directory: Path, pattern: str) -> Path | None:
    candidates = sorted(directory.glob(pattern), key=lambda item: item.stat().st_mtime)
    return candidates[-1] if candidates else None


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
    market_path = args.market_report or _latest(args.reports_dir, "market_review_*.md")
    if market_path is None or not market_path.is_file():
        print("错误：未找到严格市场复盘文件；融合已停止。", file=sys.stderr)
        return 2
    stock_path = args.stock_report or _latest(args.reports_dir, "report_*.md")
    try:
        market_report = _market_section(
            market_path.read_text(encoding="utf-8"), args.market
        )
    except ValueError as exc:
        print(f"错误：{exc}；融合已停止。", file=sys.stderr)
        return 2
    sources = FusionSources(
        market_report=market_report,
        stock_report=(
            stock_path.read_text(encoding="utf-8")
            if stock_path is not None and stock_path.is_file()
            else ""
        ),
    )
    try:
        review = call_free_qwen_review(
            args.market,
            sources,
            timeout_seconds=args.timeout,
        )
    except QwenFreeFusionError as exc:
        print(str(exc), file=sys.stderr)
        return 3

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
            title=f"🤝 {market_name}双AI融合复盘 · {datetime.now():%m-%d}",
            timeout_seconds=20,
        )
        if not sent:
            print("融合报告已生成，但 PushPlus 微信推送失败。", file=sys.stderr)
            return 4
        print("融合报告已由 PushPlus 受理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
