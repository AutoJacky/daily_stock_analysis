#!/usr/bin/env python3
"""Publish one local Qianwen scheduled report as a GitHub encrypted secret."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("cn", "us"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository", default="AutoJacky/daily_stock_analysis")
    parser.add_argument("--gh", default="gh")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def market_report_date(market: str) -> date:
    timezone = ZoneInfo("Asia/Shanghai" if market == "cn" else "America/New_York")
    return __import__("datetime").datetime.now(timezone).date()


def build_payload(market: str, report: str, report_date: date) -> str:
    content = report.strip()
    if not content:
        raise ValueError("千问报告为空")
    expected = "A股" if market == "cn" else "美股"
    if expected not in content:
        raise ValueError(f"报告中未识别到{expected}市场标记")
    body = {
        "schema_version": 1,
        "market": market,
        "report_date": report_date.isoformat(),
        "content": content,
    }
    serialized = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(serialized) > 35_000:
        raise ValueError("千问报告UTF-8正文过长，无法安全写入GitHub Encrypted Secret")
    return base64.b64encode(serialized).decode("ascii")


def main() -> int:
    args = parse_args()
    try:
        content = args.report.read_text(encoding="utf-8")
        report_date = market_report_date(args.market)
        payload = build_payload(args.market, content, report_date)
    except (OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    secret_name = f"QWEN_NATIVE_REPORT_{args.market.upper()}_B64"
    if args.dry_run:
        print(f"{secret_name}: payload ready ({len(payload)} base64 chars)")
        return 0
    result = subprocess.run(
        [
            args.gh,
            "secret",
            "set",
            secret_name,
            "-R",
            args.repository,
        ],
        input=payload,
        text=True,
        check=False,
        capture_output=True,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        print("GitHub加密报告上传失败；未输出报告正文。", file=sys.stderr)
        return result.returncode
    print(f"{secret_name}: encrypted report updated for {report_date.isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
