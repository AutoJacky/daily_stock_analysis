#!/usr/bin/env python3
"""Upload newly written Qianwen scheduled reports without exposing content."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path.home() / "Documents" / "qwen-finance-reports",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path.home()
        / "Library"
        / "Application Support"
        / "daily_stock_analysis"
        / "qwen_report_bridge_state.json",
    )
    parser.add_argument("--repository", default="AutoJacky/daily_stock_analysis")
    parser.add_argument("--gh", default="gh")
    return parser.parse_args()


def _load_state(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _same_day_file(path: Path, today: date) -> bool:
    if not path.is_file():
        return False
    return datetime.fromtimestamp(path.stat().st_mtime).date() == today


def main() -> int:
    args = parse_args()
    state = _load_state(args.state_file)
    changed = False
    publisher = Path(__file__).with_name("publish_qwen_native_report.py")
    for market in ("cn", "us"):
        report = args.reports_root / market / "latest.md"
        local_today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        if not _same_day_file(report, local_today):
            continue
        content = report.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if state.get(market) == digest:
            continue
        result = subprocess.run(
            [
                sys.executable,
                str(publisher),
                "--market",
                market,
                "--report",
                str(report),
                "--repository",
                args.repository,
                "--gh",
                args.gh,
            ],
            check=False,
        )
        if result.returncode != 0:
            return result.returncode
        state[market] = digest
        changed = True
    if changed:
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        args.state_file.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
