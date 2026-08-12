#!/usr/bin/env python3
"""Install a per-user LaunchAgent for the Qianwen report bridge."""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path


LABEL = "com.daily-stock-analysis.qwen-report-bridge"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default="AutoJacky/daily_stock_analysis")
    parser.add_argument("--gh")
    parser.add_argument("--interval", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sys.platform != "darwin":
        print("该安装器仅支持macOS。", file=sys.stderr)
        return 2
    project_root = Path(__file__).resolve().parents[1]
    bridge = project_root / "scripts" / "qwen_native_report_bridge.py"
    gh = args.gh or shutil.which("gh")
    if not gh:
        local_gh = project_root / ".tools" / "gh" / "gh_2.96.0_macOS_arm64" / "bin" / "gh"
        gh = str(local_gh) if local_gh.is_file() else ""
    if not gh:
        print("未找到gh命令，无法安装自动桥接。", file=sys.stderr)
        return 2
    reports_root = Path.home() / "Documents" / "qwen-finance-reports"
    (reports_root / "cn").mkdir(parents=True, exist_ok=True)
    (reports_root / "us").mkdir(parents=True, exist_ok=True)
    log_root = Path.home() / "Library" / "Logs" / "daily_stock_analysis"
    log_root.mkdir(parents=True, exist_ok=True)
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Label": LABEL,
        "ProgramArguments": [
            sys.executable,
            str(bridge),
            "--reports-root",
            str(reports_root),
            "--repository",
            args.repository,
            "--gh",
            str(gh),
        ],
        "RunAtLoad": True,
        "StartInterval": max(60, args.interval),
        "StandardOutPath": str(log_root / "qwen_report_bridge.log"),
        "StandardErrorPath": str(log_root / "qwen_report_bridge.error.log"),
        "ProcessType": "Background",
    }
    plist_path.write_bytes(plistlib.dumps(payload))
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", domain, str(plist_path)], check=False)
    result = subprocess.run(
        ["launchctl", "bootstrap", domain, str(plist_path)], check=False
    )
    if result.returncode != 0:
        print("LaunchAgent加载失败，请查看桥接错误日志。", file=sys.stderr)
        return result.returncode
    print(f"已安装：{plist_path}")
    print(f"千问报告目录：{reports_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
