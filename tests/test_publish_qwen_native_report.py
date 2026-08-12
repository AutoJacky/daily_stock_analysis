import base64
import json
from datetime import date

import pytest

from scripts.fuse_qwen_market_report import _market_report_date, _native_report_from_environment
from scripts import publish_qwen_native_report
from scripts.publish_qwen_native_report import build_payload


def test_native_report_payload_round_trip(monkeypatch):
    encoded = build_payload("cn", "# A股盘后复盘报告\n正文", date(2026, 8, 12))
    monkeypatch.setenv("QWEN_NATIVE_REPORT_CN_B64", encoded)
    assert "正文" in _native_report_from_environment("cn", date(2026, 8, 12))


def test_native_report_rejects_stale_secret(monkeypatch):
    encoded = build_payload("cn", "# A股盘后复盘报告\n正文", date(2026, 8, 11))
    monkeypatch.setenv("QWEN_NATIVE_REPORT_CN_B64", encoded)
    with pytest.raises(ValueError, match="不是当天"):
        _native_report_from_environment("cn", date(2026, 8, 12))


def test_native_report_rejects_wrong_market_marker():
    with pytest.raises(ValueError, match="美股"):
        build_payload("us", "# A股盘后复盘报告", date(2026, 8, 12))


def test_us_report_date_uses_new_york_session_date():
    from datetime import datetime, timezone

    assert _market_report_date(
        "us", datetime(2026, 8, 12, 21, 45, tzinfo=timezone.utc)
    ) == date(2026, 8, 12)


def test_publisher_passes_secret_only_through_stdin(monkeypatch, tmp_path):
    report = tmp_path / "latest.md"
    report.write_text("# A股盘后复盘报告\n正文", encoding="utf-8")
    captured = {}

    class Result:
        returncode = 0

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["input"] = kwargs["input"]
        return Result()

    monkeypatch.setattr(publish_qwen_native_report.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "publish_qwen_native_report.py",
            "--market",
            "cn",
            "--report",
            str(report),
            "--gh",
            "/tmp/gh",
        ],
    )

    assert publish_qwen_native_report.main() == 0
    assert captured["command"] == [
        "/tmp/gh",
        "secret",
        "set",
        "QWEN_NATIVE_REPORT_CN_B64",
        "-R",
        "AutoJacky/daily_stock_analysis",
    ]
    assert "正文" not in " ".join(captured["command"])
    assert captured["input"]
