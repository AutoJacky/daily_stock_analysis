from src.analyzer import GeminiAnalyzer


def test_market_snapshot_recomputes_change_pct_from_displayed_prices() -> None:
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    snapshot = analyzer._build_market_snapshot(
        {
            "date": "2026-08-10",
            "code": "HK02513",
            "today": {
                "close": 1252.0,
                "open": 1240.0,
                "high": 1260.0,
                "low": 1230.0,
                "volume": 4946000,
                "amount": 6216000000,
                "pct_chg": 0.48154,
            },
            "yesterday": {"close": 1087.0},
        }
    )

    assert snapshot["close"] == "1252.00"
    assert snapshot["prev_close"] == "1087.00"
    assert snapshot["pct_chg"] == "15.18%"
    assert snapshot["change_amount"] == "165.00"
    assert snapshot["currency"] == "HKD"
    assert snapshot["amount"] == "62.16 亿港元"


def test_market_snapshot_marks_intraday_realtime_bar_as_partial() -> None:
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    snapshot = analyzer._build_market_snapshot(
        {
            "date": "2026-08-10",
            "code": "AMD",
            "today": {
                "date": "2026-08-10",
                "close": 478.52,
                "open": 477.42,
                "high": 483.36,
                "low": 470.69,
                "volume": 7427600,
                "data_source": "realtime:yfinance",
                "is_partial_bar": True,
            },
            "yesterday": {"close": 489.28},
            "market_phase_context": {
                "phase": "intraday",
                "is_partial_bar": True,
            },
        }
    )

    assert snapshot["quote_section_title"] == "最新行情"
    assert snapshot["close_label"] == "盘中估算价"
    assert snapshot["is_partial_bar"] is True
