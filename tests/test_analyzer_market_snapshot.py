from src.analyzer import GeminiAnalyzer


def test_market_snapshot_recomputes_change_pct_from_displayed_prices() -> None:
    analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
    snapshot = analyzer._build_market_snapshot(
        {
            "date": "2026-08-10",
            "today": {
                "close": 1252.0,
                "open": 1240.0,
                "high": 1260.0,
                "low": 1230.0,
                "volume": 4946000,
                "pct_chg": 0.48154,
            },
            "yesterday": {"close": 1087.0},
        }
    )

    assert snapshot["close"] == "1252.00"
    assert snapshot["prev_close"] == "1087.00"
    assert snapshot["pct_chg"] == "15.18%"
    assert snapshot["change_amount"] == "165.00"
