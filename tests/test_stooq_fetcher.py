from unittest.mock import patch

import pandas as pd

from data_provider.stooq_fetcher import StooqFetcher
from data_provider.realtime_types import RealtimeSource


def test_stooq_daily_csv_normalizes_required_evidence():
    payload = (
        "Date,Open,High,Low,Close,Volume\n"
        "2026-08-03,100,104,99,103,1000\n"
        "2026-08-04,103,106,102,105,1200\n"
    )
    with patch.object(StooqFetcher, "_download_csv", return_value=payload):
        df = StooqFetcher().get_daily_data(
            "AAPL", start_date="2026-08-01", end_date="2026-08-05"
        )

    assert list(df["close"]) == [103, 105]
    assert float(df.iloc[-1]["amount"]) == 126000.0
    assert all(column in df.columns for column in ("ma5", "ma10", "ma20"))


def test_stooq_quote_uses_last_two_daily_bars():
    bars = pd.DataFrame(
        [
            {"date": "2026-08-03", "open": 99, "high": 102, "low": 98, "close": 100, "volume": 900, "amount": 90000},
            {"date": "2026-08-04", "open": 101, "high": 106, "low": 100, "close": 105, "volume": 1200, "amount": 126000},
        ]
    )
    fetcher = StooqFetcher()
    with patch.object(fetcher, "get_daily_data", return_value=bars):
        quote = fetcher.get_realtime_quote("AAPL")

    assert quote is not None
    assert quote.source is RealtimeSource.STOOQ
    assert quote.price == 105
    assert quote.pre_close == 100
    assert quote.change_pct == 5.0
