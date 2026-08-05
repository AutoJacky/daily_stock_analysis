# -*- coding: utf-8 -*-
"""Independent, keyless Stooq fallback for US end-of-day market data.

This fetcher intentionally does not depend on yfinance.  Keeping it as a
separate manager-level source means a Yahoo circuit breaker or regional block
cannot prevent the pipeline from trying Stooq.
"""

import logging
from datetime import datetime, timezone
from io import StringIO
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .base import BaseFetcher, DataFetchError
from .realtime_types import RealtimeSource, UnifiedRealtimeQuote
from .us_index_mapping import is_us_index_code, is_us_stock_code


logger = logging.getLogger(__name__)


class StooqFetcher(BaseFetcher):
    """US daily bars and delayed quote fallback backed by Stooq CSV feeds."""

    name = "StooqFetcher"
    priority = 5
    _USER_AGENT = (
        "Mozilla/5.0 (compatible; DSA/1.0; "
        "+https://github.com/ZhuLinsen/daily_stock_analysis)"
    )

    @staticmethod
    def _symbol(stock_code: str) -> str:
        code = str(stock_code or "").strip().upper()
        if is_us_index_code(code):
            raise DataFetchError("Stooq fallback currently supports US equities, not index aliases")
        if not is_us_stock_code(code):
            raise DataFetchError(f"Stooq does not support non-US symbol {stock_code}")
        return f"{code.lower()}.us"

    @classmethod
    def _download_csv(cls, url: str) -> str:
        request = Request(
            url,
            headers={
                "User-Agent": cls._USER_AGENT,
                "Accept": "text/csv,text/plain,*/*",
            },
        )
        try:
            with urlopen(request, timeout=15) as response:
                payload = response.read().decode("utf-8", "ignore").strip()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise DataFetchError(f"Stooq request failed: {type(exc).__name__}") from exc
        if not payload or payload.upper().startswith("NO DATA") or "<html" in payload.lower():
            raise DataFetchError("Stooq returned no market data")
        return payload

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol = self._symbol(stock_code)
        params = urlencode(
            {
                "s": symbol,
                "i": "d",
                "d1": start_date.replace("-", ""),
                "d2": end_date.replace("-", ""),
            }
        )
        payload = self._download_csv(f"https://stooq.com/q/d/l/?{params}")
        try:
            df = pd.read_csv(StringIO(payload))
        except Exception as exc:
            raise DataFetchError("Stooq CSV parse failed") from exc
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        if df.empty or not required.issubset(set(df.columns)):
            raise DataFetchError("Stooq CSV missing required OHLCV columns")
        return df

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        normalized = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        ).copy()
        for column in ("open", "high", "low", "close", "volume"):
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized["amount"] = normalized["close"] * normalized["volume"]
        normalized["pct_chg"] = normalized["close"].pct_change() * 100.0
        normalized = normalized[
            ["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]
        ]
        return normalized

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """Build a transparent delayed quote from the last two daily bars."""
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        df = self.get_daily_data(stock_code, end_date=end_date, days=10)
        if df is None or df.empty:
            return None
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else None
        close = float(latest["close"])
        pre_close = float(previous["close"]) if previous is not None else None
        change_amount = close - pre_close if pre_close else None
        change_pct = change_amount / pre_close * 100.0 if pre_close else None
        provider_ts = pd.Timestamp(latest["date"]).date().isoformat()
        return UnifiedRealtimeQuote(
            code=str(stock_code).strip().upper(),
            source=RealtimeSource.STOOQ,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            provider_timestamp=provider_ts,
            market="us",
            currency="USD",
            data_quality="partial",
            price=close,
            change_pct=change_pct,
            change_amount=change_amount,
            volume=int(float(latest["volume"])),
            amount=float(latest["amount"]),
            open_price=float(latest["open"]),
            high=float(latest["high"]),
            low=float(latest["low"]),
            pre_close=pre_close,
        )
