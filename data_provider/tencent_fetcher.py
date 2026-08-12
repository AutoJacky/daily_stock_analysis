# -*- coding: utf-8 -*-
"""Tencent direct daily K-line fetcher for A-share fallback routing."""

from __future__ import annotations

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

try:
    import exchange_calendars as xcals
except ImportError:  # pragma: no cover - dependency is present in supported installs
    xcals = None

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS, normalize_stock_code, is_bse_code

logger = logging.getLogger(__name__)

_MAX_KLINE_BARS = 800
_SINA_MARKET_COUNT_ENDPOINT = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeStockCount"
)
_SINA_MARKET_PAGE_ENDPOINT = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData"
)
_SINA_MARKET_PAGE_SIZE = 100
_SINA_MARKET_MAX_WORKERS = 10


class TencentFetcher(BaseFetcher):
    """Fetch qfq daily K-line data from Tencent's direct quote endpoint."""

    name = "TencentFetcher"
    priority = 0
    allow_empty_daily_data = True

    _KLINE_ENDPOINT = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    _QUOTE_ENDPOINT = "https://qt.gtimg.cn/q"
    _HTTP_TIMEOUT_SECONDS = 8

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        code = normalize_stock_code(stock_code)
        symbol = _to_tencent_symbol(code)
        if not symbol:
            raise DataFetchError(f"TencentFetcher unsupported stock code: {stock_code}")

        lookback = _estimate_lookback_days(start_date=start_date, end_date=end_date)
        explicit_start = _format_tencent_date(start_date)
        explicit_end = _format_tencent_date(end_date)
        explicit_window = (
            f"{explicit_start},{explicit_end}"
            if explicit_start and explicit_end
            else ","
        )
        response = requests.get(
            self._KLINE_ENDPOINT,
            params={"param": f"{symbol},day,{explicit_window},{lookback},qfq"},
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*"},
            timeout=self._HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        rows = _extract_kline_rows(payload, symbol=symbol)
        if not rows:
            logger.info("TencentFetcher empty daily history for %s", stock_code)
            return _empty_daily_frame()

        df = pd.DataFrame(rows)
        first_returned_date = _first_returned_date(df)
        if first_returned_date and _is_capped_history_incomplete(
            first_returned_date=first_returned_date,
            start_date=start_date,
            lookback=lookback,
            returned_rows=len(rows),
        ):
            logger.info(
                "TencentFetcher incomplete capped daily history for %s: first_date=%s requested_start=%s",
                stock_code,
                first_returned_date,
                start_date,
            )
            return _empty_daily_frame()

        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if df.empty:
            logger.info(
                "TencentFetcher daily history outside requested range for %s: %s~%s",
                stock_code,
                start_date,
                end_date,
            )
            return _empty_daily_frame()
        return df

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        normalized = df.copy()
        for column in ("open", "high", "low", "close", "volume", "amount"):
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        if "pct_chg" not in normalized.columns:
            normalized["pct_chg"] = normalized["close"].pct_change().fillna(0.0) * 100
        normalized = normalized[["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]
        return normalized

    def get_main_indices(self, region: str = "cn") -> Optional[List[Dict[str, Any]]]:
        """Fetch a same-session six-index close snapshot in one request."""
        if region != "cn":
            return None

        try:
            from src.core.trading_calendar import (
                MarketPhase,
                get_effective_trading_date,
                infer_market_phase,
            )

            if infer_market_phase("cn") in {
                MarketPhase.INTRADAY,
                MarketPhase.LUNCH_BREAK,
                MarketPhase.CLOSING_AUCTION,
            }:
                return None
            cutoff = get_effective_trading_date("cn")
        except Exception:
            return None

        names = {
            "sh000001": "上证指数",
            "sz399001": "深证成指",
            "sz399006": "创业板指",
            "sh000688": "科创50",
            "sh000016": "上证50",
            "sh000300": "沪深300",
        }
        try:
            response = requests.get(
                self._QUOTE_ENDPOINT,
                params={"q": ",".join(names)},
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://gu.qq.com/",
                },
                timeout=self._HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            response.encoding = "gbk"
            payload = response.text
        except requests.RequestException as exc:
            logger.warning("TencentFetcher index quote request failed: %s", exc)
            return None

        rows: Dict[str, Dict[str, Any]] = {}
        for match in re.finditer(r'v_([a-z0-9]+)="([^"]*)"', payload, re.I):
            symbol = match.group(1).lower()
            if symbol not in names:
                continue
            fields = match.group(2).split("~")
            if len(fields) < 36:
                continue
            try:
                current = float(fields[3])
                prev_close = float(fields[4])
                open_price = float(fields[5])
                high = float(fields[33])
                low = float(fields[34])
                volume = float(fields[6])
                quote_date = datetime.strptime(fields[30][:8], "%Y%m%d").date()
                amount_parts = fields[35].split("/")
                amount = float(amount_parts[2]) if len(amount_parts) >= 3 else 0.0
            except (TypeError, ValueError, IndexError):
                continue
            if (
                quote_date != cutoff
                or min(current, prev_close, open_price, high, low, amount) <= 0
            ):
                continue
            rows[symbol] = {
                "code": symbol,
                "name": names[symbol],
                "current": current,
                "change": current - prev_close,
                "change_pct": (current - prev_close) / prev_close * 100.0,
                "open": open_price,
                "high": high,
                "low": low,
                "prev_close": prev_close,
                "volume": volume,
                "amount": amount,
                "amplitude": (high - low) / prev_close * 100.0,
                "trade_date": quote_date.isoformat(),
                "source": "Tencent Finance (close quote)",
            }

        if len(rows) != len(names):
            logger.warning(
                "TencentFetcher incomplete main-index quote: %s/%s",
                len(rows),
                len(names),
            )
            return None
        return [rows[symbol] for symbol in names]

    def get_market_stats(self) -> Optional[Dict[str, Any]]:
        """Fetch a complete A-share breadth snapshot from Sina in parallel.

        AKShare's Sina fallback requests the same paginated public endpoint
        serially.  With more than 5,000 listings that path can exceed the
        provider timeout on GitHub-hosted runners.  This implementation first
        reads the published row count, downloads every 100-row page with
        bounded concurrency, and rejects incomplete snapshots.

        Sina rows expose only a close clock (``ticktime``), not a calendar
        date.  Outside trading hours we therefore accept the snapshot only
        when Tencent's six-index quote independently proves the effective
        session date.  The anchored date is returned with the statistics so
        downstream validation does not have to infer it.
        """
        try:
            from src.core.trading_calendar import (
                MarketPhase,
                get_effective_trading_date,
                get_market_now,
                infer_market_phase,
            )

            phase = infer_market_phase("cn")
            if phase in {
                MarketPhase.INTRADAY,
                MarketPhase.LUNCH_BREAK,
                MarketPhase.CLOSING_AUCTION,
            }:
                trade_date = get_market_now("cn").date()
            else:
                trade_date = get_effective_trading_date("cn")
                index_rows = self.get_main_indices("cn")
                index_dates = {
                    str(row.get("trade_date") or "")
                    for row in (index_rows or [])
                    if row.get("trade_date")
                }
                if index_dates != {trade_date.isoformat()}:
                    logger.warning(
                        "[MarketStats] component=market_stats provider=TencentFetcher "
                        "api=sina_full_market action=reject reason=index_session_unverified "
                        "expected=%s actual=%s",
                        trade_date,
                        sorted(index_dates),
                    )
                    return None

            headers = {
                "Referer": "https://finance.sina.com.cn/",
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json,text/plain,*/*",
            }
            count_response = requests.get(
                _SINA_MARKET_COUNT_ENDPOINT,
                params={"node": "hs_a"},
                headers=headers,
                timeout=self._HTTP_TIMEOUT_SECONDS,
            )
            count_response.raise_for_status()
            expected_count = int(count_response.json())
            if expected_count < 4_000 or expected_count > 10_000:
                logger.warning(
                    "[MarketStats] component=market_stats provider=TencentFetcher "
                    "api=sina_full_market action=reject reason=implausible_count count=%s",
                    expected_count,
                )
                return None

            page_count = int(math.ceil(expected_count / _SINA_MARKET_PAGE_SIZE))

            def fetch_page(page: int) -> List[Dict[str, Any]]:
                response = requests.get(
                    _SINA_MARKET_PAGE_ENDPOINT,
                    params={
                        "page": page,
                        "num": _SINA_MARKET_PAGE_SIZE,
                        "sort": "symbol",
                        "asc": "1",
                        "node": "hs_a",
                        "symbol": "",
                        "_s_r_a": "page",
                    },
                    headers=headers,
                    timeout=self._HTTP_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                payload = response.json()
                return payload if isinstance(payload, list) else []

            started_at = datetime.now()
            pages: Dict[int, List[Dict[str, Any]]] = {}
            with ThreadPoolExecutor(max_workers=_SINA_MARKET_MAX_WORKERS) as executor:
                futures = {
                    executor.submit(fetch_page, page): page
                    for page in range(1, page_count + 1)
                }
                for future in as_completed(futures):
                    page = futures[future]
                    try:
                        pages[page] = future.result()
                    except Exception as exc:
                        logger.warning(
                            "[MarketStats] component=market_stats provider=TencentFetcher "
                            "api=sina_full_market action=page_failed page=%s error=%s",
                            page,
                            exc,
                        )

            rows = [row for page in sorted(pages) for row in pages[page]]
            symbols = {str(row.get("symbol") or "") for row in rows if row.get("symbol")}
            if len(rows) != expected_count or len(symbols) != expected_count:
                logger.warning(
                    "[MarketStats] component=market_stats provider=TencentFetcher "
                    "api=sina_full_market action=reject reason=incomplete_snapshot "
                    "expected=%s rows=%s unique=%s pages=%s/%s",
                    expected_count,
                    len(rows),
                    len(symbols),
                    len(pages),
                    page_count,
                )
                return None

            stats = _calculate_sina_market_stats(rows)
            if not stats:
                return None
            stats.update(
                {
                    "trade_date": trade_date.isoformat(),
                    "snapshot_count": expected_count,
                    "_source": "Sina Finance full A-share snapshot",
                }
            )
            logger.info(
                "[MarketStats] component=market_stats provider=TencentFetcher "
                "api=sina_full_market action=success trade_date=%s rows=%s "
                "up=%s down=%s flat=%s elapsed=%.2fs",
                trade_date,
                expected_count,
                stats["up_count"],
                stats["down_count"],
                stats["flat_count"],
                (datetime.now() - started_at).total_seconds(),
            )
            return stats
        except Exception as exc:
            logger.warning(
                "[MarketStats] component=market_stats provider=TencentFetcher "
                "api=sina_full_market action=failed error=%s",
                exc,
            )
            return None


def _calculate_sina_market_stats(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Calculate breadth only from valid, traded A-share quote rows."""
    up_count = down_count = flat_count = 0
    limit_up_count = limit_down_count = 0
    total_amount = 0.0

    for row in rows:
        try:
            code = normalize_stock_code(str(row.get("code") or row.get("symbol") or ""))
            name = str(row.get("name") or "")
            current = float(row.get("trade"))
            previous = float(row.get("settlement"))
            amount = float(row.get("amount") or 0.0)
        except (TypeError, ValueError):
            continue
        if not code or current <= 0 or previous <= 0 or amount <= 0:
            continue

        if current > previous:
            up_count += 1
        elif current < previous:
            down_count += 1
        else:
            flat_count += 1
        total_amount += amount

        if is_bse_code(code):
            limit_ratio = 0.30
        elif code.startswith(("300", "301", "688")):
            limit_ratio = 0.20
        elif "ST" in name.upper():
            limit_ratio = 0.05
        else:
            limit_ratio = 0.10
        limit_up = math.floor(previous * (1 + limit_ratio) * 100 + 0.5) / 100.0
        limit_down = math.floor(previous * (1 - limit_ratio) * 100 + 0.5) / 100.0
        if abs(current - limit_up) < 0.005:
            limit_up_count += 1
        if abs(current - limit_down) < 0.005:
            limit_down_count += 1

    participation = up_count + down_count + flat_count
    if participation < 3_500:
        logger.warning(
            "[MarketStats] component=market_stats provider=TencentFetcher "
            "api=sina_full_market action=reject reason=insufficient_traded_rows count=%s",
            participation,
        )
        return None
    return {
        "up_count": up_count,
        "down_count": down_count,
        "flat_count": flat_count,
        "limit_up_count": limit_up_count,
        "limit_down_count": limit_down_count,
        "total_amount": total_amount / 1e8,
    }


def _to_tencent_symbol(stock_code: str) -> str:
    code = normalize_stock_code(stock_code)
    if not code or not code.isdigit() or len(code) != 6:
        return ""
    if is_bse_code(code):
        return f"bj{code}"
    if code.startswith(("6", "5", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _estimate_lookback_days(*, start_date: str, end_date: str) -> int:
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        calendar_days = max(1, (end - start).days + 1)
    except ValueError:
        calendar_days = 90
    # Trading days are sparse over calendar days; add margin for holidays/suspensions.
    return max(30, min(_MAX_KLINE_BARS, int(calendar_days * 1.8) + 20))


def _empty_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def _first_returned_date(df: pd.DataFrame) -> Optional[str]:
    if "date" not in df.columns or df.empty:
        return None
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.min().strftime("%Y-%m-%d")


def _is_capped_history_incomplete(
    *,
    first_returned_date: str,
    start_date: str,
    lookback: int,
    returned_rows: int,
) -> bool:
    hit_cap = lookback >= _MAX_KLINE_BARS and returned_rows >= _MAX_KLINE_BARS
    if not hit_cap:
        return False
    try:
        first = datetime.strptime(first_returned_date, "%Y-%m-%d")
        requested_start = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        return False
    return first > _first_trading_date_on_or_after(requested_start)


def _first_trading_date_on_or_after(start_date: datetime) -> datetime:
    if xcals is not None:
        try:
            cal = xcals.get_calendar("XSHG")
            session = cal.date_to_session(start_date.date(), direction="next")
            return datetime.combine(session.date(), datetime.min.time())
        except Exception:
            pass

    current = start_date
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def _format_tencent_date(date_text: str) -> Optional[str]:
    try:
        return datetime.strptime(date_text, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _lots_to_shares(volume: Any) -> Any:
    try:
        return float(volume) * 100
    except (TypeError, ValueError):
        return volume


def _extract_kline_rows(payload: dict[str, Any], *, symbol: str) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    item = data.get(symbol) if isinstance(data, dict) else None
    if not isinstance(item, dict):
        return []
    rows = item.get("qfqday") or item.get("day") or []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        amount: Optional[Any] = row[6] if len(row) > 6 else None
        result.append(
            {
                "date": str(row[0]),
                "open": row[1],
                "close": row[2],
                "high": row[3],
                "low": row[4],
                "volume": _lots_to_shares(row[5]),
                "amount": amount,
            }
        )
    return result
