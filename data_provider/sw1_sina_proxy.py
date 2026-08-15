# -*- coding: utf-8 -*-
"""Strict Sina-based fallback for Shenwan level-1 daily rankings.

This module does not claim to reproduce the official Shenwan index.  It uses
the public SW2021 constituent nodes exposed by Sina Market Center and builds a
one-session free-float-market-cap-weighted return proxy.  The result is only
accepted when all 31 level-1 industries resolve to the same proven trading
date and every industry has adequate constituent coverage.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


logger = logging.getLogger(__name__)

_SINA_NODE_DATA_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData"
)
_SINA_QUOTE_URL = "https://hq.sinajs.cn/list="
_SINA_HEADERS = {
    "Referer": "https://vip.stock.finance.sina.com.cn/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
}
_SINA_QUOTE_HEADERS = {
    "Referer": "https://finance.sina.com.cn/",
    "User-Agent": _SINA_HEADERS["User-Agent"],
}

# Sina's ``sw1_*`` tree follows the 2021 Shenwan classification.  Keeping the
# exact 31-node set here makes the completeness contract explicit and prevents
# a regional, concept or Eastmoney-defined board from being mislabeled SW1.
SW1_2021_NODES: Tuple[Tuple[str, str], ...] = (
    ("110000", "农林牧渔"),
    ("220000", "基础化工"),
    ("230000", "钢铁"),
    ("240000", "有色金属"),
    ("270000", "电子"),
    ("280000", "汽车"),
    ("330000", "家用电器"),
    ("340000", "食品饮料"),
    ("350000", "纺织服饰"),
    ("360000", "轻工制造"),
    ("370000", "医药生物"),
    ("410000", "公用事业"),
    ("420000", "交通运输"),
    ("430000", "房地产"),
    ("450000", "商贸零售"),
    ("460000", "社会服务"),
    ("480000", "银行"),
    ("490000", "非银金融"),
    ("510000", "综合"),
    ("610000", "建筑材料"),
    ("620000", "建筑装饰"),
    ("630000", "电力设备"),
    ("640000", "机械设备"),
    ("650000", "国防军工"),
    ("710000", "计算机"),
    ("720000", "传媒"),
    ("730000", "通信"),
    ("740000", "煤炭"),
    ("750000", "石油石化"),
    ("760000", "环保"),
    ("770000", "美容护理"),
)

_QUOTE_RE = re.compile(r'var hq_str_([^=]+)="([^"]*)";')


@dataclass(frozen=True)
class _IndustrySnapshot:
    code: str
    name: str
    change_pct: float
    valid_count: int
    raw_count: int
    representatives: Tuple[str, ...]

    @property
    def coverage(self) -> float:
        return self.valid_count / self.raw_count if self.raw_count else 0.0


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class SinaSw1ProxyFetcher:
    """Build a fully labelled SW1 constituent-weighted daily proxy."""

    def __init__(
        self,
        *,
        timeout: Tuple[float, float] = (5.0, 15.0),
        attempts: int = 3,
        workers: int = 6,
        page_size: int = 100,
        max_pages: int = 8,
        minimum_industry_coverage: float = 0.80,
    ) -> None:
        self.timeout = timeout
        self.attempts = max(int(attempts), 1)
        self.workers = max(1, min(int(workers), 8))
        self.page_size = min(max(int(page_size), 20), 100)
        self.max_pages = max(int(max_pages), 1)
        self.minimum_industry_coverage = float(minimum_industry_coverage)

    def _get(self, url: str, *, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.attempts + 1):
            try:
                request_kwargs = {
                    "params": params,
                    "headers": (
                        _SINA_QUOTE_HEADERS
                        if url.startswith(_SINA_QUOTE_URL)
                        else _SINA_HEADERS
                    ),
                    "timeout": self.timeout,
                }
                if attempt == 1:
                    response = requests.get(url, **request_kwargs)
                else:
                    # A stale HTTP(S)_PROXY is a common source of apparent
                    # provider outages on desktop and CI runners.  Preserve
                    # the configured route on the first attempt, then retry
                    # directly instead of repeating the identical failure.
                    with requests.Session() as session:
                        session.trust_env = False
                        response = session.get(url, **request_kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.attempts:
                    time.sleep(0.5 * attempt)
        raise RuntimeError(f"Sina request failed after {self.attempts} attempts: {last_error}")

    def _fetch_node_rows(self, code: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for page in range(1, self.max_pages + 1):
            response = self._get(
                _SINA_NODE_DATA_URL,
                params={
                    "node": f"sw1_{code}",
                    "sort": "symbol",
                    "asc": "1",
                    "page": page,
                    "num": self.page_size,
                },
            )
            try:
                page_rows = response.json()
            except ValueError as exc:
                raise RuntimeError(f"Sina SW1 node {code} returned non-JSON data") from exc
            if not isinstance(page_rows, list):
                raise RuntimeError(f"Sina SW1 node {code} returned an invalid payload")
            rows.extend(item for item in page_rows if isinstance(item, dict))
            if len(page_rows) < self.page_size:
                return rows
        raise RuntimeError(f"Sina SW1 node {code} exceeded the pagination safety limit")

    def _build_snapshot(self, code: str, name: str) -> _IndustrySnapshot:
        rows = self._fetch_node_rows(code)
        if not rows:
            raise RuntimeError(f"Sina SW1 node {code} ({name}) is empty")

        weighted_returns: List[Tuple[float, float]] = []
        representatives: List[Tuple[float, str]] = []
        for row in rows:
            current = _safe_float(row.get("trade"))
            previous = _safe_float(row.get("settlement"))
            current_float_mv = _safe_float(row.get("nmc"))
            volume = _safe_float(row.get("volume"))
            symbol = str(row.get("symbol") or "").strip().lower()
            if not (
                current is not None
                and previous is not None
                and current_float_mv is not None
                and volume is not None
                and current > 0
                and previous > 0
                and current_float_mv > 0
                and volume > 0
                and re.fullmatch(r"(?:sh|sz|bj)\d{6}", symbol)
            ):
                continue
            daily_return = current / previous - 1.0
            if not math.isfinite(daily_return) or not -0.35 <= daily_return <= 0.35:
                continue
            prior_float_mv = current_float_mv / (1.0 + daily_return)
            if prior_float_mv <= 0 or not math.isfinite(prior_float_mv):
                continue
            weighted_returns.append((prior_float_mv, daily_return))
            representatives.append((volume, symbol))

        valid_count = len(weighted_returns)
        coverage = valid_count / len(rows)
        if valid_count < 3 or coverage < self.minimum_industry_coverage:
            raise RuntimeError(
                f"Sina SW1 node {code} ({name}) coverage {valid_count}/{len(rows)} is insufficient"
            )
        weight_total = sum(weight for weight, _ in weighted_returns)
        if weight_total <= 0:
            raise RuntimeError(f"Sina SW1 node {code} ({name}) has no valid float-market-cap weight")

        representatives.sort(reverse=True)
        return _IndustrySnapshot(
            code=code,
            name=name,
            change_pct=(
                sum(weight * daily_return for weight, daily_return in weighted_returns)
                / weight_total
                * 100.0
            ),
            valid_count=valid_count,
            raw_count=len(rows),
            representatives=tuple(symbol for _, symbol in representatives[:2]),
        )

    def _fetch_quote_dates(self, symbols: Iterable[str]) -> Dict[str, str]:
        normalized = list(dict.fromkeys(symbol for symbol in symbols if symbol))
        if not normalized:
            return {}
        response = self._get(f"{_SINA_QUOTE_URL}{','.join(normalized)}")
        response.encoding = "gb18030"
        quote_dates: Dict[str, str] = {}
        for symbol, payload in _QUOTE_RE.findall(response.text):
            fields = payload.split(",")
            if len(fields) > 30 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields[30]):
                quote_dates[symbol.lower()] = fields[30]
        return quote_dates

    @staticmethod
    def _prove_common_trade_date(
        snapshots: Iterable[_IndustrySnapshot],
        quote_dates: Dict[str, str],
    ) -> str:
        industry_dates: List[str] = []
        for snapshot in snapshots:
            dates = [
                quote_dates[symbol]
                for symbol in snapshot.representatives
                if symbol in quote_dates
            ]
            if not dates:
                raise RuntimeError(f"Sina SW1 node {snapshot.code} has no dated representative quote")
            industry_dates.append(Counter(dates).most_common(1)[0][0])
        unique_dates = set(industry_dates)
        if len(industry_dates) != len(SW1_2021_NODES) or len(unique_dates) != 1:
            raise RuntimeError(
                "Sina SW1 constituent sessions are not aligned: "
                + ",".join(sorted(unique_dates))
            )
        return industry_dates[0]

    def get_rankings(self, n: int = 5) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        snapshots: List[_IndustrySnapshot] = []
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="sina-sw1") as executor:
            futures = {
                executor.submit(self._build_snapshot, code, name): (code, name)
                for code, name in SW1_2021_NODES
            }
            for future in as_completed(futures):
                code, name = futures[future]
                try:
                    snapshots.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"Sina SW1 node {code} ({name}) failed: {exc}") from exc

        if len(snapshots) != len(SW1_2021_NODES):
            raise RuntimeError(
                f"Sina SW1 universe incomplete: {len(snapshots)}/{len(SW1_2021_NODES)}"
            )
        quote_dates = self._fetch_quote_dates(
            symbol
            for snapshot in snapshots
            for symbol in snapshot.representatives
        )
        trade_date = self._prove_common_trade_date(snapshots, quote_dates)
        minimum_coverage = min(snapshot.coverage for snapshot in snapshots)

        def to_row(snapshot: _IndustrySnapshot) -> Dict[str, Any]:
            return {
                "code": snapshot.code,
                "name": snapshot.name,
                "change_pct": float(snapshot.change_pct),
                "classification": "SW1_PROXY",
                "source": "新浪财经·申万2021成分股流通市值加权代理",
                "method": "prior_free_float_market_cap_weighted_constituent_return",
                "quote_date": trade_date,
                "industry_count": len(SW1_2021_NODES),
                "valid_constituents": snapshot.valid_count,
                "constituent_count": snapshot.raw_count,
                "minimum_universe_coverage": minimum_coverage,
            }

        ranked = sorted(snapshots, key=lambda item: item.change_pct, reverse=True)
        limit = min(max(int(n), 1), len(ranked))
        top = [to_row(item) for item in ranked[:limit]]
        bottom = [to_row(item) for item in reversed(ranked[-limit:])]
        logger.info(
            "[SinaSw1Proxy] built complete SW1 proxy date=%s industries=%s min_coverage=%.1f%%",
            trade_date,
            len(snapshots),
            minimum_coverage * 100.0,
        )
        return top, bottom
