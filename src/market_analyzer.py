# -*- coding: utf-8 -*-
"""
===================================
大盘复盘分析模块
===================================

职责：
1. 获取大盘指数数据（上证、深证、创业板）
2. 搜索市场新闻形成复盘情报
3. 使用大模型生成每日大盘复盘报告
"""

import logging
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from inspect import getattr_static
from typing import Optional, Dict, Any, List

import pandas as pd

from src.config import get_config
from src.report_language import normalize_report_language
from src.search_service import SearchService
from src.core.market_profile import get_profile, MarketProfile
from src.core.market_strategy import get_market_strategy_blueprint
from src.llm.backend_registry import (
    resolve_generation_backend_id,
    resolve_generation_fallback_backend_id,
)
from src.llm.generation_backend import GenerationError
from src.schemas.market_light import MARKET_LIGHT_REGIONS, MarketLightSnapshot
from src.services.run_diagnostics import record_llm_run, record_llm_run_started
from src.services.intelligence_service import IntelligenceService
from data_provider.base import DataFetcherManager

logger = logging.getLogger(__name__)


_ENGLISH_SECTION_PATTERNS = {
    "market_summary": r"###\s*(?:1\.\s*)?Market Summary",
    "index_commentary": r"###\s*(?:2\.\s*)?(?:Index Commentary|Major Indices)",
    "sector_highlights": r"###\s*(?:4\.\s*)?(?:Sector Highlights|Sector/Theme Highlights)",
}

_CHINESE_SECTION_PATTERNS = {
    "market_summary": r"###\s*一、(?:盘面总览|市场总结)",
    "index_commentary": r"###\s*二、(?:指数结构|指数点评|主要指数)",
    "sector_highlights": r"###\s*三、(?:板块主线|热点解读|板块表现)",
    "funds_sentiment": r"###\s*四、(?:资金与情绪|资金动向)",
    "news_catalysts": r"###\s*五、(?:消息催化|后市展望)",
}


@dataclass
class MarketIndex:
    """大盘指数数据"""
    code: str                    # 指数代码
    name: str                    # 指数名称
    current: float = 0.0         # 当前点位
    change: float = 0.0          # 涨跌点数
    change_pct: float = 0.0      # 涨跌幅(%)
    open: float = 0.0            # 开盘点位
    high: float = 0.0            # 最高点位
    low: float = 0.0             # 最低点位
    prev_close: float = 0.0      # 昨收点位
    volume: float = 0.0          # 成交量（手）
    amount: float = 0.0          # 成交额（元）
    previous_amount: float = 0.0 # 前一交易日成交额（元）
    amplitude: float = 0.0       # 振幅(%)
    trade_date: str = ""         # 数据对应交易日
    previous_trade_date: str = "" # 前一交易日
    source: str = ""             # 行情数据源
    fetched_at: str = ""         # 抓取时间（含时区）
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'prev_close': self.prev_close,
            'volume': self.volume,
            'amount': self.amount,
            'previous_amount': self.previous_amount,
            'amplitude': self.amplitude,
            'trade_date': self.trade_date,
            'previous_trade_date': self.previous_trade_date,
            'source': self.source,
            'fetched_at': self.fetched_at,
        }


@dataclass
class MarketOverview:
    """市场概览数据"""
    date: str                           # 日期
    indices: List[MarketIndex] = field(default_factory=list)  # 主要指数
    up_count: int = 0                   # 上涨家数
    down_count: int = 0                 # 下跌家数
    flat_count: int = 0                 # 平盘家数
    limit_up_count: int = 0             # 涨停家数
    limit_down_count: int = 0           # 跌停家数
    total_amount: float = 0.0           # 两市成交额（亿元）
    previous_total_amount: float = 0.0  # 前一交易日两市成交额（亿元）
    turnover_change: float = 0.0        # 两市成交额环比（亿元）
    turnover_change_pct: float = 0.0    # 两市成交额环比（%）
    turnover_trade_date: str = ""       # 前一交易日日期
    market_stats_source: str = ""       # 宽度/涨跌停统计来源
    market_stats_trade_date: str = ""   # 宽度/涨跌停快照对应交易日
    # north_flow: float = 0.0           # 北向资金净流入（亿元）- 已废弃，接口不可用
    
    # 板块涨幅榜
    top_sectors: List[Dict] = field(default_factory=list)     # 涨幅前5板块
    bottom_sectors: List[Dict] = field(default_factory=list)  # 跌幅前5板块
    top_concepts: List[Dict] = field(default_factory=list)    # 涨幅前5概念
    bottom_concepts: List[Dict] = field(default_factory=list) # 跌幅前5概念
    # 运行时采集状态放在末尾，保持既有位置参数兼容。
    indices_attempted: bool = False
    market_stats_attempted: bool = False
    market_stats_available: bool = False
    us_context_attempted: bool = False
    us_market_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketLightReviewResult:
    """Internal market-review parts built from one overview fetch."""

    overview: MarketOverview
    report: str
    market_light_snapshot: Optional[Dict[str, Any]]
    structured_payload: Dict[str, Any] = field(default_factory=dict)


class MarketAnalyzer:
    """
    大盘复盘分析器
    
    功能：
    1. 获取大盘指数实时行情
    2. 获取市场涨跌统计
    3. 获取板块涨跌榜
    4. 搜索市场新闻
    5. 生成大盘复盘报告
    """
    
    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        analyzer=None,
        region: str = "cn",
        config: Optional[Any] = None,
    ):
        """
        初始化大盘分析器

        Args:
            search_service: 搜索服务实例
            analyzer: AI分析器实例（用于调用LLM）
            region: 市场区域 cn=A股 hk=港股 us=美股 jp=日本 kr=韩国
            config: 本次复盘使用的配置；未传时读取全局配置
        """
        self.config = config or get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        self.data_manager = DataFetcherManager()
        self.region = region if region in ("cn", "us", "hk", "jp", "kr") else "cn"
        self.profile: MarketProfile = get_profile(self.region)
        self.strategy = get_market_strategy_blueprint(self.region)

    def _log_context(self) -> str:
        return f"component=market_review region={self.region}"

    def _get_output_language(self) -> str:
        """Return the truthful report language (zh/en/ko) for payload and directives."""
        return normalize_report_language(
            getattr(getattr(self, "config", None), "report_language", "zh")
        )

    def _get_review_language(self) -> str:
        # Structural/template language. Korean reuses the English scaffolding;
        # the Korean output directive is applied in the prompt builder.
        language = self._get_output_language()
        return "en" if language == "ko" else language

    def _get_template_review_language(self) -> str:
        return self._get_review_language()

    def _get_market_scope_name(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "us":
            return "US market" if review_language == "en" else "美股市场"
        if self.region == "hk":
            return "Hong Kong market" if review_language == "en" else "港股市场"
        if self.region == "jp":
            return "Japan market" if review_language == "en" else "日本市场"
        if self.region == "kr":
            return "Korea market" if review_language == "en" else "韩国市场"
        if review_language == "en":
            return "A-share market"
        return "A股市场"

    def _get_turnover_unit_label(self) -> str:
        """Return the turnover unit label for the current market/language."""
        if self.region == "us":
            return "USD bn" if self._get_review_language() == "en" else "十亿美元"
        if self.region == "hk":
            return "HKD bn" if self._get_review_language() == "en" else "十亿港元"
        if self.region == "jp":
            return "JPY bn" if self._get_review_language() == "en" else "十亿日元"
        if self.region == "kr":
            return "KRW bn" if self._get_review_language() == "en" else "十亿韩元"
        return "CNY 100m" if self._get_review_language() == "en" else "亿"

    def _format_turnover_value(self, amount_raw: float) -> str:
        """Format raw turnover according to market-specific units."""
        if amount_raw == 0.0:
            return "N/A"
        if self.region in ("us", "hk", "jp", "kr"):
            return f"{amount_raw / 1e9:.2f}"
        if amount_raw > 1e6:
            return f"{amount_raw / 1e8:.0f}"
        return f"{amount_raw:.0f}"

    @staticmethod
    def _is_positive_number(value: Any) -> bool:
        """Return whether a value is a finite positive number."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return number > 0 and number != float("inf")

    def _recover_cn_total_amount_from_indices(self, overview: MarketOverview) -> None:
        """Recover A-share turnover from the two primary exchange indices.

        Market breadth providers may fail while the index provider still returns
        the Shanghai Composite and Shenzhen Component exchange-wide turnover.
        Only those two non-overlapping primary indices are summed; subset indices
        such as CSI 300, SSE 50, STAR 50, and ChiNext are deliberately ignored.
        """
        if self.region != "cn":
            return

        exchange_amounts: Dict[str, float] = {}
        previous_exchange_amounts: Dict[str, float] = {}
        previous_dates: List[str] = []
        for index in overview.indices:
            code = str(index.code or "").strip().upper()
            name = str(index.name or "").strip()
            digits = re.sub(r"\D", "", code)
            exchange = None
            if digits.endswith("000001") or name in {"上证指数", "上证综指"}:
                exchange = "sh"
            elif digits.endswith("399001") or name == "深证成指":
                exchange = "sz"
            if exchange is None or not self._is_positive_number(index.amount):
                continue

            amount = float(index.amount)
            # Index providers usually return yuan, while MarketOverview stores
            # A-share aggregate turnover in CNY 100m.
            exchange_amounts[exchange] = amount / 1e8 if amount > 1e6 else amount
            if self._is_positive_number(index.previous_amount):
                previous_amount = float(index.previous_amount)
                previous_exchange_amounts[exchange] = (
                    previous_amount / 1e8
                    if previous_amount > 1e6
                    else previous_amount
                )
                if index.previous_trade_date:
                    previous_dates.append(index.previous_trade_date)

        if {"sh", "sz"}.issubset(exchange_amounts):
            provider_total = float(overview.total_amount or 0.0)
            overview.total_amount = round(
                exchange_amounts["sh"] + exchange_amounts["sz"],
                2,
            )
            logger.info(
                "[大盘] %s action=recover_total_amount status=success "
                "source=primary_indices amount=%.0f亿 provider_amount=%.0f亿",
                self._log_context(),
                overview.total_amount,
                provider_total,
            )
        if {"sh", "sz"}.issubset(previous_exchange_amounts):
            overview.previous_total_amount = round(
                previous_exchange_amounts["sh"]
                + previous_exchange_amounts["sz"],
                2,
            )
            overview.turnover_change = round(
                overview.total_amount - overview.previous_total_amount,
                2,
            )
            overview.turnover_change_pct = round(
                overview.turnover_change / overview.previous_total_amount * 100.0,
                4,
            )
            overview.turnover_trade_date = max(previous_dates) if previous_dates else ""

    def _assess_market_data_quality(self, overview: MarketOverview) -> Dict[str, Any]:
        """Assess whether inputs are sufficient for directional market analysis."""
        valid_indices = [
            index
            for index in overview.indices
            if self._is_positive_number(index.current)
        ]
        if self.region == "us":
            core_index_dates = [
                index.trade_date
                for index in valid_indices
                if index.code != "VIX" and index.trade_date
            ]
            latest_index_date = max(core_index_dates) if core_index_dates else ""
            aligned_core_indices = [
                index
                for index in valid_indices
                if index.code != "VIX" and index.trade_date == latest_index_date
            ]
            aligned_codes = {index.code for index in aligned_core_indices}
            required_codes = {"SPX", "IXIC", "DJI"}
            required_index_count = 3 if overview.indices_attempted else 1
            indices_available = bool(
                len(aligned_core_indices) >= required_index_count
                and (
                    not overview.indices_attempted
                    or required_codes.issubset(aligned_codes)
                )
                and all(
                    self._is_positive_number(index.prev_close)
                    and self._is_positive_number(index.open)
                    and self._is_positive_number(index.high)
                    and self._is_positive_number(index.low)
                    for index in aligned_core_indices
                )
            )
            us_context = overview.us_market_context or {}
            us_quality = us_context.get("quality") or {}
            participation = us_context.get("participation") or {}
            sector_rankings = us_context.get("sector_rankings") or {}
            macro = us_context.get("macro") or {}
            context_date = str(us_context.get("as_of") or "")
            date_aligned = bool(
                latest_index_date
                and context_date
                and latest_index_date == context_date
            )
            breadth_available = bool(
                us_quality.get("proxy_ready")
                and participation.get("sector_coverage", 0) >= 8
                and date_aligned
            )
            turnover_available = bool(
                us_quality.get("liquidity_ready")
                and self._is_positive_number(participation.get("spy_volume_ratio_20d"))
                and 0.25 <= float(participation.get("spy_volume_ratio_20d")) <= 4.0
            )
            sector_rankings_available = bool(
                us_quality.get("sector_ready")
                and int(sector_rankings.get("coverage") or 0)
                == int(sector_rankings.get("universe") or 11)
                and sector_rankings.get("top")
                and sector_rankings.get("bottom")
            )
            concept_rankings_available = True
            macro_available = bool(
                us_quality.get("macro_ready")
                and macro.get("DGS2")
                and macro.get("DGS10")
                and all(
                    bool(str((macro.get(series_id) or {}).get("as_of") or ""))
                    and str((macro.get(series_id) or {}).get("as_of") or "")
                    <= latest_index_date
                    for series_id in ("DGS2", "DGS10")
                )
            )

            proxy_consistent = True
            index_by_code = {index.code: index for index in aligned_core_indices}
            proxies = us_context.get("proxies") or {}
            for index_code, proxy_code, tolerance in (
                ("SPX", "SPY", 0.20),
                ("NDX", "QQQ", 0.25),
                ("RUT", "IWM", 0.35),
            ):
                index = index_by_code.get(index_code)
                proxy = proxies.get(proxy_code) or {}
                if not index or proxy.get("change_pct") is None:
                    continue
                if abs(float(index.change_pct) - float(proxy["change_pct"])) > tolerance:
                    proxy_consistent = False
                    break

            missing_core_fields: List[str] = []
            if overview.indices_attempted and not indices_available:
                missing_core_fields.append("major_indices")
            if overview.us_context_attempted and not date_aligned:
                missing_core_fields.append("us_trade_date_alignment")
            if overview.us_context_attempted and not breadth_available:
                missing_core_fields.append("us_participation_proxies")
            if overview.us_context_attempted and not turnover_available:
                missing_core_fields.append("us_liquidity_proxy")
            if overview.us_context_attempted and not sector_rankings_available:
                missing_core_fields.append("us_sector_etfs")
            if overview.us_context_attempted and not macro_available:
                missing_core_fields.append("us_treasury_yields")
            if overview.us_context_attempted and not proxy_consistent:
                missing_core_fields.append("us_index_proxy_consistency")

            missing_optional_fields: List[str] = []
            if "DTWEXBGS" not in macro:
                missing_optional_fields.append("us_dollar_index")

            core_data_ready = not missing_core_fields
            return {
                "status": (
                    "unavailable"
                    if not core_data_ready
                    else ("partial" if missing_optional_fields else "ok")
                ),
                "core_data_ready": core_data_ready,
                "indices_available": indices_available,
                "valid_index_count": len(aligned_core_indices),
                "breadth_available": breadth_available,
                "turnover_available": turnover_available,
                "sector_rankings_available": sector_rankings_available,
                "concept_rankings_available": concept_rankings_available,
                "macro_available": macro_available,
                "index_proxy_consistent": proxy_consistent,
                "trade_date_aligned": date_aligned,
                "missing_core_fields": missing_core_fields,
                "missing_optional_fields": missing_optional_fields,
            }

        required_index_count = 6 if self.region == "cn" and overview.indices_attempted else 1
        cn_index_dates = {
            str(index.trade_date or "")
            for index in valid_indices
            if index.trade_date
        }
        cn_index_dates_aligned = bool(
            self.region != "cn"
            or not overview.indices_attempted
            or (
                len(cn_index_dates) == 1
                and next(iter(cn_index_dates), "") == str(overview.date or "")
            )
        )
        indices_available = bool(
            len(valid_indices) >= required_index_count
            and cn_index_dates_aligned
            and (
                self.region != "cn"
                or not overview.indices_attempted
                or all(
                    self._is_positive_number(index.prev_close)
                    and self._is_positive_number(index.open)
                    and self._is_positive_number(index.high)
                    and self._is_positive_number(index.low)
                    and self._is_positive_number(index.amount)
                    and bool(index.trade_date)
                    for index in valid_indices
                )
            )
        )

        breadth_total = (
            max(int(overview.up_count or 0), 0)
            + max(int(overview.down_count or 0), 0)
            + max(int(overview.flat_count or 0), 0)
        )
        market_stats_date_aligned = bool(
            not overview.market_stats_attempted
            or (
                overview.market_stats_trade_date
                and overview.market_stats_trade_date == overview.date
            )
        )
        breadth_available = bool(
            not self.profile.has_market_stats
            or (
                breadth_total > 0
                and market_stats_date_aligned
                and (
                    not overview.market_stats_attempted
                    or overview.market_stats_available
                )
            )
        )
        turnover_available = bool(
            not self.profile.has_market_stats
            or self._is_positive_number(overview.total_amount)
        )
        turnover_comparison_available = bool(
            not self.profile.has_market_stats
            or (
                self._is_positive_number(overview.previous_total_amount)
                and bool(overview.turnover_trade_date)
            )
        )
        sector_rows = list(overview.top_sectors or []) + list(overview.bottom_sectors or [])

        def is_accepted_cn_sector_row(item: Any) -> bool:
            if not isinstance(item, dict):
                return False
            classification = str(item.get("classification") or "").upper()
            if classification == "SW1":
                return True
            return bool(
                classification == "SW1_PROXY"
                and item.get("quote_date") == overview.date
                and int(item.get("industry_count") or 0) == 31
                and float(item.get("minimum_universe_coverage") or 0.0) >= 0.80
                and str(item.get("method") or "")
                == "prior_free_float_market_cap_weighted_constituent_return"
            )

        sector_rankings_available = bool(
            not self.profile.has_sector_rankings
            or (
                sector_rows
                and (
                    self.region != "cn"
                    or not overview.indices_attempted
                    or all(is_accepted_cn_sector_row(item) for item in sector_rows)
                )
            )
        )
        concept_rankings_available = bool(
            not self.profile.has_sector_rankings
            or overview.top_concepts
            or overview.bottom_concepts
        )

        missing_core_fields: List[str] = []
        if overview.indices_attempted and not indices_available:
            missing_core_fields.append("major_indices")
        if (
            self.region == "cn"
            and overview.indices_attempted
            and not cn_index_dates_aligned
        ):
            missing_core_fields.append("index_trade_date_alignment")
        if (
            self.profile.has_market_stats
            and overview.market_stats_attempted
            and not breadth_available
        ):
            missing_core_fields.append("market_breadth")
        if (
            self.profile.has_market_stats
            and overview.market_stats_attempted
            and not market_stats_date_aligned
        ):
            missing_core_fields.append("market_breadth_trade_date")
        if (
            self.profile.has_market_stats
            and overview.market_stats_attempted
            and not turnover_available
        ):
            missing_core_fields.append("aggregate_turnover")
        if (
            self.profile.has_market_stats
            and overview.market_stats_attempted
            and not turnover_comparison_available
        ):
            missing_core_fields.append("prior_session_turnover")
        if (
            self.region == "cn"
            and overview.indices_attempted
            and not sector_rankings_available
        ):
            missing_core_fields.append("sw1_sector_rankings")

        missing_optional_fields: List[str] = []
        if self.profile.has_sector_rankings and not sector_rankings_available:
            missing_optional_fields.append("sector_rankings")
        if self.profile.has_sector_rankings and not concept_rankings_available:
            missing_optional_fields.append("concept_rankings")

        core_data_ready = not missing_core_fields
        if not core_data_ready:
            status = "unavailable"
        elif missing_optional_fields:
            status = "partial"
        else:
            status = "ok"

        return {
            "status": status,
            "core_data_ready": core_data_ready,
            "indices_available": indices_available,
            "valid_index_count": len(valid_indices),
            "index_trade_date_aligned": cn_index_dates_aligned,
            "breadth_available": breadth_available,
            "turnover_available": turnover_available,
            "turnover_comparison_available": turnover_comparison_available,
            "market_stats_date_aligned": market_stats_date_aligned,
            "sector_rankings_available": sector_rankings_available,
            "concept_rankings_available": concept_rankings_available,
            "missing_core_fields": missing_core_fields,
            "missing_optional_fields": missing_optional_fields,
        }

    def _get_index_change_arrow(self, change_pct: float) -> str:
        if change_pct == 0:
            return "⚪"
        color_scheme = getattr(getattr(self, "config", None), "market_review_color_scheme", "green_up")
        if color_scheme == "red_up":
            return "🔴" if change_pct > 0 else "🟢"
        return "🟢" if change_pct > 0 else "🔴"

    def _get_review_title(self, date: str) -> str:
        if self._get_review_language() == "en":
            market_names = {
                "us": "US Market Recap",
                "hk": "HK Market Recap",
                "jp": "Japan Market Recap",
                "kr": "Korea Market Recap",
            }
            market_name = market_names.get(self.region, "A-share Market Recap")
            return f"## {date} {market_name}"
        return f"## {date} 大盘复盘"

    def _get_index_hint(self) -> str:
        if self._get_review_language() == "en":
            if self.region == "us":
                return "Analyze the key moves in the S&P 500, Nasdaq, Dow, and other major indices."
            if self.region == "hk":
                return "Analyze the key moves in the HSI, Hang Seng Tech, HSCEI, and other major indices."
            if self.region == "jp":
                return "Analyze the key moves in the Nikkei 225, TOPIX, and other major Japanese indices."
            if self.region == "kr":
                return "Analyze the key moves in the KOSPI, KOSDAQ, and other major Korean indices."
            return "Analyze the price action in the SSE, SZSE, ChiNext, and other major indices."
        return self.profile.prompt_index_hint

    def _get_strategy_prompt_block(self) -> str:
        if self.region == "hk" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Hong Kong Market Regime Strategy
Focus on HSI trend, southbound flow dynamics, and sector rotation to define next-session risk posture.

### Strategy Principles
- Read market regime from HSI, HSTECH, and HSCEI alignment first.
- Track southbound capital flow as a key sentiment driver.
- Translate recap into actionable risk-on/risk-off stance with clear invalidation points.

### Analysis Dimensions
- Trend Regime: Classify the market as momentum, range, or risk-off.
  - Are HSI/HSTECH/HSCEI directionally aligned
  - Did volume confirm the move
  - Are key index levels reclaimed or lost
- Capital Flows: Map southbound flow and macro narrative into equity risk appetite.
  - Southbound net flow direction and magnitude
  - USD/HKD and China policy implications
  - Breadth and leadership concentration
- Sector Themes: Identify persistent leaders and vulnerable laggards.
  - Tech/internet platform trend persistence
  - Financials/property sensitivity to policy shifts
  - Defensive vs growth factor rotation

### Action Framework
- Risk-on: broad index breakout with expanding southbound participation.
- Neutral: mixed index signals; focus on selective relative strength.
- Risk-off: failed breakouts and rising volatility; prioritize capital preservation."""
        if self.region == "jp" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Japan Market Regime Strategy
Focus on Nikkei 225, TOPIX, currency dynamics, and global risk appetite to define the next-session trading plan.

### Strategy Principles
- Read Nikkei 225 and TOPIX alignment first, then assess yen moves, semiconductor/export chains, and financials.
- Translate index conclusions into position sizing, trading pace, and risk-control actions.
- Base judgments only on available index data, news, and price action without inventing breadth or sector statistics.

### Analysis Dimensions
- Trend Regime: Classify Japan equities as advancing, range-bound, or defensive.
  - Are Nikkei 225 and TOPIX directionally aligned
  - Have key index ranges been reclaimed or lost
  - Are large-cap weights and growth chains moving together
- Macro & FX: Map yen, rates, and global risk appetite into equity impact.
  - Yen direction and implications for exporters
  - Bank of Japan and US Treasury yield narratives
  - Overseas technology and semiconductor read-through
- Theme Signals: Identify durable leadership and crowded areas to avoid.
  - Semiconductor, automation, and auto-chain persistence
  - Rotation between financials and domestic-demand stocks
  - Whether news catalysts confirm price action

### Action Framework
- Risk-on: major indices rise together with improving external risk appetite and stronger leadership.
- Neutral: index divergence or FX disruption; avoid chasing and wait for confirmation.
- Risk-off: major indices weaken or external risk rises; prioritize position control."""
        if self.region == "kr" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Korea Market Regime Strategy
Focus on KOSPI, KOSDAQ, semiconductor heavyweights, and global technology risk appetite to define the next-session trading plan.

### Strategy Principles
- Read KOSPI and KOSDAQ alignment first, then assess heavyweight signals from Samsung Electronics, SK Hynix, and related technology leaders.
- Separate broad index beta, semiconductor cycle exposure, and growth-stock risk appetite.
- Base judgments only on available index data, news, and price action without inventing breadth or sector statistics.

### Analysis Dimensions
- Trend Regime: Classify Korea equities as advancing, range-bound, or defensive.
  - Are KOSPI and KOSDAQ directionally aligned
  - Are heavyweight technology names supporting the indices
  - Have key support or resistance levels been reclaimed or lost
- Technology Cycle: Map semiconductor, AI hardware, and global technology moves into Korea equity risk.
  - Memory and semiconductor-chain catalysts
  - US technology-market read-through
  - Foreign investor risk appetite signals
- Theme Signals: Identify durable leadership and crowded areas to avoid.
  - Rotation across batteries, autos, and internet platforms
  - KOSDAQ growth-stock risk appetite
  - Whether news catalysts confirm price action

### Action Framework
- Risk-on: KOSPI and KOSDAQ rise together with confirmed technology leadership and improving external risk appetite.
- Neutral: index or heavyweight divergence; keep sizing controlled and wait for confirmation.
- Risk-off: technology heavyweights weaken or external risk rises; prioritize drawdown control."""
        if self.region == "us" and self._get_review_language() == "zh":
            return """## 美股市场严格多因子复盘策略
采用与A股复盘相同的“行情事实→结构验证→催化核验→次日计划”链路，但指标按美股制度做等价映射。

### 策略原则
- 只引用输入中带数值、数据日和来源的事实；新闻只能陈述标题、来源、日期及其可验证影响链。
- 先核对标普500、纳指、道指、罗素2000与VIX，再核对等权/市值加权、大小盘和11个行业ETF。
- 同时检查2年/10年美债、美元、联储/财政/监管/关税/地缘政治以及标普500公司财报和指引。
- 价格与新闻矛盾时，以价格、市场宽度代理和利率定价为主，不用情绪化措辞强行解释。
- 风险偏好只能由价格、参与度、流动性、波动率与宏观定价共同确认，不能由标题情绪单独决定。
- 数据缺失时必须明确写“无法验证”并降低结论与仓位确定性，禁止补数。

### 分析维度
- 趋势结构：指数是否共振，科技/小盘是否背离，VIX是否确认风险方向。
- 参与度与流动性：RSP相对SPY、IWM相对SPY、行业ETF上涨覆盖及SPY量比是否确认指数。
- 宏观政策：短长端美债、美元以及联储/财政/监管/关税/政治事件如何改变贴现率和风险溢价。
- 财报业绩：盈利、营收、利润率、自由现金流和管理层指引是否支持估值与行业轮动。
- 新闻催化：只采用有来源和日期的最新信息，区分已发生事实、市场预期与未经证实观点。

### 行动框架
- 进攻：指数共振、参与度扩散、量能确认、VIX/利率不构成反向压力。
- 均衡：指数上涨但等权/小盘/行业覆盖未确认，或宏观与财报信号互相抵消。
- 防守：指数转弱、参与度收缩、VIX上升且短端利率/美元同步施压。"""
        if not (self.region == "cn" and self._get_review_language() == "en"):
            return self.strategy.to_prompt_block()
        return """## Strategy Blueprint: A-share Three-Phase Recap Strategy
Focus on index trend, liquidity, and sector rotation to shape the next-session trading plan.

### Strategy Principles
- Read index direction first, then confirm liquidity structure, and finally test sector persistence.
- Every conclusion must map to position sizing, trading pace, and risk-control actions.
- Base judgments on today's data and the latest 3-day news flow without inventing unverified information.

### Analysis Dimensions
- Trend Structure: Determine whether the market is in an uptrend, range, or defensive phase.
  - Are the SSE, SZSE, and ChiNext moving in the same direction
  - Is the market advancing on expanding volume or slipping on contracting volume
  - Have key support or resistance levels been reclaimed or broken
- Liquidity & Sentiment: Identify near-term risk appetite and market temperature.
  - Advance/decline breadth and limit-up/limit-down structure
  - Whether turnover is expanding or fading
  - Whether high-beta leaders are showing divergence
- Leading Themes: Distill tradable leadership and areas to avoid.
  - Whether leading sectors have clear event catalysts
  - Whether sector leaders are pulling the group higher
  - Whether weakness is broadening across lagging sectors

### Action Framework
- Offensive: indices rise in sync, turnover expands, and core themes strengthen.
- Balanced: index divergence or low-volume consolidation; keep sizing controlled and wait for confirmation.
- Defensive: indices weaken and laggards broaden; prioritize risk control and de-risking."""

    def _get_strategy_markdown_block(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "hk" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify the market as momentum, range, or risk-off based on HSI/HSTECH/HSCEI alignment.
- **Capital Flows**: Track southbound flow direction and macro narrative for risk appetite signals.
- **Sector Themes**: Focus on tech/internet platform persistence and financials/property policy sensitivity.
"""
        if self.region == "jp" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify Japan equities as advancing, range-bound, or defensive based on Nikkei 225/TOPIX alignment.
- **Macro & FX**: Track yen, rates, and global risk appetite for exporter and financial-sector implications.
- **Theme Signals**: Focus on semiconductor, automation, auto-chain, financial, and domestic-demand rotation.
"""
        if self.region == "kr" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify Korea equities as advancing, range-bound, or defensive based on KOSPI/KOSDAQ alignment.
- **Technology Cycle**: Track semiconductor, AI hardware, and global technology read-through for market risk appetite.
- **Theme Signals**: Focus on battery, auto, internet-platform, and KOSDAQ growth-stock rotation.
"""
        if self.region == "us" and review_language == "zh":
            return """### 六、策略框架
- **趋势结构**：判断市场在进攻、震荡与防守中的状态是否一致。
- **资金与情绪**：结合波动率、宽度和主题轮动评估风险偏好。
- **主题主线**：识别可延续和可放大的行业主线与防守线索。
"""
        if not (self.region == "cn" and review_language == "en"):
            return self.strategy.to_markdown_block()
        return """### 6. Strategy Framework
- **Trend Structure**: Determine whether the market is in an uptrend, range, or defensive phase.
- **Liquidity & Sentiment**: Track breadth, turnover expansion, and whether leaders are diverging.
- **Leading Themes**: Focus on sectors with catalysts and sustained leadership while avoiding broadening weakness.
"""

    def _get_market_mood_text(self, mood_key: str, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if review_language == "en":
            mapping = {
                "strong_up": "strong gains",
                "mild_up": "moderate gains",
                "mild_down": "mild losses",
                "strong_down": "clear weakness",
                "range": "range-bound trading",
            }
        else:
            mapping = {
                "strong_up": "强势上涨",
                "mild_up": "小幅上涨",
                "mild_down": "小幅下跌",
                "strong_down": "明显下跌",
                "range": "震荡整理",
            }
        return mapping[mood_key]

    def get_market_overview(self) -> MarketOverview:
        """
        获取市场概览数据
        
        Returns:
            MarketOverview: 市场概览数据对象
        """
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)
        
        # 1. 获取主要指数行情（按 region 切换 A 股/美股）
        overview.indices = self._get_main_indices()
        overview.indices_attempted = True
        trade_dates = [
            index.trade_date
            for index in overview.indices
            if index.code != "VIX" and index.trade_date
        ]
        if trade_dates:
            overview.date = max(trade_dates)

        # 2. 美股使用透明等价指标；A股继续使用交易所宽度统计。
        if self.region == "us":
            self._get_us_market_context(overview)
        elif self.profile.has_market_stats:
            self._get_market_statistics(overview)
            self._recover_cn_total_amount_from_indices(overview)

        # 3. A股板块/题材；美股板块已由 11 个 S&P 行业 ETF 填充。
        if self.profile.has_sector_rankings and self.region != "us":
            self._get_sector_rankings(overview)
            self._get_concept_rankings(overview)
        
        # 4. 获取北向资金（可选）
        # self._get_north_flow(overview)
        
        return overview

    
    def _get_main_indices(self) -> List[MarketIndex]:
        """获取主要指数实时行情"""
        indices = []

        try:
            logger.info("[大盘] %s action=get_main_indices status=start", self._log_context())

            # 使用 DataFetcherManager 获取指数行情（按 region 切换）
            data_list = self.data_manager.get_main_indices(region=self.region)

            if data_list:
                for item in data_list:
                    current = float(item.get('current') or 0.0)
                    prev_close = float(item.get('prev_close') or 0.0)
                    provider_change = float(item.get('change') or 0.0)
                    provider_change_pct = float(item.get('change_pct') or 0.0)

                    # 涨跌额/幅必须与报告展示的现价和昨收同源且
                    # 算术一致。这里是所有指数供应商的最后一道校验，
                    # 避免任一备用源把缺失值默认成真实的 0.00%。
                    if current > 0 and prev_close > 0:
                        derived_change = current - prev_close
                        derived_change_pct = derived_change / prev_close * 100.0
                        if (
                            abs(provider_change - derived_change) > 0.01
                            or abs(provider_change_pct - derived_change_pct) > 0.03
                        ):
                            logger.warning(
                                "[大盘] %s index=%s action=reconcile_index_quote "
                                "provider_change=%.4f provider_pct=%.4f "
                                "derived_change=%.4f derived_pct=%.4f",
                                self._log_context(),
                                item.get('code', ''),
                                provider_change,
                                provider_change_pct,
                                derived_change,
                                derived_change_pct,
                            )
                        provider_change = derived_change
                        provider_change_pct = derived_change_pct
                    index = MarketIndex(
                        code=item['code'],
                        name=item['name'],
                        current=current,
                        change=provider_change,
                        change_pct=provider_change_pct,
                        open=item['open'],
                        high=item['high'],
                        low=item['low'],
                        prev_close=prev_close,
                        volume=item['volume'],
                        amount=item['amount'],
                        previous_amount=float(item.get('previous_amount') or 0.0),
                        amplitude=item['amplitude'],
                        trade_date=str(item.get("trade_date") or ""),
                        previous_trade_date=str(
                            item.get("previous_trade_date") or ""
                        ),
                        source=str(item.get("source") or ""),
                        fetched_at=str(item.get("fetched_at") or ""),
                    )
                    indices.append(index)

            if not indices:
                logger.warning("[大盘] %s action=get_main_indices status=empty", self._log_context())
            else:
                logger.info(
                    "[大盘] %s action=get_main_indices status=success count=%d",
                    self._log_context(),
                    len(indices),
                )

        except Exception as e:
            logger.error("[大盘] %s action=get_main_indices status=failed error=%s", self._log_context(), e)

        return indices

    def _get_us_market_context(self, overview: MarketOverview) -> None:
        """Fetch and attach strict US participation, sector, and macro context."""
        overview.us_context_attempted = True
        try:
            logger.info("[大盘] %s action=get_us_market_context status=start", self._log_context())
            context = self.data_manager.get_us_market_context()
            overview.us_market_context = context or {}
            rankings = overview.us_market_context.get("sector_rankings") or {}
            overview.top_sectors = list(rankings.get("top") or [])
            overview.bottom_sectors = list(rankings.get("bottom") or [])
            quality = overview.us_market_context.get("quality") or {}
            logger.info(
                "[大盘] %s action=get_us_market_context status=%s as_of=%s "
                "sector_coverage=%s missing=%s",
                self._log_context(),
                quality.get("status", "empty"),
                overview.us_market_context.get("as_of", ""),
                rankings.get("coverage", 0),
                ",".join(quality.get("missing_core_fields") or []),
            )
        except Exception as exc:
            overview.us_market_context = {}
            logger.error(
                "[大盘] %s action=get_us_market_context status=failed error=%s",
                self._log_context(),
                exc,
            )

    def _get_market_statistics(self, overview: MarketOverview):
        """获取市场涨跌统计"""
        overview.market_stats_attempted = True
        overview.market_stats_available = False
        try:
            logger.info("[大盘] %s action=get_market_stats status=start", self._log_context())

            stats = self.data_manager.get_market_stats(purpose=f"market_review:{self.region}")

            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)
                overview.market_stats_source = str(stats.get('_source') or '')
                overview.market_stats_trade_date = str(
                    stats.get("trade_date") or self._resolve_cn_stats_trade_date()
                )
                overview.market_stats_available = bool(
                    overview.up_count
                    + overview.down_count
                    + overview.flat_count
                    > 0
                )

                if overview.market_stats_available:
                    logger.info(
                        "[大盘] %s action=get_market_stats status=success up=%s down=%s flat=%s "
                        "limit_up=%s limit_down=%s amount=%.0f亿",
                        self._log_context(),
                        overview.up_count,
                        overview.down_count,
                        overview.flat_count,
                        overview.limit_up_count,
                        overview.limit_down_count,
                        overview.total_amount,
                    )
                else:
                    logger.warning(
                        "[大盘] %s action=get_market_stats status=invalid "
                        "reason=zero_market_breadth",
                        self._log_context(),
                    )
            else:
                logger.warning("[大盘] %s action=get_market_stats status=empty", self._log_context())

        except Exception as e:
            logger.error("[大盘] %s action=get_market_stats status=failed error=%s", self._log_context(), e)

    @staticmethod
    def _resolve_cn_stats_trade_date() -> str:
        """Date a realtime A-share breadth snapshot actually represents.

        Before the close, daily bars intentionally stop at the previous fully
        completed session, while realtime breadth already belongs to today's
        partial session.  Labelling the two dates separately lets validation
        block that otherwise invisible mixed-session report.
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
                return get_market_now("cn").date().isoformat()
            return get_effective_trading_date("cn").isoformat()
        except Exception as exc:
            logger.warning("[MarketStats] 无法标记A股宽度交易日: %s", exc)
            return ""

    def _get_sector_rankings(self, overview: MarketOverview):
        """获取板块涨跌榜"""
        try:
            logger.info("[大盘] %s action=get_sector_rankings status=start", self._log_context())

            sw1_method = getattr(self.data_manager, "get_sw1_sector_rankings", None)
            if callable(sw1_method):
                top_sectors, bottom_sectors = sw1_method(5)
            else:
                top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)

            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors

                logger.info(
                    "[大盘] %s action=get_sector_rankings status=success top=%s bottom=%s",
                    self._log_context(),
                    [s['name'] for s in overview.top_sectors],
                    [s['name'] for s in overview.bottom_sectors],
                )
            else:
                logger.warning("[大盘] %s action=get_sector_rankings status=empty", self._log_context())

        except Exception as e:
            logger.error("[大盘] %s action=get_sector_rankings status=failed error=%s", self._log_context(), e)

    def _get_concept_rankings(self, overview: MarketOverview):
        """获取概念/题材涨跌榜（fail-open）。"""
        try:
            logger.info("[大盘] %s action=get_concept_rankings status=start", self._log_context())

            top_concepts, bottom_concepts = self.data_manager.get_concept_rankings(5)

            if top_concepts or bottom_concepts:
                overview.top_concepts = top_concepts
                overview.bottom_concepts = bottom_concepts

                logger.info(
                    "[大盘] %s action=get_concept_rankings status=success top=%s bottom=%s",
                    self._log_context(),
                    [s.get('name') for s in overview.top_concepts],
                    [s.get('name') for s in overview.bottom_concepts],
                )
            else:
                logger.warning("[大盘] %s action=get_concept_rankings status=empty", self._log_context())

        except Exception as e:
            logger.warning("[大盘] %s action=get_concept_rankings status=failed error=%s", self._log_context(), e)
    
    # def _get_north_flow(self, overview: MarketOverview):
    #     """获取北向资金流入"""
    #     try:
    #         logger.info("[大盘] 获取北向资金...")
    #         
    #         # 获取北向资金数据
    #         df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
    #         
    #         if df is not None and not df.empty:
    #             # 取最新一条数据
    #             latest = df.iloc[-1]
    #             if '当日净流入' in df.columns:
    #                 overview.north_flow = float(latest['当日净流入']) / 1e8  # 转为亿元
    #             elif '净流入' in df.columns:
    #                 overview.north_flow = float(latest['净流入']) / 1e8
    #                 
    #             logger.info(f"[大盘] 北向资金净流入: {overview.north_flow:.2f}亿")
    #             
    #     except Exception as e:
    #         logger.warning(f"[大盘] 获取北向资金失败: {e}")
    
    def search_market_news(self) -> List[Dict]:
        """
        搜索市场新闻
        
        Returns:
            新闻列表
        """
        if not self.search_service:
            logger.warning(
                "[大盘] %s action=search_market_news status=skipped reason=no_search_service",
                self._log_context(),
            )
            return []
        
        all_news = []

        # 按 region 使用不同的新闻搜索词
        search_queries = self.profile.news_queries
        review_language = self._get_review_language()
        market_names = {
            "cn": "大盘" if review_language == "zh" else "A-share market",
            "us": "美股市场" if review_language == "zh" else "US market",
            "hk": "港股市场" if review_language == "zh" else "HK market",
            "jp": "日本股市" if review_language == "zh" else "Japan stock market",
            "kr": "韩国股市" if review_language == "zh" else "Korea stock market",
        }
        
        try:
            logger.info("[大盘] %s action=search_market_news status=start", self._log_context())
            
            # 根据 region 设置搜索上下文名称，避免美股搜索被解读为 A 股语境
            market_name = market_names.get(self.region, "大盘")
            for query in search_queries:
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name=market_name,
                    max_results=2,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(
                        "[大盘] %s action=search_market_news status=query_success count=%d",
                        self._log_context(),
                        len(response.results),
                    )
            
            logger.info(
                "[大盘] %s action=search_market_news status=success count=%d",
                self._log_context(),
                len(all_news),
            )
            
        except Exception as e:
            logger.error("[大盘] %s action=search_market_news status=failed error=%s", self._log_context(), e)
        
        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用大模型生成大盘复盘报告
        
        Args:
            overview: 市场概览数据
            news: 市场新闻列表 (SearchResult 对象列表)
            
        Returns:
            大盘复盘报告文本
        """
        quality = self._assess_market_data_quality(overview)
        if not quality["core_data_ready"]:
            logger.error(
                "[大盘] %s action=generate_review status=blocked reason=data_quality "
                "missing_core_fields=%s",
                self._log_context(),
                ",".join(quality["missing_core_fields"]),
            )
            return self._generate_data_unavailable_review(overview, quality)

        if self.region == "us":
            logger.info(
                "[大盘] %s action=generate_review status=deterministic "
                "reason=us_verified_fact_policy",
                self._log_context(),
            )
            return self._generate_strict_data_review(overview, news)

        if getattr(self.config, "market_review_strict_data_only", False) is True:
            logger.info(
                "[大盘] %s action=generate_review status=deterministic "
                "reason=strict_data_only",
                self._log_context(),
            )
            return self._generate_strict_data_review(overview, news)

        backend_error = self._get_analyzer_generation_backend_config_error()
        if backend_error is not None:
            logger.error(
                "[大盘] %s action=generate_review status=failed error_type=%s error=%s",
                self._log_context(),
                type(backend_error).__name__,
                backend_error,
            )
            record_llm_run(
                success=False,
                provider="litellm",
                model=getattr(self.config, "litellm_model", None),
                call_type="market_review",
                error_type=type(backend_error).__name__,
                error_message=backend_error,
            )
            raise backend_error

        if not self.analyzer or not self.analyzer.is_available():
            logger.warning(
                "[大盘] %s action=generate_review status=fallback_template reason=no_analyzer",
                self._log_context(),
            )
            return self._generate_template_review(overview, news)

        # 构建 Prompt
        prompt = self._build_review_prompt(overview, news)

        logger.info("[大盘] %s action=generate_review status=start", self._log_context())
        # Use the public generate_text() entry point - never access private analyzer attributes.
        llm_started_at = time.perf_counter()
        try:
            record_llm_run_started(
                provider="litellm",
                model=getattr(self.config, "litellm_model", None),
                call_type="market_review",
            )
            review = self.analyzer.generate_text(prompt, max_tokens=8192, temperature=0.2)
        except Exception as exc:
            record_llm_run(
                success=False,
                provider="litellm",
                model=getattr(self.config, "litellm_model", None),
                call_type="market_review",
                duration_ms=int((time.perf_counter() - llm_started_at) * 1000),
                error_type=type(exc).__name__,
                error_message=exc,
            )
            raise

        record_llm_run(
            success=bool(review),
            provider="litellm",
            model=getattr(self.config, "litellm_model", None),
            call_type="market_review",
            duration_ms=int((time.perf_counter() - llm_started_at) * 1000),
            error_type=None if review else "EmptyResponse",
            error_message=None if review else "empty market review response",
        )

        if review:
            logger.info(
                "[大盘] %s action=generate_review status=success length=%d",
                self._log_context(),
                len(review),
            )
            # Inject structured data tables into LLM prose sections
            return self._inject_data_into_review(review, overview, news)

        logger.warning(
            "[大盘] %s action=generate_review status=fallback_template reason=empty_llm_response",
            self._log_context(),
        )
        return self._generate_template_review(overview, news)

    def _get_analyzer_generation_backend_config_error(self) -> Optional[GenerationError]:
        """Return analyzer backend config errors without relying on dynamic mock attributes."""
        if self.analyzer is None:
            try:
                resolve_generation_backend_id(self.config)
                resolve_generation_fallback_backend_id(self.config)
            except GenerationError as exc:
                return exc
            return None
        missing = object()
        if getattr_static(self.analyzer, "get_generation_backend_config_error", missing) is missing:
            return None
        method = getattr(self.analyzer, "get_generation_backend_config_error", None)
        if not callable(method):
            return None
        error = method()
        return error if isinstance(error, GenerationError) else None

    def build_market_review_payload(
        self,
        overview: MarketOverview,
        news: List,
        report: str,
        market_light_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the structured market-review contract consumed by API, Web, and notifications."""
        language = self._get_output_language()
        sections = self._split_report_sections(report)
        title = self._extract_report_title(report) or self._get_review_title(overview.date).lstrip("# ").strip()
        quality = self._assess_market_data_quality(overview)
        light = market_light_snapshot
        if (
            light is None
            and self._supports_market_light()
            and quality["core_data_ready"]
        ):
            light = self.build_market_light_snapshot(overview)

        has_breadth_data = bool(
            self.region != "us"
            and
            self.profile.has_market_stats
            and quality["breadth_available"]
            and quality["turnover_available"]
        )

        payload = {
            "version": 1,
            "kind": "market_review",
            "region": self.region,
            "language": language,
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "date": overview.date,
            "market_scope": self._get_market_scope_name(language),
            "indices": [idx.to_dict() for idx in overview.indices],
            "sectors": {
                "top": list(overview.top_sectors or []),
                "bottom": list(overview.bottom_sectors or []),
            },
            "concepts": {
                "top": list(overview.top_concepts or []),
                "bottom": list(overview.bottom_concepts or []),
            },
            "news": [self._normalize_news_item(item) for item in (news or [])[:8]],
            "sections": sections,
            "markdown_report": report,
            "data_quality": quality,
        }

        if light is not None:
            payload["market_light"] = light

        if self.region == "us" and overview.us_market_context:
            payload["us_market_context"] = dict(overview.us_market_context)
            payload["participation"] = dict(
                overview.us_market_context.get("participation") or {}
            )
            payload["macro"] = dict(
                overview.us_market_context.get("macro") or {}
            )
            payload["us_score_dimensions"] = dict(
                self._build_us_market_light_scores(overview).get("dimensions") or {}
            )

        if has_breadth_data:
            payload["breadth"] = {
                "up_count": overview.up_count,
                "down_count": overview.down_count,
                "flat_count": overview.flat_count,
                "limit_up_count": overview.limit_up_count,
                "limit_down_count": overview.limit_down_count,
                "total_amount": overview.total_amount,
                "previous_total_amount": overview.previous_total_amount,
                "turnover_change": overview.turnover_change,
                "turnover_change_pct": overview.turnover_change_pct,
                "previous_trade_date": overview.turnover_trade_date,
                "market_stats_trade_date": overview.market_stats_trade_date,
                "source": overview.market_stats_source,
                "turnover_unit": self._get_turnover_unit_label(),
            }

        return payload

    def _supports_market_light(self) -> bool:
        return self.region in MARKET_LIGHT_REGIONS

    @staticmethod
    def _extract_report_title(report: str) -> str:
        for line in (report or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return ""

    @classmethod
    def _split_report_sections(cls, report: str) -> List[Dict[str, str]]:
        text = (report or "").strip()
        if not text:
            return []
        matches = list(re.finditer(r"^(#{2,3})\s+(.+?)\s*$", text, flags=re.MULTILINE))
        if not matches:
            return [{"key": "full_review", "title": "Review", "markdown": text}]

        sections: List[Dict[str, str]] = []
        first_match = matches[0]
        starts_with_report_title = first_match.start() == 0 and first_match.group(1) == "##"
        content_start_index = 1 if starts_with_report_title else 0
        intro_start = first_match.end() if starts_with_report_title else 0
        intro_end = (
            matches[1].start()
            if starts_with_report_title and len(matches) > 1
            else (len(text) if starts_with_report_title else matches[0].start())
        )
        intro = text[intro_start:intro_end].strip()
        if intro:
            sections.append({"key": "overview", "title": "Overview", "markdown": intro})

        for index, match in enumerate(matches[content_start_index:], start=content_start_index):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            title = match.group(2).strip()
            markdown = text[start:end].strip()
            if not markdown:
                continue
            key = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", title).strip("_").lower()
            sections.append({
                "key": key or f"section_{index + 1}",
                "title": title,
                "markdown": markdown,
            })
        return sections

    @classmethod
    def _normalize_news_item(cls, item: Any) -> Dict[str, str]:
        return {
            "title": cls._compact_news_text(cls._get_news_field(item, "title"), limit=120),
            "snippet": cls._compact_news_text(cls._get_news_field(item, "snippet"), limit=260),
            "source": cls._compact_news_text(cls._get_news_field(item, "source"), limit=80),
            "published_date": cls._compact_news_text(cls._get_news_field(item, "published_date"), limit=40),
            "url": cls._compact_news_text(cls._get_news_field(item, "url"), limit=240),
        }
    
    def _inject_data_into_review(
        self,
        review: str,
        overview: MarketOverview,
        news: Optional[List] = None,
    ) -> str:
        """Inject structured data tables into the corresponding LLM prose sections."""
        # Build data blocks
        stats_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)
        patterns = (
            _ENGLISH_SECTION_PATTERNS
            if self._get_review_language() == "en"
            else _CHINESE_SECTION_PATTERNS
        )

        if stats_block:
            review = self._insert_after_section(
                review,
                patterns["market_summary"],
                stats_block,
            )

        if indices_block:
            review = self._insert_after_section(
                review,
                patterns["index_commentary"],
                indices_block,
            )

        if sector_block:
            original_review = review
            review = self._insert_after_section(
                review,
                patterns["sector_highlights"],
                sector_block,
            )
            if review == original_review and sector_block not in review:
                fallback_heading = (
                    "### 4. Sector Highlights"
                    if self._get_review_language() == "en"
                    else "### 三、板块主线"
                )
                review = f"{review.rstrip()}\n\n{fallback_heading}\n{sector_block}\n"

        return review

    @staticmethod
    def _insert_after_section(text: str, heading_pattern: str, block: str) -> str:
        """Insert a data block at the end of a markdown section (before the next ### heading)."""
        import re
        # Find the heading
        match = re.search(heading_pattern, text)
        if not match:
            return text
        start = match.end()
        # Find the next ### heading after this one
        next_heading = re.search(r'\n###\s', text[start:])
        if next_heading:
            insert_pos = start + next_heading.start()
        else:
            # No next heading — append at end
            insert_pos = len(text)
        # Insert the block before the next heading, with spacing
        return text[:insert_pos].rstrip() + '\n\n' + block + '\n\n' + text[insert_pos:].lstrip('\n')

    def _build_stats_block(self, overview: MarketOverview) -> str:
        """Build market statistics block."""
        if self.region == "us":
            return self._build_us_stats_block(overview)
        has_stats = overview.up_count or overview.down_count or overview.total_amount
        if not has_stats:
            return ""
        if self._get_review_language() == "en":
            light = self.build_market_light_snapshot(overview)
            return "\n".join(
                [
                    f"- **Market Signal**: {light['score']}/100 "
                    f"({light['temperature_label']}, {light['label']})",
                    f"- **Drivers**: {'; '.join(light['reasons'])}",
                    f"- **Guidance**: {light['guidance']}",
                    "",
                    f"- **Breadth**: Advancers {overview.up_count} / Decliners {overview.down_count} / "
                    f"Flat {overview.flat_count}; "
                    f"Limit-up {overview.limit_up_count} / Limit-down {overview.limit_down_count}; "
                    f"Turnover {overview.total_amount:.0f} ({self._get_turnover_unit_label()})",
                ]
            )
        light = self.build_market_light_snapshot(overview)
        score, label = light["score"], light["temperature_label"]
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else 0.0
        limit_spread = overview.limit_up_count - overview.limit_down_count
        lines = [
            f"- **盘面信号**：{score}/100（{label}，{light['label']}）",
            f"- **信号依据**：{'；'.join(light['reasons'])}",
            f"- **操作建议**：{light['guidance']}",
            "",
            "| 指标 | 数值 | 观察 |",
            "|------|------|------|",
            f"| 上涨/下跌/平盘 | {overview.up_count} / {overview.down_count} / {overview.flat_count} | 上涨占比(不含平盘) {up_ratio:.1%} |",
            f"| 涨停/跌停 | {overview.limit_up_count} / {overview.limit_down_count} | 涨跌停差 {limit_spread:+d} |",
            f"| 两市成交额 | {overview.total_amount:.0f} 亿 | {self._describe_turnover(overview.total_amount)} |",
        ]
        return "\n".join(lines)

    def _build_us_stats_block(self, overview: MarketOverview) -> str:
        """Build a source-aware US participation, liquidity, and macro dashboard."""
        context = overview.us_market_context or {}
        participation = context.get("participation") or {}
        proxies = context.get("proxies") or {}
        macro = context.get("macro") or {}
        if not context:
            return ""

        light = self.build_market_light_snapshot(overview)
        if self._get_review_language() == "en":
            lines = [
                f"- **Market Signal**: {light['score']}/100 "
                f"({light['temperature_label']}, {light['label']})",
                f"- **Drivers**: {'; '.join(light['reasons'])}",
                f"- **Guidance**: {light['guidance']}",
                "",
                "| Participation / liquidity proxy | Value | As of |",
                "|---|---:|---|",
            ]
        else:
            lines = [
                f"- **盘面信号**：{light['score']}/100（{light['temperature_label']}，{light['label']}）",
                f"- **信号依据**：{'；'.join(light['reasons'])}",
                f"- **操作建议**：{light['guidance']}",
                "",
                "#### 美股宽度与流动性等价指标",
                "| 指标 | 数值 | 数据日 |",
                "|---|---:|---|",
            ]

        for symbol in ("SPY", "RSP", "IWM", "QQQ"):
            item = proxies.get(symbol) or {}
            if not item:
                continue
            lines.append(
                f"| {item.get('name', symbol)} ({symbol}) | "
                f"{self._format_signed_pct(item.get('change_pct'))} | "
                f"{item.get('as_of', '-')} |"
            )

        relative_rows = [
            (
                "RSP相对SPY（等权-市值加权）",
                participation.get("equal_weight_vs_cap_weight_pct"),
            ),
            (
                "IWM相对SPY（小盘-大盘）",
                participation.get("small_cap_vs_large_cap_pct"),
            ),
            (
                "QQQ相对SPY（科技成长-大盘）",
                participation.get("nasdaq100_vs_large_cap_pct"),
            ),
        ]
        for label, value in relative_rows:
            lines.append(
                f"| {label} | {self._format_signed_pct(value)} | "
                f"{participation.get('as_of', '-')} |"
            )
        lines.append(
            "| 11个标普行业ETF上涨/下跌/平盘 | "
            f"{participation.get('sector_advancers', 0)} / "
            f"{participation.get('sector_decliners', 0)} / "
            f"{participation.get('sector_flat', 0)} | "
            f"{participation.get('as_of', '-')} |"
        )
        volume_ratio = participation.get("spy_volume_ratio_20d")
        volume_text = (
            f"{float(volume_ratio):.2f}x"
            if volume_ratio is not None
            else "N/A"
        )
        lines.append(
            f"| SPY成交量/前20日均量 | {volume_text} | "
            f"{participation.get('as_of', '-')} |"
        )

        if macro:
            source_labels = []
            for item in macro.values():
                source_name = str(item.get("source") or "")
                if source_name == "Federal Reserve Bank of St. Louis (FRED)":
                    label = "FRED"
                elif source_name == "U.S. Department of the Treasury":
                    label = "美国财政部"
                else:
                    label = source_name or "未标注"
                if label not in source_labels:
                    source_labels.append(label)
            source_summary = " / ".join(source_labels)
            macro_heading = (
                f"#### 宏观定价锚（官方来源：{source_summary}）"
                if self._get_review_language() != "en"
                else f"#### Macro Pricing Anchors (official: {source_summary})"
            )
            lines.extend(
                [
                    "",
                    macro_heading,
                    "| 指标 | 最新 | 日变化 | 数据日 | 来源 |",
                    "|---|---:|---:|---|---|",
                ]
            )
            for series_id in ("DGS2", "DGS10", "DTWEXBGS"):
                item = macro.get(series_id) or {}
                if not item:
                    continue
                unit = str(item.get("unit") or "")
                value = float(item.get("value") or 0.0)
                change = float(item.get("change") or 0.0)
                if unit == "%":
                    value_text = f"{value:.2f}%"
                    change_text = f"{change * 100:+.2f}bp"
                else:
                    value_text = f"{value:.2f}"
                    change_text = f"{change:+.2f}"
                source_name = str(item.get("source") or "未标注")
                lines.append(
                    f"| {item.get('name', series_id)} | {value_text} | "
                    f"{change_text} | {item.get('as_of', '-')} | {source_name} |"
                )

        note = (
            "> 口径说明：美股没有A股式涨跌停和北向资金。这里用SPY/RSP/IWM/QQQ相对表现、"
            "11个标普行业ETF和SPY量比衡量参与度与流动性；所有字段都保留来源和数据日，"
            "不得表述为交易所涨跌家数。"
        )
        if self._get_review_language() == "en":
            note = (
                "> Method: participation is measured with transparent SPY/RSP/IWM/QQQ "
                "relative-performance proxies, 11 S&P sector ETFs, and SPY volume ratio. "
                "These are not exchange advance/decline counts."
            )
        lines.extend(["", note])
        return "\n".join(lines)

    def build_market_light_snapshot(self, overview: MarketOverview) -> Dict[str, Any]:
        """Build a deterministic market-light snapshot from structured breadth data."""
        quality = self._assess_market_data_quality(overview)
        if not quality["core_data_ready"]:
            index_available = bool(quality["indices_available"])
            if self._get_review_language() == "en":
                label = "data unavailable"
                temperature_label = "unavailable"
                reasons = [
                    "core market data validation failed: "
                    + ", ".join(quality["missing_core_fields"])
                ]
                guidance = (
                    "Pause directional and position-sizing judgments until "
                    "the market data source recovers."
                )
            else:
                label = "数据不足"
                temperature_label = "不可用"
                reasons = [
                    "核心行情校验未通过："
                    + "、".join(quality["missing_core_fields"])
                ]
                guidance = "暂停方向和仓位判断，等待行情数据源恢复。"
            return MarketLightSnapshot(
                region=self.region,
                trade_date=overview.date,
                status="yellow",
                label=label,
                score=50,
                temperature_label=temperature_label,
                reasons=reasons,
                guidance=guidance,
                dimensions={
                    "breadth": {"score": 50, "available": False},
                    "index": {"score": 50, "available": index_available},
                    "limit": {"score": 50, "available": False},
                },
                data_quality="unavailable",
            ).model_dump()

        scores = self._build_market_light_scores(overview)
        score = int(scores["score"])
        temperature_label = str(scores["temperature_label"])
        if score >= 60:
            status = "green"
        elif score >= 40:
            status = "yellow"
        else:
            status = "red"

        if self._get_review_language() == "en":
            label_map = {
                "green": "risk-on",
                "yellow": "balanced",
                "red": "risk-off",
            }
            guidance_map = {
                "green": "Risk appetite is acceptable; focus on leading themes and position discipline.",
                "yellow": "Signals are mixed; keep position sizing moderate and wait for confirmation.",
                "red": "Risk is elevated; prioritize drawdown control and avoid chasing weak rebounds.",
            }
            reasons = self._build_market_light_reasons_en(overview, score)
        else:
            label_map = {
                "green": "可进攻",
                "yellow": "需观察",
                "red": "偏防守",
            }
            guidance_map = {
                "green": "风险偏好尚可，关注主线延续与仓位纪律。",
                "yellow": "信号分化，控制仓位并等待量价确认。",
                "red": "风险偏高，优先控制回撤，避免追高弱反弹。",
            }
            reasons = self._build_market_light_reasons_zh(overview, score)

        snapshot = MarketLightSnapshot(
            region=self.region,
            trade_date=overview.date,
            status=status,
            label=label_map[status],
            score=score,
            temperature_label=temperature_label,
            reasons=reasons,
            guidance=guidance_map[status],
            dimensions=scores["dimensions"],
            data_quality=str(scores["data_quality"]),
        )
        return snapshot.model_dump()

    def _build_market_light_reasons_zh(self, overview: MarketOverview, score: int) -> List[str]:
        if self.region == "us":
            context = overview.us_market_context or {}
            participation = context.get("participation") or {}
            macro = context.get("macro") or {}
            reasons = [
                "11个标普行业ETF上涨/下跌 "
                f"{participation.get('sector_advancers', 0)}/"
                f"{participation.get('sector_decliners', 0)}",
                "等权相对市值加权 "
                f"{self._format_signed_pct(participation.get('equal_weight_vs_cap_weight_pct'))}",
                "小盘相对大盘 "
                f"{self._format_signed_pct(participation.get('small_cap_vs_large_cap_pct'))}",
            ]
            dgs2 = macro.get("DGS2") or {}
            if dgs2:
                reasons.append(
                    f"2年期美债 {float(dgs2.get('value') or 0):.2f}%"
                    f"（日变动 {float(dgs2.get('change') or 0) * 100:+.1f}bp）"
                )
            return reasons[:4]
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else None
        reasons: List[str] = []
        if up_ratio is not None:
            if up_ratio >= 0.6:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，赚钱效应扩散")
            elif up_ratio <= 0.4:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，亏钱效应较强")
            else:
                reasons.append(f"上涨家数占比 {up_ratio:.0%}，市场分化")
        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        if index_changes:
            avg_change = sum(index_changes) / len(index_changes)
            reasons.append(f"主要指数平均涨跌幅 {avg_change:+.2f}%")
        if self._is_positive_number(overview.previous_total_amount):
            direction = "放量" if overview.turnover_change > 0 else "缩量"
            reasons.append(
                f"两市成交额较前一交易日{direction} "
                f"{abs(overview.turnover_change):.0f} 亿"
                f"（{overview.turnover_change_pct:+.2f}%）"
            )
        if overview.limit_up_count or overview.limit_down_count:
            reasons.append(f"涨跌停差 {overview.limit_up_count - overview.limit_down_count:+d}")
        if not reasons and overview.total_amount:
            reasons.append(f"成交额 {overview.total_amount:.0f} 亿，{self._describe_turnover(overview.total_amount)}")
        if not reasons:
            reasons.append("结构化涨跌数据有限，按可用行情综合判断")
        return reasons[:4]

    def _build_market_light_reasons_en(self, overview: MarketOverview, score: int) -> List[str]:
        if self.region == "us":
            context = overview.us_market_context or {}
            participation = context.get("participation") or {}
            macro = context.get("macro") or {}
            reasons = [
                "S&P sector ETFs advancing/declining "
                f"{participation.get('sector_advancers', 0)}/"
                f"{participation.get('sector_decliners', 0)}",
                "equal weight vs cap weight "
                f"{self._format_signed_pct(participation.get('equal_weight_vs_cap_weight_pct'))}",
                "small cap vs large cap "
                f"{self._format_signed_pct(participation.get('small_cap_vs_large_cap_pct'))}",
            ]
            dgs2 = macro.get("DGS2") or {}
            if dgs2:
                reasons.append(
                    f"2Y Treasury {float(dgs2.get('value') or 0):.2f}% "
                    f"({float(dgs2.get('change') or 0) * 100:+.1f}bp daily)"
                )
            return reasons[:4]
        participation = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participation if participation else None
        reasons: List[str] = []
        if up_ratio is not None:
            if up_ratio >= 0.6:
                reasons.append(f"advancers ratio {up_ratio:.0%}, breadth is expanding")
            elif up_ratio <= 0.4:
                reasons.append(f"advancers ratio {up_ratio:.0%}, downside pressure dominates")
            else:
                reasons.append(f"advancers ratio {up_ratio:.0%}, breadth is mixed")
        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        if index_changes:
            avg_change = sum(index_changes) / len(index_changes)
            reasons.append(f"average major-index change {avg_change:+.2f}%")
        if overview.limit_up_count or overview.limit_down_count:
            reasons.append(f"limit-up/down spread {overview.limit_up_count - overview.limit_down_count:+d}")
        if not reasons and overview.total_amount:
            reasons.append(f"turnover {overview.total_amount:.0f} ({self._get_turnover_unit_label()})")
        if not reasons:
            reasons.append("limited structured breadth data; using available market inputs")
        return reasons[:4]

    def _build_indices_block(self, overview: MarketOverview) -> str:
        """Build a mobile-readable index summary plus OHLC audit lines."""
        if not overview.indices:
            return ""
        if self.region == "us":
            if self._get_review_language() == "en":
                lines = [
                    "| Index | Close | Prior close | Change % | Trade date |",
                    "|---|---:|---:|---:|---|",
                ]
                detail_heading = "**Session OHLC audit**"
            else:
                lines = [
                    "| 指数 | 收盘 | 昨收 | 涨跌幅 | 交易日 |",
                    "|---|---:|---:|---:|---|",
                ]
                detail_heading = "**日内 OHLC 校验**"
            details: List[str] = []
            for idx in overview.indices:
                arrow = self._get_index_change_arrow(idx.change_pct)
                lines.append(
                    f"| {idx.name} | {idx.current:.2f} | "
                    f"{self._format_optional_number(idx.prev_close)} | "
                    f"{arrow} {idx.change_pct:+.2f}% | "
                    f"{idx.trade_date or '-'} |"
                )
                details.append(
                    f"- **{idx.name}**：开 {self._format_optional_number(idx.open)}｜"
                    f"高 {self._format_optional_number(idx.high)}｜"
                    f"低 {self._format_optional_number(idx.low)}｜"
                    f"振幅 {self._format_optional_pct(idx.amplitude)}"
                )
            return "\n".join([*lines, "", detail_heading, *details])
        if self._get_review_language() == "en":
            lines = [
                f"| Index | Close | Prior close | Change % | Turnover ({self._get_turnover_unit_label()}) |",
                "|---|---:|---:|---:|---:|",
            ]
            detail_heading = "**Session OHLC audit**"
        else:
            lines = [
                "| 指数 | 收盘 | 昨收 | 涨跌幅 | 成交额(亿) |",
                "|---|---:|---:|---:|---:|",
            ]
            detail_heading = "**日内 OHLC 校验**"
        details = []
        for idx in overview.indices:
            arrow = self._get_index_change_arrow(idx.change_pct)
            amount_raw = idx.amount or 0.0
            amount_str = self._format_turnover_value(amount_raw)
            lines.append(
                f"| {idx.name} | {idx.current:.2f} | "
                f"{self._format_optional_number(idx.prev_close)} | "
                f"{arrow} {idx.change_pct:+.2f}% | {amount_str} |"
            )
            details.append(
                f"- **{idx.name}**：开 {self._format_optional_number(idx.open)}｜"
                f"高 {self._format_optional_number(idx.high)}｜"
                f"低 {self._format_optional_number(idx.low)}｜"
                f"振幅 {self._format_optional_pct(idx.amplitude)}｜"
                f"交易日 {idx.trade_date or '-'}"
            )
        return "\n".join([*lines, "", detail_heading, *details])

    def _build_sector_block(self, overview: MarketOverview) -> str:
        """Build industry and concept ranking blocks."""
        if (
            not overview.top_sectors
            and not overview.bottom_sectors
            and not overview.top_concepts
            and not overview.bottom_concepts
        ):
            return ""
        lines = []
        language = self._get_review_language()

        def append_ranking(title: str, name_label: str, rows: List[Dict]) -> None:
            if not rows:
                return
            if lines:
                lines.append("")
            lines.extend([
                title,
                f"| {'Rank' if language == 'en' else '排名'} | {name_label} | {'Change' if language == 'en' else '涨跌幅'} |",
                "|------|------|--------|",
            ])
            for rank, item in enumerate(rows[:5], 1):
                lines.append(
                    f"| {rank} | {item.get('name', '-')} | {self._format_signed_pct(item.get('change_pct'))} |"
                )

        if language == "en":
            leading_title = (
                "#### Leading S&P Sector ETFs"
                if self.region == "us"
                else "#### Leading Industry Sectors"
            )
            lagging_title = (
                "#### Lagging S&P Sector ETFs"
                if self.region == "us"
                else "#### Lagging Industry Sectors"
            )
            append_ranking(leading_title, "Sector", overview.top_sectors)
            append_ranking(lagging_title, "Sector", overview.bottom_sectors)
            append_ranking("#### Leading Concept Themes", "Concept", overview.top_concepts)
            append_ranking("#### Lagging Concept Themes", "Concept", overview.bottom_concepts)
        else:
            uses_sw1_proxy = self.region == "cn" and any(
                str(item.get("classification") or "").upper() == "SW1_PROXY"
                for item in list(overview.top_sectors or []) + list(overview.bottom_sectors or [])
                if isinstance(item, dict)
            )
            cn_sector_label = (
                "申万一级行业（成分股流通市值加权代理）"
                if uses_sw1_proxy
                else "申万一级行业"
            )
            leading_title = (
                "#### 标普行业ETF领涨 Top 5"
                if self.region == "us"
                else f"#### {cn_sector_label}领涨 Top 5"
            )
            lagging_title = (
                "#### 标普行业ETF领跌 Top 5"
                if self.region == "us"
                else f"#### {cn_sector_label}领跌 Top 5"
            )
            append_ranking(leading_title, cn_sector_label, overview.top_sectors)
            append_ranking(lagging_title, cn_sector_label, overview.bottom_sectors)
            if uses_sw1_proxy:
                proxy_row = next(
                    item
                    for item in list(overview.top_sectors or []) + list(overview.bottom_sectors or [])
                    if isinstance(item, dict)
                    and str(item.get("classification") or "").upper() == "SW1_PROXY"
                )
                lines.extend([
                    "",
                    "- **口径说明**：申万官方指数接口本轮不可用；以上为新浪公开的完整31个"
                    "申万2021一级行业成分股，按前一交易日流通市值加权的当日收益代理排名，"
                    f"数据日 {proxy_row.get('quote_date', '-')}。代理值不冒充申万官方指数涨跌幅。",
                ])
            append_ranking("#### 概念板块领涨 Top 5", "概念板块", overview.top_concepts)
            append_ranking("#### 概念板块领跌 Top 5", "概念板块", overview.bottom_concepts)
        return "\n".join(lines)

    def _build_news_block(self, news: List) -> str:
        """Build a compact source-aware news catalyst list for the rendered report."""
        if not news:
            return ""
        language = self._get_review_language()
        if language == "en":
            lines = [
                "#### News Catalysts",
            ]
        else:
            lines = [
                "#### 近三日市场线索",
            ]

        for idx, item in enumerate(news[:5], 1):
            lines.append(self._format_news_catalyst_line(idx, item, language=language))
        return "\n".join(lines)

    @staticmethod
    def _get_news_field(item: Any, field: str) -> str:
        if hasattr(item, field):
            value = getattr(item, field, "") or ""
        elif isinstance(item, dict):
            value = item.get(field, "") or ""
        else:
            value = ""
        return str(value).strip()

    @classmethod
    def _format_news_catalyst_line(cls, idx: int, item: Any, *, language: str = "zh") -> str:
        fallback_title = "Untitled catalyst" if language == "en" else "未命名线索"
        title = cls._compact_news_text(cls._get_news_field(item, "title"), limit=90) or fallback_title
        source = cls._compact_news_text(cls._get_news_field(item, "source"), limit=40)
        date_text = cls._compact_news_text(cls._get_news_field(item, "published_date"), limit=24)
        url = cls._compact_news_text(cls._get_news_field(item, "url"), limit=0)
        title_text = cls._escape_markdown_link_label(title)
        if url:
            title_text = f"[{title_text}]({url})"
        meta_parts = [part for part in (source, date_text) if part]
        if language == "en":
            meta = f" ({' / '.join(meta_parts)})" if meta_parts else ""
        else:
            meta = f"（{' / '.join(meta_parts)}）" if meta_parts else ""
        return f"- {idx}. {title_text}{meta}"

    @staticmethod
    def _compact_news_text(value: str, *, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if limit <= 0 or len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    @staticmethod
    def _format_optional_number(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}"

    @staticmethod
    def _format_optional_pct(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}%"

    @staticmethod
    def _format_signed_pct(value: Any) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{numeric_value:+.2f}%"

    @classmethod
    def _format_ranking_summary(cls, rows: List[Dict], limit: int = 3) -> str:
        parts = []
        for item in (rows or [])[:limit]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            parts.append(f"{name}({cls._format_signed_pct(item.get('change_pct'))})")
        return ", ".join(parts)

    @staticmethod
    def _escape_markdown_link_label(value: str) -> str:
        return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")

    @staticmethod
    def _describe_turnover(total_amount: float) -> str:
        if total_amount >= 15000:
            return "高活跃度"
        if total_amount >= 9000:
            return "中等活跃"
        if total_amount > 0:
            return "缩量观望"
        return "暂无数据"

    def _build_market_light_scores(self, overview: MarketOverview) -> Dict[str, Any]:
        """Build the canonical Market Light scores used by reports and alerts."""
        if self.region == "us":
            return self._build_us_market_light_scores(overview)

        participants = overview.up_count + overview.down_count
        breadth_available = bool(self.profile.has_market_stats and participants > 0)
        breadth_score = 50
        if breadth_available:
            breadth_score = int(overview.up_count / participants * 100)

        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        index_available = bool(overview.indices and index_changes)
        index_score = 50
        if index_available:
            avg_change = sum(index_changes) / len(index_changes)
            index_score = int(max(0, min(100, 50 + avg_change * 12)))

        limit_total = overview.limit_up_count + overview.limit_down_count
        limit_available = bool(self.profile.has_market_stats and limit_total > 0)
        limit_score = 50
        if limit_available:
            # 涨停数容易被小市值题材股放大，单项不允许
            # 因“几乎全是涨停”而接近满分。
            limit_score = min(
                85,
                int(overview.limit_up_count / limit_total * 100),
            )

        liquidity_available = bool(
            self._is_positive_number(overview.total_amount)
            and self._is_positive_number(overview.previous_total_amount)
        )
        liquidity_score = 50
        if liquidity_available:
            liquidity_score = int(max(
                0,
                min(100, 50 + float(overview.turnover_change_pct) * 2.0),
            ))

        structure_available = len(index_changes) >= 4
        structure_score = 50
        if structure_available:
            dispersion = max(index_changes) - min(index_changes)
            avg_change = sum(index_changes) / len(index_changes)
            structure_score = int(max(
                0,
                min(100, 50 + avg_change * 8.0 - dispersion * 12.0),
            ))

        dimensions = {
            "breadth": {"score": breadth_score, "available": breadth_available},
            "index": {"score": index_score, "available": index_available},
            "limit": {"score": limit_score, "available": limit_available},
            "liquidity": {"score": liquidity_score, "available": liquidity_available},
            "structure": {"score": structure_score, "available": structure_available},
        }

        if not index_available:
            data_quality = "unavailable"
        elif all(dimension["available"] for dimension in dimensions.values()):
            data_quality = "ok"
        else:
            data_quality = "partial"

        score = int(round(
            breadth_score * 0.30
            + index_score * 0.25
            + liquidity_score * 0.20
            + structure_score * 0.15
            + limit_score * 0.10
        ))
        if self._get_review_language() == "en":
            if score >= 70:
                label = "risk-on"
            elif score >= 55:
                label = "constructive"
            elif score >= 40:
                label = "mixed"
            else:
                label = "defensive"
        else:
            if score >= 70:
                label = "强势"
            elif score >= 55:
                label = "偏暖"
            elif score >= 40:
                label = "震荡"
            else:
                label = "偏弱"
        return {
            "score": score,
            "temperature_label": label,
            "dimensions": dimensions,
            "data_quality": data_quality,
        }

    def _build_us_market_light_scores(self, overview: MarketOverview) -> Dict[str, Any]:
        """Score US risk posture from deterministic, printed market inputs."""
        context = overview.us_market_context or {}
        participation = context.get("participation") or {}
        macro = context.get("macro") or {}

        def clamp(value: float) -> int:
            return int(round(max(0.0, min(100.0, value))))

        sector_coverage = int(participation.get("sector_coverage") or 0)
        sector_advancers = int(participation.get("sector_advancers") or 0)
        sector_ratio = (
            sector_advancers / sector_coverage
            if sector_coverage
            else 0.5
        )
        equal_weight_relative = float(
            participation.get("equal_weight_vs_cap_weight_pct") or 0.0
        )
        small_cap_relative = float(
            participation.get("small_cap_vs_large_cap_pct") or 0.0
        )
        participation_score = clamp(
            sector_ratio * 50.0
            + (50.0 + equal_weight_relative * 20.0) * 0.25
            + (50.0 + small_cap_relative * 15.0) * 0.25
        )

        core_changes = [
            float(index.change_pct)
            for index in overview.indices
            if index.code != "VIX" and index.change_pct is not None
        ]
        index_score = clamp(
            50.0 + (sum(core_changes) / len(core_changes) * 12.0)
            if core_changes else 50.0
        )

        volume_ratio = participation.get("spy_volume_ratio_20d")
        liquidity_score = clamp(
            50.0 + (float(volume_ratio) - 1.0) * 50.0
            if volume_ratio is not None else 50.0
        )

        vix = next((index for index in overview.indices if index.code == "VIX"), None)
        volatility_score = 50
        if vix and self._is_positive_number(vix.current):
            if vix.current <= 15:
                volatility_score = 72
            elif vix.current <= 20:
                volatility_score = 62
            elif vix.current <= 25:
                volatility_score = 45
            elif vix.current <= 30:
                volatility_score = 32
            else:
                volatility_score = 18
            volatility_score = clamp(
                volatility_score - float(vix.change_pct or 0.0) * 1.2
            )

        dgs2_change_bp = float((macro.get("DGS2") or {}).get("change") or 0.0) * 100.0
        dgs10_change_bp = float((macro.get("DGS10") or {}).get("change") or 0.0) * 100.0
        dollar_change = float((macro.get("DTWEXBGS") or {}).get("change") or 0.0)
        macro_score = clamp(
            50.0
            - dgs2_change_bp * 0.8
            - dgs10_change_bp * 0.5
            - dollar_change * 2.0
        )

        participation_dimension = {
            "score": participation_score,
            "available": sector_coverage >= 8,
        }
        volatility_dimension = {
            "score": volatility_score,
            "available": vix is not None,
        }
        dimensions = {
            # Keep the canonical fields consumed by existing alert rules.
            "breadth": dict(participation_dimension),
            "index": {"score": index_score, "available": bool(core_changes)},
            "limit": dict(volatility_dimension),
            # Preserve the US-native dimensions for reports/API consumers.
            "participation": participation_dimension,
            "liquidity": {"score": liquidity_score, "available": volume_ratio is not None},
            "volatility": volatility_dimension,
            "macro": {
                "score": macro_score,
                "available": "DGS2" in macro and "DGS10" in macro,
            },
        }
        score = int(round(
            participation_score * 0.30
            + index_score * 0.25
            + liquidity_score * 0.15
            + volatility_score * 0.15
            + macro_score * 0.15
        ))
        if self._get_review_language() == "en":
            label = (
                "risk-on" if score >= 70
                else "constructive" if score >= 55
                else "mixed" if score >= 40
                else "defensive"
            )
        else:
            label = (
                "强势" if score >= 70
                else "偏暖" if score >= 55
                else "震荡" if score >= 40
                else "偏弱"
            )
        return {
            "score": score,
            "temperature_label": label,
            "dimensions": dimensions,
            "data_quality": (
                "ok"
                if all(
                    dimensions[key]["available"]
                    for key in ("participation", "index", "liquidity", "volatility", "macro")
                )
                else "partial"
            ),
        }

    def _build_market_temperature(self, overview: MarketOverview) -> tuple[int, str]:
        scores = self._build_market_light_scores(overview)
        score = int(scores["score"])
        label = str(scores["temperature_label"])
        return score, label

    def _build_output_template_sections(
        self,
        review_language: str,
        *,
        market_stats_available: Optional[bool] = None,
        sector_rankings_available: Optional[bool] = None,
    ) -> str:
        """Build LLM output sections according to market data capabilities."""
        has_market_stats = (
            self.profile.has_market_stats
            if market_stats_available is None
            else market_stats_available
        )
        has_sector_rankings = (
            self.profile.has_sector_rankings
            if sector_rankings_available is None
            else sector_rankings_available
        )
        if self.region == "us":
            if review_language == "en":
                return """### 3. Participation & Liquidity
(Interpret only SPY/RSP/IWM/QQQ relative performance, sector-ETF coverage, SPY volume ratio, and VIX.)

### 4. Sector Rotation
(Explain the verified 11-sector ETF leaders and laggards; distinguish broad participation from narrow mega-cap leadership.)

### 5. Macro, Policy & Earnings
(Connect verified Treasury/USD observations and source-labelled Fed, fiscal, regulatory, political, geopolitical, earnings, and guidance news. Separate facts from inference.)

### 6. Next-session Quant Plan / Strategy Plan
(Give an offensive/balanced/defensive posture, an exposure band, confirmation conditions, and one invalidation condition derived only from supplied data.)

### 7. Data Boundary & Risks
(List missing or lagged inputs and end with "For reference only, not investment advice.")"""
            return """### 三、参与度与流动性
（只解读RSP/SPY、IWM/SPY、QQQ/SPY、11个行业ETF上涨覆盖、SPY量比与VIX，不写成交易所涨跌家数）

### 四、行业轮动
（分析已提供的11个标普行业ETF领涨/领跌，区分全面扩散与少数巨头拉动）

### 五、宏观、政策与财报
（结合已校验的美债/美元和带来源新闻，覆盖联储、财政、监管、关税、政治/地缘以及财报与管理层指引；明确区分事实与推断）

### 六、下一交易日量化计划
（给出进攻/均衡/防守、仓位区间、确认条件与一个失效条件；只能使用输入数据）

### 七、数据边界与风险
（逐项列出缺失或滞后数据；最后补充“建议仅供参考，不构成投资建议”。）"""
        if review_language == "en":
            if has_market_stats and has_sector_rankings:
                return """### 3. Fund Flows
(Interpret what turnover, participation, and flow signals imply.)

### 4. Sector Highlights
(Distinguish industry-sector moves from concept/theme moves, then analyze drivers and persistence.)

### 5. Outlook
(Provide the near-term outlook based on price action and news.)

### 6. Risk Alerts
(List the main risks to monitor.)

### 7. Strategy Plan
(Provide an offensive/balanced/defensive stance, a position-sizing guideline, one invalidation trigger, and end with "For reference only, not investment advice.")"""

            section_number = 3
            sections: List[str] = []
            if has_market_stats:
                sections.append(f"""### {section_number}. Fund Flows
(Interpret only the provided turnover, participation, breadth, and flow signals.)""")
                section_number += 1
            if has_sector_rankings:
                sections.append(f"""### {section_number}. Sector Highlights
(Analyze only the provided industry-sector and concept/theme rankings.)""")
                section_number += 1
            sections.extend([
                f"""### {section_number}. News Catalysts
(Connect recent news to index price action and macro/external-market clues. Do not infer unsupported breadth, fund-flow, or sector-ranking data.)""",
                f"""### {section_number + 1}. Outlook
(Provide the near-term outlook based on index price action and the available news.)""",
                f"""### {section_number + 2}. Risk Alerts
(List the main risks to monitor.)""",
                f"""### {section_number + 3}. Strategy Plan
(Provide an offensive/balanced/defensive stance, a position-sizing guideline, one invalidation trigger, and end with "For reference only, not investment advice.")""",
            ])
            return "\n\n".join(sections)

        if has_market_stats and has_sector_rankings:
            return """### 三、板块主线
（区分行业板块与概念题材，分析领涨/领跌背后的逻辑、持续性和是否形成主线）

### 四、资金与情绪
（解读成交额、涨跌停结构、市场宽度和风险偏好）

### 五、消息催化
（结合近三日新闻，提炼真正影响明日交易的催化或扰动）

### 六、明日交易计划
（给出进攻/均衡/防守结论、仓位区间、关注方向、回避方向和一个触发失效条件）

### 七、风险提示
（列出需要关注的风险点；最后补充“建议仅供参考，不构成投资建议”。）"""

        numerals = ["一", "二", "三", "四", "五", "六", "七", "八"]
        section_number = 3
        sections: List[str] = []

        def add_section(title: str, hint: str) -> None:
            nonlocal section_number
            sections.append(f"### {numerals[section_number - 1]}、{title}\n{hint}")
            section_number += 1

        if has_sector_rankings:
            add_section("板块主线", "（仅分析已提供的行业板块与概念题材榜单，不扩展未提供的数据）")
        if has_market_stats:
            add_section("资金与情绪", "（仅解读已提供的成交额、涨跌停结构、市场宽度和风险偏好数据）")
        add_section(
            "消息催化",
            "（结合近三日新闻和指数表现，提炼真正影响明日交易的催化或扰动；不要推断未提供的资金流、市场宽度或板块榜）",
        )
        add_section("明日交易计划", "（给出进攻/均衡/防守结论、仓位区间、关注方向、回避方向和一个触发失效条件）")
        add_section("风险提示", "（列出需要关注的风险点；最后补充“建议仅供参考，不构成投资建议”。）")
        return "\n\n".join(sections)

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """构建复盘报告 Prompt"""
        review_language = self._get_review_language()
        quality = self._assess_market_data_quality(overview)
        market_stats_available = bool(
            self.profile.has_market_stats
            and quality["breadth_available"]
            and quality["turnover_available"]
        )
        sector_rankings_available = bool(
            self.profile.has_sector_rankings
            and quality["sector_rankings_available"]
        )
        # Korean reuses the English structural template but the model is told to
        # write the entire shell, headings, guidance and conclusion in Korean.
        shell_language_label = "Korean (한국어)" if self._get_output_language() == "ko" else "English"

        # 指数行情信息（简洁格式，不用emoji）
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息
        top_sectors_text = self._format_ranking_summary(overview.top_sectors)
        bottom_sectors_text = self._format_ranking_summary(overview.bottom_sectors)
        top_concepts_text = self._format_ranking_summary(overview.top_concepts)
        bottom_concepts_text = self._format_ranking_summary(overview.bottom_concepts)
        
        # 新闻信息 - 支持 SearchResult 对象或字典
        news_text = ""
        for i, n in enumerate(news[:10], 1):
            # 兼容 SearchResult 对象和字典
            title = self._compact_news_text(self._get_news_field(n, "title"), limit=90)
            snippet = self._compact_news_text(self._get_news_field(n, "snippet"), limit=220)
            source = self._compact_news_text(self._get_news_field(n, "source"), limit=60)
            published_date = self._compact_news_text(self._get_news_field(n, "published_date"), limit=30)
            url = self._compact_news_text(self._get_news_field(n, "url"), limit=180)
            meta_parts = [part for part in (source, published_date) if part]
            meta = f" ({' / '.join(meta_parts)})" if meta_parts else ""
            url_line = f"\n   URL: {url}" if url else ""
            news_text += f"{i}. {title}{meta}\n   {snippet or '-'}{url_line}\n"
        
        # 按 region 组装市场概况与板块区块（美股/港股/日韩无涨跌家数、板块数据）
        stats_block = ""
        sector_block = ""
        data_limits_block = ""
        if review_language == "en":
            if market_stats_available:
                stats_block = f"""## Market Breadth
- Advancers: {overview.up_count} | Decliners: {overview.down_count} | Flat: {overview.flat_count}
- Limit-up: {overview.limit_up_count} | Limit-down: {overview.limit_down_count}
- Turnover: {overview.total_amount:.0f} ({self._get_turnover_unit_label()})"""

            if self.region == "us" and market_stats_available:
                stats_block = "## US Participation, Liquidity & Macro\n" + self._build_us_stats_block(overview)

            if sector_rankings_available:
                sector_block = f"""## Sector / Theme Performance
Industry leading: {top_sectors_text if top_sectors_text else "N/A"}
Industry lagging: {bottom_sectors_text if bottom_sectors_text else "N/A"}
Concept leading: {top_concepts_text if top_concepts_text else "N/A"}
Concept lagging: {bottom_concepts_text if bottom_concepts_text else "N/A"}"""
                if self.region == "us":
                    sector_block = f"""## S&P Sector ETF Performance
Leading: {top_sectors_text if top_sectors_text else "N/A"}
Lagging: {bottom_sectors_text if bottom_sectors_text else "N/A"}"""

            data_limit_lines = []
            if self.profile.has_market_stats and not market_stats_available:
                data_limit_lines.append(
                    "- Market breadth or aggregate turnover is missing because the data provider failed. "
                    "Missing values are not zero and must not be interpreted as market inactivity."
                )
            elif not self.profile.has_market_stats:
                data_limit_lines.append(
                    "- Market breadth, aggregate turnover, participation, and fund-flow signals are not available for this market."
                )
            if self.profile.has_sector_rankings and not sector_rankings_available:
                data_limit_lines.append("- Sector/theme ranking data is missing for this run.")
            elif not self.profile.has_sector_rankings:
                data_limit_lines.append("- Sector/theme ranking data is not available for this market.")
            if data_limit_lines:
                data_limits_block = "## Data Limits\n" + "\n".join(data_limit_lines)
        else:
            if market_stats_available:
                stats_block = f"""## 市场概况
- 上涨: {overview.up_count} 家 | 下跌: {overview.down_count} 家 | 平盘: {overview.flat_count} 家
- 涨停: {overview.limit_up_count} 家 | 跌停: {overview.limit_down_count} 家
- 两市成交额: {overview.total_amount:.0f} 亿元"""

            if self.region == "us" and market_stats_available:
                stats_block = "## 美股参与度、流动性与宏观定价\n" + self._build_us_stats_block(overview)

            if sector_rankings_available:
                sector_block = f"""## 板块表现
行业领涨: {top_sectors_text if top_sectors_text else "暂无数据"}
行业领跌: {bottom_sectors_text if bottom_sectors_text else "暂无数据"}
概念领涨: {top_concepts_text if top_concepts_text else "暂无数据"}
概念领跌: {bottom_concepts_text if bottom_concepts_text else "暂无数据"}"""
                if self.region == "us":
                    sector_block = f"""## 标普行业ETF表现
行业领涨: {top_sectors_text if top_sectors_text else "暂无数据"}
行业领跌: {bottom_sectors_text if bottom_sectors_text else "暂无数据"}"""

            data_limit_lines = []
            if self.profile.has_market_stats and not market_stats_available:
                data_limit_lines.append(
                    "- 本次市场宽度或两市成交额因数据源失败而缺失；缺失不等于 0，"
                    "禁止解读为市场无成交、无涨跌或流动性冻结。"
                )
            elif not self.profile.has_market_stats:
                data_limit_lines.append("- 该市场暂无涨跌家数、涨跌停、成交额汇总、参与度或资金流信号。")
            if self.profile.has_sector_rankings and not sector_rankings_available:
                data_limit_lines.append("- 本次行业板块/概念题材涨跌榜缺失。")
            elif not self.profile.has_sector_rankings:
                data_limit_lines.append("- 该市场暂无行业板块/概念题材涨跌榜。")
            if data_limit_lines:
                data_limits_block = "## 数据边界\n" + "\n".join(data_limit_lines)

        data_no_indices_hint = (
            "注意：由于行情数据获取失败，请主要根据【市场新闻】进行定性分析和总结，不要编造具体的指数点位。"
            if not indices_text
            else ""
        )
        if review_language == "en":
            data_no_indices_hint = (
                "Note: Market data fetch failed. Rely mainly on [Market News] for qualitative analysis. Do not invent index levels."
                if not indices_text
                else ""
            )
            indices_placeholder = indices_text if indices_text else "No index data (API error)"
            news_placeholder = news_text if news_text else "No relevant news"
            data_boundary_requirement = (
                "- Respect Data Limits: do not invent or over-interpret unsupported breadth, fund-flow, turnover, participation, or sector-ranking data.\n"
                if data_limits_block
                else ""
            )
            if self.region == "us":
                data_boundary_requirement += (
                    "- Treat ETF breadth and volume as explicitly labelled proxies, never as exchange-wide counts.\n"
                    "- Separate observed facts, source-labelled news, and analytical inference. If policy or earnings evidence is absent, say it is unverified.\n"
                    "- Do not let headlines override contradictory price, participation, volatility, or rates data.\n"
                )
            market_summary_hint = (
                "2-3 sentences summarizing overall market tone, index moves, and liquidity."
                if market_stats_available
                else "2-3 sentences summarizing overall market tone, index moves, and available news context."
            )
        else:
            indices_placeholder = indices_text if indices_text else "暂无指数数据（接口异常）"
            news_placeholder = news_text if news_text else "暂无相关新闻"
            data_boundary_requirement = (
                "- 严格遵守数据边界：未提供涨跌家数、资金流、成交额汇总或板块榜时，不要编造或过度解读。\n"
                if data_limits_block
                else ""
            )
            if self.region == "us":
                data_boundary_requirement += (
                    "- ETF宽度与量比只能称为“等价代理指标”，禁止写成交易所全市场涨跌家数或真实资金净流入。\n"
                    "- 必须把“行情事实、带来源新闻、分析推断”分开；政策或财报证据未检索到时明确写“无法验证”。\n"
                    "- 新闻叙事与价格、参与度、VIX或利率矛盾时，不得用情绪化叙事覆盖数据。\n"
                    "- 仓位与方向结论只能来自已打印的确定性评分及其数据，不得自行创造目标位。\n"
                )
            market_summary_hint = (
                "2-3句话概括指数、涨跌家数、成交额和情绪温度，明确“强势/偏暖/震荡/偏弱”判断"
                if market_stats_available
                else "2-3句话概括指数表现、新闻线索和整体风险状态，不要补写未提供的市场宽度或资金流数据"
            )

        output_template_sections = self._build_output_template_sections(
            review_language,
            market_stats_available=market_stats_available,
            sector_rankings_available=sector_rankings_available,
        )
        zh_market_scope_name = self._get_market_scope_name("zh")
        zh_report_title = f"{overview.date} 大盘复盘"
        if self.region != "cn":
            zh_report_title = f"{overview.date} {zh_market_scope_name}大盘复盘"
        workflow_hint = (
            "报告要像交易员盘后工作台：先给结论，再按数据表、主线、催化、计划展开"
            if market_stats_available or sector_rankings_available
            else "报告要像交易员盘后工作台：先给结论，再按指数、新闻催化和计划展开"
        )

        if review_language == "en":
            report_title = self._get_review_title(overview.date).removeprefix("## ").strip()
            return f"""You are a professional {self._get_market_scope_name('en')} analyst. Please produce a concise market recap report based on the data below.

[Requirements]
- Output pure Markdown only
- No JSON
- No code blocks
- Use emoji sparingly in headings (at most one per heading)
- The entire fixed shell, headings, guidance, and conclusion must be in {shell_language_label}
{data_boundary_requirement}

---

# Today's Market Data

## Date
{overview.date}

## Major Indices
{indices_placeholder}

{stats_block}

{sector_block}

{data_limits_block}

## Market News
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

---

# Output Template (follow this structure)

## {report_title}

### 1. Market Summary
({market_summary_hint})

### 2. Index Commentary
({self._get_index_hint()})

{output_template_sections}

---

Output the report content directly, no extra commentary.
"""

        # A 股场景使用中文提示语
        return f"""你是一位专业的{self._get_market_scope_name('zh')}分析师，请根据以下数据生成一份结构化的{self._get_market_scope_name('zh')}大盘复盘报告。

【重要】输出要求：
- 必须输出纯 Markdown 文本格式
- 禁止输出 JSON 格式
- 禁止输出代码块
- emoji 仅在标题处少量使用（每个标题最多1个）
- {workflow_hint}
- 不要重复列出已由系统注入的表格数据；正文负责解释表格背后的含义
{data_boundary_requirement}

---

# 今日市场数据

## 日期
{overview.date}

## 主要指数
{indices_placeholder}

{stats_block}

{sector_block}

{data_limits_block}

## 市场新闻
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

---

# 输出格式模板（请严格按此格式输出）

## {zh_report_title}

> 一句话给出今日市场状态、核心矛盾和明日优先观察方向。

### 一、盘面总览
（{market_summary_hint}）

### 二、指数结构
（{self._get_index_hint()}，说明谁在护盘、谁在拖累，以及关键支撑/压力）

{output_template_sections}

---

请直接输出复盘报告内容，不要输出其他说明文字。
"""

    def _generate_data_unavailable_review(
        self,
        overview: MarketOverview,
        quality: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a deterministic fail-closed report for invalid core data."""
        quality = quality or self._assess_market_data_quality(overview)
        if self.region == "us":
            return self._generate_us_data_unavailable_review(overview, quality)
        indices_block = self._build_indices_block(overview)
        missing_fields = list(quality.get("missing_core_fields") or [])
        review_time = datetime.now().strftime("%H:%M")

        if self._get_review_language() == "en":
            labels = {
                "major_indices": "major indices",
                "market_breadth": "market breadth",
                "market_breadth_trade_date": "market-breadth / index trade-date alignment",
                "aggregate_turnover": "aggregate turnover",
                "prior_session_turnover": "prior-session turnover",
                "sw1_sector_rankings": "Shenwan level-1 sector rankings",
            }
            missing_text = ", ".join(labels.get(field, field) for field in missing_fields)
            turnover_status = (
                f"{overview.total_amount:.0f} {self._get_turnover_unit_label()} "
                "(recovered from primary index turnover)"
                if self._is_positive_number(overview.total_amount)
                else "unavailable"
            )
            return f"""## {overview.date} Market Recap — Data Validation Failed

> Core market data did not pass validation. This run does not produce a market-direction judgment, sentiment score, position-sizing range, or trading plan.

### 1. Validation Status
- **Missing core fields**: {missing_text or "unknown"}
- **Major indices**: {quality.get("valid_index_count", 0)} valid
- **Market breadth**: {"available" if quality.get("breadth_available") else "unavailable"}
- **Aggregate turnover**: {turnover_status}
- **Meaning**: missing values are API failures, not zero trading activity.

### 2. Available Index Data
{indices_block or "- No validated index data is available."}

### 3. Processing Decision
- LLM market interpretation was skipped.
- Directional, sector, sentiment, and position recommendations were suppressed.
- Retry after the market-data source recovers.

### 4. Risk Notice
- Do not treat this diagnostic report as a market state or investment signal.
- For reference only, not investment advice.

---
*Validation Time: {review_time}*
"""

        labels = {
            "major_indices": "主要指数",
            "market_breadth": "上涨/下跌/平盘家数",
            "market_breadth_trade_date": "市场宽度与指数交易日对齐",
            "aggregate_turnover": "两市成交额",
            "prior_session_turnover": "前一交易日可比成交额",
            "sw1_sector_rankings": "申万一级行业排名",
            "index_trade_date_alignment": "六大指数同一交易日",
        }
        missing_text = "、".join(labels.get(field, field) for field in missing_fields)
        turnover_status = (
            f"{overview.total_amount:.0f} 亿元（由上证指数与深证成指成交额恢复）"
            if self._is_positive_number(overview.total_amount)
            else "不可用"
        )
        return f"""## {overview.date} 大盘复盘（数据校验未通过）

> 核心行情数据未通过校验。本次不生成市场方向、情绪温度、仓位区间或交易计划。

### 一、数据状态
- **缺失核心字段**：{missing_text or "未知"}
- **主要指数**：{quality.get("valid_index_count", 0)} 个有效
- **市场宽度**：{"可用" if quality.get("breadth_available") else "不可用"}
- **两市成交额**：{turnover_status}
- **字段含义**：缺失来自行情接口失败，不代表市场成交额或涨跌家数为 0。

### 二、已取得的指数数据
{indices_block or "- 暂无通过校验的指数数据。"}

### 三、处理结论
- 已跳过大模型行情解读。
- 已禁止生成方向、板块、情绪和仓位建议。
- 待行情数据源恢复后重新执行。

### 四、风险提示
- 本报告仅用于提示数据异常，不应视为市场状态或投资信号。
- 建议仅供参考，不构成投资建议。

---
*校验时间: {review_time}*
"""

    def _generate_us_data_unavailable_review(
        self,
        overview: MarketOverview,
        quality: Dict[str, Any],
    ) -> str:
        """Fail closed when any required US-market evidence layer is invalid."""
        labels = {
            "major_indices": "至少3个同交易日主要指数",
            "us_trade_date_alignment": "指数与ETF交易日对齐",
            "us_participation_proxies": "SPY/RSP/IWM/QQQ参与度代理",
            "us_liquidity_proxy": "SPY相对20日均量",
            "us_sector_etfs": "11个标普行业ETF（全部同日有效）",
            "us_treasury_yields": "美国2年/10年期国债收益率",
            "us_index_proxy_consistency": "指数与对应ETF涨跌方向/幅度交叉校验",
        }
        missing = "、".join(
            labels.get(item, item)
            for item in quality.get("missing_core_fields", [])
        )
        indices_block = self._build_indices_block(overview)
        context = overview.us_market_context or {}
        context_date = context.get("as_of") or "不可用"
        review_time = datetime.now().strftime("%H:%M")
        return f"""## {overview.date} 美股大盘复盘（数据校验未通过）

> 核心数据没有同时通过“交易日对齐、指数、参与度、流动性、行业轮动、利率”六层校验。本次禁止生成方向、仓位和买卖计划。

### 一、失败项
- **缺失或无效**：{missing or "未知"}
- **有效主要指数**：{quality.get("valid_index_count", 0)} 个
- **ETF数据日**：{context_date}
- **处理规则**：缺失不等于0，旧数据不冒充当日数据，模型不补数。

### 二、已取得的指数数据
{indices_block or "- 暂无通过校验的指数数据。"}

### 三、处理结论
- 已跳过大模型方向解读和情绪叙事。
- 已禁止生成仓位区间、目标位、支撑/压力和行业推荐。
- 待数据源恢复且交易日重新对齐后再执行。

### 四、风险提示
- 本页只是数据异常告警，不是美股市场结论。
- 建议仅供参考，不构成投资建议。

---
*校验时间: {review_time}*
"""

    def _generate_strict_data_review(self, overview: MarketOverview, news: List) -> str:
        """Render a deterministic review without allowing an LLM to rewrite facts."""
        if self.region == "us":
            return self._generate_us_template_review(overview, news, strict=True)
        review_time = datetime.now().astimezone().isoformat(timespec="seconds")
        valid_indices = [
            index
            for index in overview.indices
            if self._is_positive_number(index.current)
        ]
        strict_overview = replace(overview, indices=valid_indices)
        indices_block = self._build_indices_block(strict_overview)
        sector_block = self._build_sector_block(overview)
        news_block = self._build_news_block(news)
        light = self.build_market_light_snapshot(strict_overview)
        participants = overview.up_count + overview.down_count
        up_ratio = overview.up_count / participants if participants else 0.0
        index_changes = [
            float(index.change_pct)
            for index in valid_indices
            if index.change_pct is not None
        ]
        average_change = (
            sum(index_changes) / len(index_changes)
            if index_changes
            else 0.0
        )
        turnover_direction = "放量" if overview.turnover_change > 0 else "缩量"
        turnover_comparison_zh = (
            f"较 {overview.turnover_trade_date or '前一交易日'}"
            f"{turnover_direction} {abs(overview.turnover_change):.0f} 亿"
            f"（{overview.turnover_change_pct:+.2f}%）"
        )
        turnover_comparison_en = (
            f"{overview.turnover_change:+.0f} ({overview.turnover_change_pct:+.2f}%) "
            f"vs {overview.turnover_trade_date or 'prior session'}"
        )
        index_sources = sorted({index.source for index in valid_indices if index.source})
        index_source_text = " / ".join(index_sources) or "未标注"
        fetched_times = sorted({index.fetched_at for index in valid_indices if index.fetched_at})
        fetched_at_text = fetched_times[-1] if fetched_times else review_time
        sector_sources = sorted({
            str(item.get("source") or "")
            for item in overview.top_sectors + overview.bottom_sectors
            if isinstance(item, dict) and item.get("source")
        })
        sector_source_text = " / ".join(sector_sources) or "未标注"
        score = int(light["score"])
        if score >= 75:
            position_range = "60%-80%"
            posture_zh = "进攻"
            posture_en = "offensive"
        elif score >= 60:
            position_range = "40%-60%"
            posture_zh = "偏进攻"
            posture_en = "constructive"
        elif score >= 45:
            position_range = "20%-40%"
            posture_zh = "均衡"
            posture_en = "balanced"
        else:
            position_range = "0%-20%"
            posture_zh = "防守"
            posture_en = "defensive"

        mood_code = self.profile.mood_index_code
        reference_index = next(
            (
                index
                for index in valid_indices
                if index.code == mood_code or index.code.endswith(mood_code)
            ),
            valid_indices[0] if valid_indices else None,
        )
        reference_levels_valid = bool(
            reference_index
            and self._is_positive_number(reference_index.high)
            and self._is_positive_number(reference_index.low)
            and float(reference_index.low)
            <= float(reference_index.current)
            <= float(reference_index.high)
        )
        if reference_levels_valid:
            add_trigger_en = (
                f"A 30-minute close above {reference_index.name} "
                f"{reference_index.current:.2f}, with the live advancer ratio at or above 60% "
                "and projected turnover no lower than the prior session."
            )
            reduce_trigger_en = (
                f"A 30-minute close below {reference_index.name} "
                f"{reference_index.low:.2f}, or the live advancer ratio at or below 40%."
            )
            add_trigger_zh = (
                f"{reference_index.name} 30分钟级别站稳 {reference_index.current:.2f} 点上方，"
                "实时上涨占比不低于60%，且预估两市成交额不低于前一交易日。"
            )
            reduce_trigger_zh = (
                f"{reference_index.name} 30分钟级别收于 {reference_index.low:.2f} 点下方，"
                "或实时上涨占比不高于40%。"
            )
        else:
            add_trigger_en = "Unavailable because the reference index high/low did not pass validation."
            reduce_trigger_en = "Unavailable because the reference index high/low did not pass validation."
            add_trigger_zh = "参考指数高低点未通过校验，本次不提供数值型加仓触发位。"
            reduce_trigger_zh = "参考指数高低点未通过校验，本次不提供数值型减仓触发位。"

        dimensions = light.get("dimensions") or {}
        breadth_score = int((dimensions.get("breadth") or {}).get("score", 50))
        index_score = int((dimensions.get("index") or {}).get("score", 50))
        limit_score = int((dimensions.get("limit") or {}).get("score", 50))
        liquidity_score = int((dimensions.get("liquidity") or {}).get("score", 50))
        structure_score = int((dimensions.get("structure") or {}).get("score", 50))
        focus_text = self._format_ranking_summary(overview.top_sectors, limit=3)
        avoid_text = self._format_ranking_summary(overview.bottom_sectors, limit=3)

        if self._get_review_language() == "en":
            return f"""## {overview.date} Market Recap — Verified Data Edition

> Structured market signal: **{light['temperature_label']}** ({light['score']}/100). This edition contains only programmatically derived market facts and source-labelled news titles.

### 1. Validation Status
- **Core market data**: passed
- **Valid major indices**: {len(valid_indices)}
- **Market breadth**: {overview.up_count + overview.down_count + overview.flat_count} Shanghai/Shenzhen/Beijing securities
- **Aggregate turnover**: {overview.total_amount:.0f} {self._get_turnover_unit_label()}
- **Index source / fetched at**: {index_source_text} / {fetched_at_text}
- **Breadth source / session**: {overview.market_stats_source or "unlabelled"} / {overview.market_stats_trade_date or "unlabelled"}
- **Sector taxonomy / source**: Shenwan level-1 / {sector_source_text}

### 2. Market Breadth & Liquidity
- Advancers / decliners / flat: {overview.up_count} / {overview.down_count} / {overview.flat_count}
- Advancer ratio excluding flat issues: {up_ratio:.1%}
- Limit-up / limit-down: {overview.limit_up_count} / {overview.limit_down_count}
- Average major-index change: {average_change:+.2f}%
- Aggregate turnover: {overview.total_amount:.0f} {self._get_turnover_unit_label()}
- Prior-session turnover: {overview.previous_total_amount:.0f}; change {turnover_comparison_en}.

### 3. Major Indices
{indices_block or "- No validated index data is available."}

### 4. Sector / Theme Rankings
{sector_block or "- No validated sector or theme rankings are available."}

### 5. Source-labelled News
{news_block or "- No source-labelled news item is available."}

### 6. Next-session Quant Plan
- **Rule posture**: {posture_en}; model portfolio exposure band {position_range}.
- **Score formula**: breadth {breadth_score} × 30% + index {index_score} × 25% + liquidity {liquidity_score} × 20% + index alignment {structure_score} × 15% + limit-up/down {limit_score} × 10% = {score}/100.
- **Add-risk trigger**: {add_trigger_en} If confirmed, move only toward the upper bound of the exposure band.
- **Reduce-risk trigger**: {reduce_trigger_en} If confirmed, move toward the lower bound of the exposure band.
- **Otherwise**: keep exposure inside the band and do not chase an unconfirmed breakout.
- **Strength watchlist**: {focus_text or "No validated leading-sector ranking."}
- **Weakness / avoid list**: {avoid_text or "No validated lagging-sector ranking."}

### 7. Data Boundary
- Position ranges and triggers come from the fixed rules printed above; no LLM-generated support/resistance level or price target is used.
- Signal labels are deterministic summaries of breadth, index change, and limit-up/down data; they are not investment instructions.
- For reference only, not investment advice.

---
*Validation Time: {review_time}*
"""

        return f"""## {overview.date} 大盘复盘（严格数据版）

> 结构化盘面信号：**{light['temperature_label']}**（{light['score']}/100）。本报告只呈现程序计算的行情事实和带来源标识的新闻标题。

### 一、数据校验
- **核心行情数据**：通过
- **有效主要指数**：{len(valid_indices)} 个
- **市场宽度**：沪深京三市共 {overview.up_count + overview.down_count + overview.flat_count} 只证券
- **两市成交额**：{overview.total_amount:.0f} 亿元
- **指数来源 / 抓取时间**：{index_source_text} / {fetched_at_text}
- **宽度与涨跌停来源 / 数据日**：{overview.market_stats_source or "未标注"} / {overview.market_stats_trade_date or "未标注"}
- **行业分类 / 来源**：申万一级 / {sector_source_text}

### 二、市场宽度与成交
- 上涨 / 下跌 / 平盘：{overview.up_count} / {overview.down_count} / {overview.flat_count}
- 上涨占比（不含平盘）：{up_ratio:.1%}
- 涨停 / 跌停：{overview.limit_up_count} / {overview.limit_down_count}
- 主要指数平均涨跌幅：{average_change:+.2f}%
- 两市成交额：{overview.total_amount:.0f} 亿元
- 前一交易日成交额：{overview.previous_total_amount:.0f} 亿元；{turnover_comparison_zh}。

### 三、主要指数
{indices_block or "- 暂无通过校验的指数数据。"}

### 四、板块与题材排名
{sector_block or "- 暂无通过校验的板块或题材排名。"}

### 五、带来源的市场线索
{news_block or "- 暂无带来源标识的市场新闻。"}

### 六、次日量化计划
- **规则姿态**：{posture_zh}；模型组合仓位区间 {position_range}。
- **评分公式**：市场宽度 {breadth_score} × 30% + 指数强弱 {index_score} × 25% + 量价配合 {liquidity_score} × 20% + 权重/成长一致性 {structure_score} × 15% + 涨跌停结构 {limit_score} × 10% = {score}/100。
- **加仓触发**：{add_trigger_zh}满足时只向仓位区间上限移动。
- **减仓触发**：{reduce_trigger_zh}满足时向仓位区间下限移动。
- **其余情况**：仓位保持在区间内，不追未经确认的突破。
- **强势观察池**：{focus_text or "暂无通过校验的领涨板块排名。"}
- **弱势/回避池**：{avoid_text or "暂无通过校验的领跌板块排名。"}

### 七、数据边界
- 仓位区间和触发条件均来自上方固定规则，不采用大模型生成的支撑位、压力位或目标价。
- 盘面信号由宽度、指数、成交额环比、权重/成长一致性和涨跌停结构确定性计算，不构成交易指令。
- 建议仅供参考，不构成投资建议。

---
*校验时间: {review_time}*
"""

    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """使用模板生成复盘报告（无大模型时的备选方案）"""
        if self.region == "us":
            return self._generate_us_template_review(overview, news, strict=False)
        template_language = self._get_template_review_language()
        mood_code = self.profile.mood_index_code
        # 根据 mood_index_code 查找对应指数
        # cn: mood_code="000001"，idx.code 可能为 "sh000001"（以 mood_code 结尾）
        # us: mood_code="SPX"，idx.code 直接为 "SPX"
        mood_index = next(
            (
                idx
                for idx in overview.indices
                if idx.code == mood_code or idx.code.endswith(mood_code)
            ),
            None,
        )
        if mood_index:
            if mood_index.change_pct > 1:
                market_mood = self._get_market_mood_text("strong_up", template_language)
            elif mood_index.change_pct > 0:
                market_mood = self._get_market_mood_text("mild_up", template_language)
            elif mood_index.change_pct > -1:
                market_mood = self._get_market_mood_text("mild_down", template_language)
            else:
                market_mood = self._get_market_mood_text("strong_down", template_language)
        else:
            market_mood = self._get_market_mood_text("range", template_language)
        
        # 指数行情（简洁格式）
        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息
        separator = ", " if template_language == "en" else "、"
        top_text = separator.join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = separator.join([s['name'] for s in overview.bottom_sectors[:3]])
        top_concept_text = separator.join([s['name'] for s in overview.top_concepts[:3]])
        bottom_concept_text = separator.join([s['name'] for s in overview.bottom_concepts[:3]])

        if template_language == "en":
            stats_section = ""
            if self.profile.has_market_stats:
                stats_section = f"""
### 3. Breadth & Liquidity
| Metric | Value |
|--------|-------|
| Advancers | {overview.up_count} |
| Decliners | {overview.down_count} |
| Limit-up | {overview.limit_up_count} |
| Limit-down | {overview.limit_down_count} |
| Turnover ({self._get_turnover_unit_label()}) | {overview.total_amount:.0f} |
"""
            sector_section = ""
            if self.profile.has_sector_rankings and (top_text or bottom_text or top_concept_text or bottom_concept_text):
                sector_section = f"""
### 4. Sector / Theme Highlights
- **Industry Leaders**: {top_text or "N/A"}
- **Industry Laggards**: {bottom_text or "N/A"}
- **Concept Leaders**: {top_concept_text or "N/A"}
- **Concept Laggards**: {bottom_concept_text or "N/A"}
"""
            market_names = {
                "us": "US Market Recap",
                "hk": "HK Market Recap",
                "jp": "Japan Market Recap",
                "kr": "Korea Market Recap",
            }
            market_name = market_names.get(self.region, "A-share Market Recap")
            report = f"""## {overview.date} {market_name}

### 1. Market Summary
Today's {self._get_market_scope_name(template_language)} showed **{market_mood}**.

### 2. Major Indices
{indices_text or "- No index data available"}
{stats_section}
{sector_section}
### 5. Risk Alerts
Market conditions can change quickly. The data above is for reference only and does not constitute investment advice.

{self._get_strategy_markdown_block(template_language)}

---
*Review Time: {datetime.now().strftime('%H:%M')}*
"""
            return report

        market_labels = {"cn": "A股", "us": "美股", "hk": "港股", "jp": "日股", "kr": "韩股"}
        market_label = market_labels.get(self.region, "A股")
        dashboard_block = self._build_stats_block(overview) if self.profile.has_market_stats else ""
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview) if self.profile.has_sector_rankings else ""
        summary_focus = (
            "指数承接、成交额变化和板块持续性"
            if self.profile.has_market_stats and self.profile.has_sector_rankings
            else "指数承接、消息催化和整体风险状态"
        )
        market_summary_block = (
            dashboard_block
            if dashboard_block
            else (
                "暂无市场宽度数据。"
                if self.profile.has_market_stats
                else "- 当前以主要指数与可用新闻线索评估整体风险状态。"
            )
        )
        sector_section = (
            f"""
### 三、板块主线
{sector_block or "- 暂无板块涨跌榜数据。"}
"""
            if self.profile.has_sector_rankings
            else ""
        )
        funds_section = (
            """
### 四、资金与情绪
- 结合成交额和涨跌家数看，当前更适合等待确认，避免仅凭单一热点追高。
"""
            if self.profile.has_market_stats
            else ""
        )
        return f"""## {overview.date} 大盘复盘

> 今日{market_label}市场整体呈现**{market_mood}**态势，优先观察{summary_focus}。

### 一、盘面总览
{market_summary_block}

### 二、指数结构
{indices_block or indices_text or "暂无指数数据。"}
{sector_section}
{funds_section}

### 五、消息催化
- 暂无可用新闻时，应降低对题材持续性的确定性判断。

{self._get_strategy_markdown_block(template_language)}

### 七、风险提示
- 市场有风险，投资需谨慎。以上数据仅供参考，不构成投资建议。

---
*复盘时间: {datetime.now().strftime('%H:%M')}*
"""

    def _generate_us_template_review(
        self,
        overview: MarketOverview,
        news: List,
        *,
        strict: bool,
    ) -> str:
        """Deterministic US fallback with the same evidence layers as the LLM report."""
        light = self.build_market_light_snapshot(overview)
        score = int(light["score"])
        if score >= 70:
            posture, exposure = "进攻", "50%-70%"
        elif score >= 55:
            posture, exposure = "偏进攻", "35%-55%"
        elif score >= 40:
            posture, exposure = "均衡", "20%-40%"
        else:
            posture, exposure = "防守", "0%-20%"

        dimensions = self._build_us_market_light_scores(overview).get("dimensions") or {}
        context = overview.us_market_context or {}
        source_rows = context.get("sources") or []
        source_text = "；".join(
            f"{item.get('name', '未标注')}（{item.get('scope', '未标注')}，"
            f"数据日 {item.get('as_of', '-')}）"
            for item in source_rows
            if isinstance(item, dict)
        ) or "未标注"
        fetched_at = str(
            context.get("fetched_at")
            or datetime.now().astimezone().isoformat(timespec="seconds")
        )
        score_formula = (
            f"参与度 {int((dimensions.get('participation') or {}).get('score', 50))}×30% + "
            f"指数 {int((dimensions.get('index') or {}).get('score', 50))}×25% + "
            f"流动性 {int((dimensions.get('liquidity') or {}).get('score', 50))}×15% + "
            f"波动率 {int((dimensions.get('volatility') or {}).get('score', 50))}×15% + "
            f"宏观 {int((dimensions.get('macro') or {}).get('score', 50))}×15% = {score}/100"
        )
        news_block = self._build_news_block(news)
        strict_label = "严格数据版" if strict else "确定性回退版"
        return f"""## {overview.date} 美股大盘复盘（{strict_label}）

> **结论：{posture}**；模型组合风险仓位区间 **{exposure}**。该结论只由下方已校验数据计算，不根据新闻情绪改分。

### 一、数据校验
- 核心指数、ETF参与度、行业轮动、SPY量比与2年/10年美债均已通过。
- 指数和ETF交易日：{overview.date}
- 指数与 SPY/QQQ/IWM 涨跌幅交叉校验：通过
- 数据源：{source_text}
- 抓取时间：{fetched_at}
- 评分公式：{score_formula}

### 二、指数结构
{self._build_indices_block(overview)}

### 三、参与度与流动性
{self._build_us_stats_block(overview)}

### 四、行业轮动
{self._build_sector_block(overview)}

### 五、政策、政治与财报证据
{news_block or "- 本次未取得带来源的有效新闻，无法验证政策、政治或财报催化；不据此调整量化结论。"}

### 六、下一交易日量化计划
- **规则姿态**：{posture}；风险仓位保持在 {exposure}。
- **加风险条件**：主要指数同向、RSP与IWM相对SPY改善、行业上涨覆盖扩大，且VIX不反向上升。
- **减风险条件**：指数转弱，同时行业覆盖收缩，或VIX与短端利率共同上行。
- **执行纪律**：条件未确认时不追涨；政策或财报新闻必须等官方披露和价格确认。

### 七、数据边界与风险
- ETF指标是透明的全市场参与度等价代理，不是交易所涨跌家数，也不是资金净流入。
- 新闻只列带来源事实；本模板不把标题自动解释为利好或利空。
- 建议仅供参考，不构成投资建议。

---
*校验时间: {datetime.now().astimezone().isoformat(timespec='seconds')}*
"""
    
    def _run_daily_review_parts(self) -> MarketLightReviewResult:
        """Run market review once and keep report/snapshot on the same overview."""
        logger.info("========== 开始大盘复盘分析 ==========")

        # 1. 获取市场概览
        overview = self.get_market_overview()
        quality = self._assess_market_data_quality(overview)

        # 2. 搜索市场新闻
        if quality["core_data_ready"]:
            news = self.search_market_news()
            news = self._merge_persisted_market_intelligence(news)
        else:
            news = []
            logger.warning(
                "[大盘] %s action=search_market_news status=skipped "
                "reason=core_data_unavailable missing_core_fields=%s",
                self._log_context(),
                ",".join(quality["missing_core_fields"]),
            )

        # 3. 生成复盘报告
        report = self.generate_market_review(overview, news)
        snapshot = (
            self.build_market_light_snapshot(overview)
            if self._supports_market_light() and quality["core_data_ready"]
            else None
        )
        structured_payload = self.build_market_review_payload(
            overview,
            news,
            report,
            snapshot,
        )

        logger.info("========== 大盘复盘分析完成 ==========")

        return MarketLightReviewResult(
            overview=overview,
            report=report,
            market_light_snapshot=snapshot,
            structured_payload=structured_payload,
        )

    def _merge_persisted_market_intelligence(self, news: List) -> List:
        """Merge local persisted market intelligence and search news with bounded prompt/payload slot preservation."""
        search_news = list(news or [])
        merged_local = []
        seen_urls = {
            self._get_news_field(item, "url")
            for item in search_news
            if self._get_news_field(item, "url")
        }
        try:
            service = IntelligenceService(config=self.config)
            service.refresh_auto_sources()
            payload = service.list_items(
                scope_type="market",
                market=self.region,
                published_days=max(1, int(self.config.get_effective_news_window_days() or 1)),
                page=1,
                page_size=6,
            )
            for item in payload.get("items", []):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "")
                if url and url in seen_urls:
                    continue
                seen_urls.add(url)
                merged_local.append({
                    "title": item.get("title") or "未命名资讯",
                    "snippet": item.get("summary") or "",
                    "source": item.get("source") or item.get("source_name") or "local-intel",
                    "published_date": item.get("published_at") or "",
                    "url": "" if url.startswith("no-url:intel:") else url,
                })
        except Exception as exc:
            logger.debug("[大盘] %s action=load_local_intelligence status=failed error=%s", self._log_context(), exc)
        merged_news = []
        merged_local_index = 0
        merged_search_index = 0
        while merged_local_index < len(merged_local) or merged_search_index < len(search_news):
            if merged_local_index < len(merged_local):
                merged_news.append(merged_local[merged_local_index])
                merged_local_index += 1
            if merged_search_index < len(search_news):
                merged_news.append(search_news[merged_search_index])
                merged_search_index += 1
        return merged_news

    def run_daily_review(self) -> str:
        """
        执行每日大盘复盘流程

        Returns:
            复盘报告文本
        """
        return self.run_daily_review_with_snapshot().report

    def run_daily_review_with_snapshot(self) -> MarketLightReviewResult:
        """Run daily review and return the report plus its structured Market Light snapshot."""
        return self._run_daily_review_parts()


# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    analyzer = MarketAnalyzer()
    
    # 测试获取市场概览
    overview = analyzer.get_market_overview()
    print(f"\n=== 市场概览 ===")
    print(f"日期: {overview.date}")
    print(f"指数数量: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"上涨: {overview.up_count} | 下跌: {overview.down_count}")
    print(f"成交额: {overview.total_amount:.0f}亿")
    
    # 测试生成模板报告
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== 复盘报告 ===")
    print(report)
