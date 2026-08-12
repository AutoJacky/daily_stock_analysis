"""Free, source-labelled institutional context for fused market reports.

The collector is deliberately deterministic: every block carries a status,
source and as-of date.  Missing public data remains missing instead of being
filled by an LLM or a paid fallback.
"""

from __future__ import annotations

import math
import io
import re
from contextlib import redirect_stderr
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import pandas as pd


@dataclass(frozen=True)
class EvidenceBlock:
    status: str
    source: str
    as_of: str
    data: Any
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "as_of": self.as_of,
            "data": self.data,
            "note": self.note,
        }


def _date_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return str(value).strip()[:10]


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(history: Iterable[Any], current: float) -> Optional[float]:
    values = [value for item in history if (value := _number(item)) is not None]
    if not values:
        return None
    return round(sum(value <= current for value in values) / len(values) * 100.0, 1)


def _latest_on_or_before(
    frame: pd.DataFrame,
    as_of: date,
    *,
    date_column: str = "日期",
) -> tuple[Optional[pd.Series], str]:
    if frame is None or frame.empty or date_column not in frame.columns:
        return None, ""
    work = frame.copy()
    work["__date"] = pd.to_datetime(work[date_column], errors="coerce").dt.date
    work = work[work["__date"].notna() & (work["__date"] <= as_of)]
    if work.empty:
        return None, ""
    row = work.sort_values("__date").iloc[-1]
    return row, row["__date"].isoformat()


class InstitutionalMarketContextCollector:
    """Collect public, no-key context with fail-open evidence blocks."""

    def __init__(self, ak_module: Any = None):
        if ak_module is None:
            import akshare as ak_module
        self.ak = ak_module

    @staticmethod
    def _missing(source: str, note: str) -> Dict[str, Any]:
        return EvidenceBlock("missing", source, "", {}, note).to_dict()

    def _safe(self, source: str, call: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        try:
            return call()
        except Exception as exc:
            return self._missing(source, f"公开接口暂不可用：{type(exc).__name__}")

    def _index_valuation(self, symbol: str, as_of: date) -> Dict[str, Any]:
        pe_frame = self.ak.stock_index_pe_lg(symbol=symbol)
        pb_frame = self.ak.stock_index_pb_lg(symbol=symbol)
        pe_row, pe_date = _latest_on_or_before(pe_frame, as_of)
        pb_row, pb_date = _latest_on_or_before(pb_frame, as_of)
        if pe_row is None or pb_row is None or pe_date != pb_date:
            return self._missing("乐咕乐股 / AkShare", "PE/PB同日数据未取得")
        pe = _number(pe_row.get("滚动市盈率"))
        pb = _number(pb_row.get("市净率"))
        if pe is None or pb is None or pe <= 0 or pb <= 0:
            return self._missing("乐咕乐股 / AkShare", "PE/PB字段无效")
        start = pd.Timestamp(pe_date) - pd.Timedelta(days=365 * 5)
        pe_dates = pd.to_datetime(pe_frame["日期"], errors="coerce")
        pb_dates = pd.to_datetime(pb_frame["日期"], errors="coerce")
        pe_history = pe_frame.loc[pe_dates >= start, "滚动市盈率"]
        pb_history = pb_frame.loc[pb_dates >= start, "市净率"]
        return EvidenceBlock(
            "ok",
            "乐咕乐股 / AkShare",
            pe_date,
            {
                "name": symbol,
                "pe_ttm": round(pe, 2),
                "pe_5y_percentile": _percentile(pe_history, pe),
                "pb": round(pb, 2),
                "pb_5y_percentile": _percentile(pb_history, pb),
            },
        ).to_dict()

    def _rates_and_erp(self, as_of: date, hs300: Mapping[str, Any]) -> Dict[str, Any]:
        start = (as_of - timedelta(days=20)).strftime("%Y%m%d")
        frame = self.ak.bond_zh_us_rate(start_date=start)
        row, row_date = _latest_on_or_before(frame, as_of)
        if row is None:
            return self._missing("东方财富中美国债 / AkShare", "国债收益率未取得")
        cn10 = _number(row.get("中国国债收益率10年"))
        us10 = _number(row.get("美国国债收益率10年"))
        pe = _number((hs300.get("data") or {}).get("pe_ttm"))
        data: Dict[str, Any] = {
            "cn_10y_yield": cn10,
            "us_10y_yield": us10,
        }
        if pe and pe > 0 and cn10 is not None:
            data["equity_risk_premium_proxy"] = round(100.0 / pe - cn10, 2)
            data["erp_method"] = "沪深300盈利收益率(1/PE)-中国10年国债收益率"
        return EvidenceBlock(
            "ok" if cn10 is not None else "partial",
            "东方财富中美国债 / AkShare；ERP为程序计算",
            row_date,
            data,
        ).to_dict()

    def _margin(self, as_of: date) -> Dict[str, Any]:
        sh = self.ak.macro_china_market_margin_sh()
        sz = self.ak.macro_china_market_margin_sz()
        sh_row, sh_date = _latest_on_or_before(sh, as_of)
        sz_row, sz_date = _latest_on_or_before(sz, as_of)
        if sh_row is None or sz_row is None or sh_date != sz_date:
            return self._missing("上交所/深交所融资融券汇总 / AkShare", "沪深同日数据未取得")
        sh_balance = _number(sh_row.get("融资余额"))
        sz_balance = _number(sz_row.get("融资余额"))
        if sh_balance is None or sz_balance is None:
            return self._missing("上交所/深交所融资融券汇总 / AkShare", "融资余额字段无效")

        def previous_balance(frame: pd.DataFrame, current_date: str) -> Optional[float]:
            work = frame.copy()
            work["__date"] = pd.to_datetime(work["日期"], errors="coerce").dt.date
            rows = work[work["__date"] < date.fromisoformat(current_date)].sort_values("__date")
            return _number(rows.iloc[-1].get("融资余额")) if not rows.empty else None

        prior_sh = previous_balance(sh, sh_date)
        prior_sz = previous_balance(sz, sz_date)
        total = sh_balance + sz_balance
        prior_total = (
            prior_sh + prior_sz
            if prior_sh is not None and prior_sz is not None
            else None
        )
        return EvidenceBlock(
            "ok",
            "上交所/深交所融资融券汇总 / AkShare",
            sh_date,
            {
                "financing_balance_100m_cny": round(total / 1e8, 2),
                "change_100m_cny": (
                    round((total - prior_total) / 1e8, 2)
                    if prior_total is not None
                    else None
                ),
            },
        ).to_dict()

    def _northbound(self, as_of: date) -> Dict[str, Any]:
        frame = self.ak.stock_hsgt_hist_em(symbol="北向资金")
        row, row_date = _latest_on_or_before(frame, as_of)
        if row is None:
            return self._missing("东方财富沪深港通 / AkShare", "北向历史记录未取得")
        net = _number(row.get("当日成交净买额"))
        if net is None:
            return EvidenceBlock(
                "not_supported",
                "东方财富沪深港通 / AkShare",
                row_date,
                {},
                "互联互通披露口径调整后该字段为空，不把0解释为无流入",
            ).to_dict()
        return EvidenceBlock(
            "ok",
            "东方财富沪深港通 / AkShare",
            row_date,
            {"net_buy_billion_cny": round(net, 2)},
        ).to_dict()

    def _industry_valuation(self, as_of: date) -> Dict[str, Any]:
        frame = None
        used_date = ""
        last_error = ""
        for offset in range(0, 8):
            candidate = as_of - timedelta(days=offset)
            try:
                frame = self.ak.stock_industry_pe_ratio_cninfo(
                    symbol="证监会行业分类", date=candidate.strftime("%Y%m%d")
                )
            except Exception as exc:
                last_error = type(exc).__name__
                continue
            if frame is not None and not frame.empty:
                used_date = candidate.isoformat()
                break
        if frame is None or frame.empty:
            note = "近8日行业估值未取得"
            if last_error:
                note += f"（最近错误 {last_error}）"
            return self._missing("巨潮资讯行业市盈率 / AkShare", note)
        level = pd.to_numeric(frame.get("行业层级"), errors="coerce")
        pe = pd.to_numeric(frame.get("静态市盈率-加权平均"), errors="coerce")
        work = frame[(level == 1) & pe.notna() & (pe > 0)].copy()
        work["__pe"] = pe[(level == 1) & pe.notna() & (pe > 0)]
        rows = [
            {"name": str(row["行业名称"]), "pe_static": round(float(row["__pe"]), 2)}
            for _, row in work.sort_values("__pe").iterrows()
        ]
        return EvidenceBlock(
            "ok" if rows else "missing",
            "巨潮资讯（证监会一级行业）/ AkShare",
            used_date,
            {"lowest": rows[:5], "highest": list(reversed(rows[-5:]))},
            "与申万一级涨跌榜口径不同，仅作独立估值观察，不直接归因",
        ).to_dict()

    @staticmethod
    def _yahoo_global_linkage(as_of: date) -> Dict[str, Any]:
        """Use Yahoo's free delayed history; never label it realtime."""

        import yfinance as yf

        symbols = {
            "sp500": ("^GSPC", "标普500"),
            "nasdaq": ("^IXIC", "纳斯达克综合"),
            "semiconductor": ("^SOX", "费城半导体"),
            "dxy": ("DX-Y.NYB", "美元指数DXY"),
            "usdcny": ("CNY=X", "美元/在岸人民币"),
            "usdcnh": ("CNH=X", "美元/离岸人民币"),
            "brent": ("BZ=F", "布伦特原油"),
            "gold": ("GC=F", "COMEX黄金"),
            "copper": ("HG=F", "COMEX铜"),
            "vix": ("^VIX", "VIX"),
        }
        start = as_of - timedelta(days=12)
        # yfinance currently writes batch failures to stderr even with
        # progress disabled. Keep report/Actions logs clean and represent the
        # failure through the evidence status below.
        with redirect_stderr(io.StringIO()):
            frame = yf.download(
                [symbol for symbol, _ in symbols.values()],
                start=start.isoformat(),
                end=(as_of + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )
        if frame is None or frame.empty:
            return EvidenceBlock(
                "missing", "Yahoo Finance / yfinance", "", {}, "全球代理行情未取得"
            ).to_dict()
        close = frame.get("Close")
        if close is None:
            return EvidenceBlock(
                "missing", "Yahoo Finance / yfinance", "", {}, "收盘字段未取得"
            ).to_dict()
        if isinstance(close, pd.Series):
            close = close.to_frame()
        items: Dict[str, Any] = {}
        dates: list[str] = []
        for key, (symbol, label) in symbols.items():
            if symbol not in close.columns:
                continue
            series = pd.to_numeric(close[symbol], errors="coerce").dropna()
            if len(series) < 2:
                continue
            current = float(series.iloc[-1])
            previous = float(series.iloc[-2])
            item_date = pd.Timestamp(series.index[-1]).date().isoformat()
            dates.append(item_date)
            items[key] = {
                "name": label,
                "close": round(current, 4),
                "change_pct": round((current / previous - 1.0) * 100.0, 2),
                "as_of": item_date,
            }
        return EvidenceBlock(
            "ok" if len(items) >= 7 else "partial",
            "Yahoo Finance / yfinance（免费延时收盘代理）",
            max(dates) if dates else "",
            items,
            "不同市场休市日可能不同，每项保留独立数据日",
        ).to_dict()

    @staticmethod
    def _sina_global_linkage(as_of: date) -> Dict[str, Any]:
        """Independent, free delayed fallback with per-item session dates."""

        import requests

        symbols = "gb_$inx,gb_ixic,gb_sox,DINIW,hf_GC,hf_CAD,fx_susdcny,fx_susdcnh"
        response = requests.get(
            f"https://hq.sinajs.cn/list={symbols}",
            headers={"Referer": "https://finance.sina.com.cn/"},
            timeout=15,
        )
        response.raise_for_status()
        text = response.content.decode("gb18030", errors="replace")
        records = {
            key: value.split(",")
            for key, value in re.findall(r'hq_str_([^=]+)="([^"]*)"', text)
            if value
        }
        items: Dict[str, Any] = {}
        dates: list[str] = []

        def add_item(
            key: str,
            name: str,
            close: Any,
            change_pct: Any,
            item_date: str,
        ) -> None:
            price = _number(close)
            change = _number(change_pct)
            try:
                parsed_date = date.fromisoformat(item_date)
            except ValueError:
                return
            if price is None or parsed_date > as_of:
                return
            dates.append(item_date)
            items[key] = {
                "name": name,
                "close": round(price, 4),
                "change_pct": round(change, 2) if change is not None else None,
                "as_of": item_date,
            }

        index_specs = {
            "gb_$inx": ("sp500", "标普500"),
            "gb_ixic": ("nasdaq", "纳斯达克综合"),
            "gb_sox": ("semiconductor", "费城半导体"),
        }
        for provider_key, (key, label) in index_specs.items():
            values = records.get(provider_key) or []
            if len(values) < 4:
                continue
            date_match = re.search(r"\b([A-Z][a-z]{2}\s+\d{1,2})\b", ",".join(values))
            year = next((value for value in reversed(values) if re.fullmatch(r"20\d{2}", value)), "")
            try:
                item_date = datetime.strptime(
                    f"{date_match.group(1)} {year}", "%b %d %Y"
                ).date().isoformat() if date_match and year else ""
            except ValueError:
                item_date = ""
            add_item(key, label, values[1], values[2], item_date)

        dxy = records.get("DINIW") or []
        if len(dxy) >= 11:
            add_item("dxy", "美元指数DXY", dxy[1], None, dxy[10])

        for provider_key, key, label in (
            ("fx_susdcny", "usdcny", "美元/在岸人民币"),
            ("fx_susdcnh", "usdcnh", "美元/离岸人民币"),
        ):
            values = records.get(provider_key) or []
            if len(values) >= 18:
                add_item(key, label, values[8], values[10], values[17])

        for provider_key, key, label in (
            ("hf_GC", "gold", "COMEX黄金"),
            ("hf_CAD", "copper", "LME铜"),
        ):
            values = records.get(provider_key) or []
            if len(values) < 13:
                continue
            current = _number(values[0])
            previous_settlement = _number(values[7])
            change = (
                (current / previous_settlement - 1.0) * 100.0
                if current is not None and previous_settlement not in (None, 0)
                else None
            )
            add_item(key, label, current, change, values[12])

        return EvidenceBlock(
            "partial" if items else "missing",
            "新浪财经免费延时行情（Yahoo受限时独立备用）",
            max(dates) if dates else "",
            items,
            "指数使用美国交易日；汇率/商品为带数据日的延时快照；布伦特与VIX未取得时保持缺失",
        ).to_dict()

    @classmethod
    def _global_linkage(cls, as_of: date) -> Dict[str, Any]:
        try:
            yahoo = cls._yahoo_global_linkage(as_of)
        except Exception:
            yahoo = {}
        if yahoo.get("status") in {"ok", "partial"} and yahoo.get("data"):
            return yahoo
        return cls._sina_global_linkage(as_of)

    def collect(self, market: str, as_of: Optional[date] = None) -> Dict[str, Any]:
        as_of = as_of or datetime.now().date()
        if market != "cn":
            return {
                "market": market,
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "coverage": {
                    "valuation": self._missing("免费公开源", "美股指数估值历史分位本轮未接入"),
                    "capital_flow": self._missing("免费公开源", "美股ETF申赎与机构资金流本轮未接入"),
                },
                "global_linkage": self._safe(
                    "Yahoo Finance / yfinance",
                    lambda: self._global_linkage(as_of),
                ),
            }
        hs300 = self._safe(
            "乐咕乐股 / AkShare", lambda: self._index_valuation("沪深300", as_of)
        )
        csi500 = self._safe(
            "乐咕乐股 / AkShare", lambda: self._index_valuation("中证500", as_of)
        )
        return {
            "market": market,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "valuation": {
                "hs300": hs300,
                "csi500": csi500,
                "star50": EvidenceBlock(
                    "not_supported",
                    "免费公开源",
                    "",
                    {},
                    "未取得科创50同口径PE/PB历史序列；不以科创板全市场口径冒充",
                ).to_dict(),
                "rates_erp": self._safe(
                    "东方财富中美国债 / AkShare",
                    lambda: self._rates_and_erp(as_of, hs300),
                ),
            },
            "capital_flow": {
                "northbound": self._safe(
                    "东方财富沪深港通 / AkShare", lambda: self._northbound(as_of)
                ),
                "margin": self._safe(
                    "上交所/深交所融资融券汇总 / AkShare", lambda: self._margin(as_of)
                ),
                "etf_creation_redemption": EvidenceBlock(
                    "not_supported",
                    "免费公开源",
                    "",
                    {},
                    "全市场ETF申购赎回统一口径未取得",
                ).to_dict(),
            },
            "industry_valuation": self._safe(
                "巨潮资讯行业市盈率 / AkShare",
                lambda: self._industry_valuation(as_of),
            ),
            "global_linkage": self._safe(
                "Yahoo Finance / yfinance",
                lambda: self._global_linkage(as_of),
            ),
            "industry_activity": EvidenceBlock(
                "partial",
                "申万一级涨跌榜 + 免费公开源",
                as_of.isoformat(),
                {},
                "本轮仅有价格强弱；销售面积、CPO出货、炼厂开工率、煤炭库存等高频景气数据未形成稳定同日免费契约",
            ).to_dict(),
        }


def context_as_prompt_text(context: Mapping[str, Any]) -> str:
    """Stable JSON evidence text for Qwen review and numeric allow-listing."""

    import json

    return json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)
