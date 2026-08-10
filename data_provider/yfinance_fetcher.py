# -*- coding: utf-8 -*-
"""
===================================
YfinanceFetcher - 兜底数据源 (Priority 4)
===================================

数据来源：Yahoo Finance（通过 yfinance 库）
特点：国际数据源、可能有延迟或缺失
定位：当所有国内数据源都失败时的最后保障

关键策略：
1. 自动将 A 股代码转换为 yfinance 格式（.SS / .SZ）
2. 处理 Yahoo Finance 的数据格式差异
3. 失败后指数退避重试
"""

import csv
import logging
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional, List, Dict, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS, is_bse_code
from .realtime_types import UnifiedRealtimeQuote, RealtimeSource
from .us_index_mapping import get_us_index_yf_symbol, is_us_stock_code
from .yfinance_fundamental_adapter import _safe_float
from src.services.market_symbol_utils import get_suffix_market, is_suffix_market_symbol

# 可选导入本地股票映射补丁，若缺失则使用空字典兜底
try:
    from src.data.stock_mapping import STOCK_NAME_MAP, is_meaningful_stock_name
except (ImportError, ModuleNotFoundError):
    STOCK_NAME_MAP = {}

    def is_meaningful_stock_name(name: str | None, stock_code: str) -> bool:
        """简单的名称有效性校验兜底"""
        if not name:
            return False
        n = str(name).strip()
        return bool(n and n.upper() != str(stock_code).strip().upper())

import os

logger = logging.getLogger(__name__)


class YfinanceFetcher(BaseFetcher):
    """
    Yahoo Finance 数据源实现

    优先级：4（最低，作为兜底）
    数据来源：Yahoo Finance

    关键策略：
    - 自动转换股票代码格式
    - 处理时区和数据格式差异
    - 失败后指数退避重试

    注意事项：
    - A 股数据可能有延迟
    - 某些股票可能无数据
    - 数据精度可能与国内源略有差异
    """

    name = "YfinanceFetcher"
    priority = int(os.getenv("YFINANCE_PRIORITY", "4"))

    _US_MARKET_PROXIES = {
        "SPY": "标普500 ETF",
        "RSP": "标普500等权 ETF",
        "IWM": "罗素2000 ETF",
        "QQQ": "纳斯达克100 ETF",
    }
    _US_SECTOR_ETFS = {
        "XLK": "信息技术",
        "XLC": "通信服务",
        "XLY": "可选消费",
        "XLP": "必需消费",
        "XLE": "能源",
        "XLF": "金融",
        "XLV": "医疗保健",
        "XLI": "工业",
        "XLB": "原材料",
        "XLRE": "房地产",
        "XLU": "公用事业",
    }
    _US_FRED_SERIES = {
        "DGS2": ("美国2年期国债收益率", "%"),
        "DGS10": ("美国10年期国债收益率", "%"),
        "DTWEXBGS": ("美联储广义美元指数", "index"),
    }

    def __init__(self):
        """初始化 YfinanceFetcher"""
        pass

    @staticmethod
    def _completed_index_session(return_code: str) -> str:
        """Resolve the latest fully closed session for an index family."""

        code = str(return_code or "").upper()
        if code in {"SPX", "IXIC", "NDX", "DJI", "RUT", "VIX"}:
            market = "us"
        elif code in {"HSI", "HSTECH", "HSCEI"}:
            market = "hk"
        elif code in {"N225", "TOPX"}:
            market = "jp"
        elif code in {"KOSPI", "KOSDAQ"}:
            market = "kr"
        elif code in {"TWII", "TWOII"}:
            market = "tw"
        else:
            market = "cn"
        try:
            from src.core.trading_calendar import get_effective_trading_date

            return get_effective_trading_date(market).isoformat()
        except Exception:
            return ""

    @staticmethod
    def _is_jp_kr_suffix_stock(stock_code: str) -> bool:
        """Return True for supported JP/KR suffix-only Yahoo symbols."""
        return is_suffix_market_symbol(stock_code, "jp") or is_suffix_market_symbol(stock_code, "kr")

    @staticmethod
    def _is_tw_suffix_stock(stock_code: str) -> bool:
        """Return True for supported Taiwan suffix-only Yahoo symbols (TWSE `.TW` / TPEx `.TWO`).

        Taiwan base codes are 4-6 digits (common stocks 4, ETFs/others up to 6,
        e.g. 00878 / 006208), wider than the JP `.T` range.
        """
        return is_suffix_market_symbol(stock_code, "tw")

    def _convert_stock_code(self, stock_code: str) -> str:
        """
        转换股票代码为 Yahoo Finance 格式

        Yahoo Finance 代码格式：
        - A股沪市：600519.SS (Shanghai Stock Exchange)
        - A股深市：000001.SZ (Shenzhen Stock Exchange)
        - 港股：0700.HK (Hong Kong Stock Exchange)
        - 美股：AAPL, TSLA, GOOGL (无需后缀)

        Args:
            stock_code: 原始代码，如 '600519', 'hk00700', 'AAPL'

        Returns:
            Yahoo Finance 格式代码

        Examples:
            >>> fetcher._convert_stock_code('600519')
            '600519.SS'
            >>> fetcher._convert_stock_code('hk00700')
            '0700.HK'
            >>> fetcher._convert_stock_code('AAPL')
            'AAPL'
        """
        code = stock_code.strip().upper()

        # 美股指数：映射到 Yahoo Finance 符号（如 SPX -> ^GSPC）
        yf_symbol, _ = get_us_index_yf_symbol(code)
        if yf_symbol:
            logger.debug(f"识别为美股指数: {code} -> {yf_symbol}")
            return yf_symbol

        # 美股：1-5 个大写字母（可选 .X 后缀），原样返回
        if is_us_stock_code(code):
            logger.debug(f"识别为美股代码: {code}")
            return code

        # 日股/韩股/台股 MVP：显式 Yahoo Finance suffix-only 代码，原样传给 Yahoo。
        if self._is_jp_kr_suffix_stock(code) or self._is_tw_suffix_stock(code):
            logger.debug(f"识别为日韩台 Yahoo suffix 代码: {code}")
            return code

        # 港股：hk前缀 -> .HK后缀
        if code.startswith('HK'):
            hk_code = code[2:].lstrip('0') or '0'  # 去除前导0，但保留至少一个0
            hk_code = hk_code.zfill(4)  # 补齐到4位
            logger.debug(f"转换港股代码: {stock_code} -> {hk_code}.HK")
            return f"{hk_code}.HK"

        # 已经包含后缀的情况
        if '.SS' in code or '.SZ' in code or '.HK' in code or '.BJ' in code:
            return code

        # 去除可能的 .SH 后缀
        code = code.replace('.SH', '')

        # ETF: Shanghai ETF (51xx, 52xx, 56xx, 58xx) -> .SS; Shenzhen ETF (15xx, 16xx, 18xx) -> .SZ
        if len(code) == 6:
            if code.startswith(('51', '52', '56', '58')):
                return f"{code}.SS"
            if code.startswith(('15', '16', '18')):
                return f"{code}.SZ"

        # BSE (Beijing Stock Exchange): 8xxxxx, 4xxxxx, 920xxx
        if is_bse_code(code):
            base = code.split('.')[0] if '.' in code else code
            return f"{base}.BJ"

        # A股：根据代码前缀判断市场
        if code.startswith(('600', '601', '603', '688')):
            return f"{code}.SS"
        elif code.startswith(('000', '002', '300')):
            return f"{code}.SZ"
        else:
            logger.warning(f"无法确定股票 {code} 的市场，默认使用深市")
            return f"{code}.SZ"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Yahoo Finance 获取原始数据

        使用 yfinance.download() 获取历史数据

        流程：
        1. 转换股票代码格式
        2. 调用 yfinance API
        3. 处理返回数据
        """
        import yfinance as yf

        # 转换代码格式
        yf_code = self._convert_stock_code(stock_code)

        logger.debug(f"调用 yfinance.download({yf_code}, {start_date}, {end_date})")

        try:
            # 使用 yfinance 下载数据
            df = yf.download(
                tickers=yf_code,
                start=start_date,
                end=end_date,
                progress=False,  # 禁止进度条
                auto_adjust=True,  # 自动调整价格（复权）
                multi_level_index=True
            )

            # 筛选出 yf_code 的列, 避免多只股票数据混淆
            if isinstance(df.columns, pd.MultiIndex) and len(df.columns) > 1:
                ticker_level = df.columns.get_level_values(1)
                mask = ticker_level == yf_code
                if mask.any():
                    df = df.loc[:, mask].copy()

            if df.empty:
                raise DataFetchError(f"Yahoo Finance 未查询到 {stock_code} 的数据")

            return df

        except Exception as e:
            if isinstance(e, DataFetchError):
                raise
            raise DataFetchError(f"Yahoo Finance 获取数据失败: {e}") from e

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 Yahoo Finance 数据

        yfinance 返回的列名：
        Open, High, Low, Close, Volume（索引是日期）

        注意：新版 yfinance 返回 MultiIndex 列名，如 ('Close', 'AMD')
        需要先扁平化列名再进行处理

        需要映射到标准列名：
        date, open, high, low, close, volume, amount, pct_chg
        """
        df = df.copy()

        # 处理 MultiIndex 列名（新版 yfinance 返回格式）
        # 例如: ('Close', 'AMD') -> 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            logger.debug("检测到 MultiIndex 列名，进行扁平化处理")
            # 取第一级列名（Price level: Close, High, Low, etc.）
            df.columns = df.columns.get_level_values(0)

        # 重置索引，将日期从索引变为列
        df = df.reset_index()

        # 列名映射（yfinance 使用首字母大写）
        column_mapping = {
            'Date': 'date',
            'Datetime': 'date',
            'datetime': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        }

        df = df.rename(columns=column_mapping)
        if 'date' not in df.columns:
            index_col = df.columns[0] if len(df.columns) else None
            if index_col is not None:
                candidate = df[index_col]
                if pd.api.types.is_datetime64_any_dtype(candidate):
                    df = df.rename(columns={index_col: 'date'})
                elif not pd.api.types.is_numeric_dtype(candidate):
                    parsed_dates = pd.to_datetime(candidate, errors='coerce')
                    if parsed_dates.notna().any():
                        df = df.rename(columns={index_col: 'date'})
                        df['date'] = parsed_dates

        # 计算涨跌幅（因为 yfinance 不直接提供）
        if 'close' in df.columns:
            df['pct_chg'] = df['close'].pct_change() * 100
            df['pct_chg'] = df['pct_chg'].fillna(0).round(2)

        # 计算成交额（yfinance 不提供，使用估算值）
        # 成交额 ≈ 成交量 * 平均价格
        if 'volume' in df.columns and 'close' in df.columns:
            df['amount'] = df['volume'] * df['close']
        else:
            df['amount'] = 0

        # 添加股票代码列
        df['code'] = stock_code

        # 只保留需要的列
        keep_cols = ['code'] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]

        return df

    def _fetch_yf_ticker_data(self, yf, yf_code: str, name: str, return_code: str) -> Optional[Dict[str, Any]]:
        """
        通过 yfinance 拉取单个指数/股票的行情数据。

        Args:
            yf: yfinance 模块引用
            yf_code: yfinance 使用的代码（如 '000001.SS'、'^GSPC'）
            name: 指数显示名称
            return_code: 写入结果 dict 的 code 字段（如 'sh000001'、'SPX'）

        Returns:
            行情字典，失败时返回 None
        """
        ticker = yf.Ticker(yf_code)
        # Yahoo 的 ``2d`` 窗口在部分亚洲指数的盘中/收盘后查询中
        # 会只返回当天一行。旧逻辑因此把当天收盘同时当作昨收，
        # 让多个实际涨跌的指数显示为 0.00%。扩大窗口并强制要求
        # 两个可用交易日；仍不足时宁可让上层换源，不伪造平盘。
        hist = ticker.history(period='5d')
        if hist.empty:
            return None
        hist = hist.dropna(subset=['Close']).sort_index()
        completed_session = self._completed_index_session(return_code)
        if completed_session:
            index_dates = pd.to_datetime(hist.index, errors="coerce")
            session_dates = pd.Series(index_dates, index=hist.index).dt.date
            cutoff_date = datetime.fromisoformat(completed_session).date()
            hist = hist.loc[session_dates <= cutoff_date]
        if len(hist) < 2:
            logger.warning(
                "[Yfinance] %s 指数历史仅 %d 个有效交易日，"
                "无法校验昨收与涨跌幅",
                yf_code,
                len(hist),
            )
            return None
        today_row = hist.iloc[-1]
        prev_row = hist.iloc[-2]
        price = float(today_row['Close'])
        prev_close = float(prev_row['Close'])
        change = price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0
        high = float(today_row['High'])
        low = float(today_row['Low'])
        # 振幅 = (最高 - 最低) / 昨收 * 100
        amplitude = ((high - low) / prev_close * 100) if prev_close else 0
        return {
            'code': return_code,
            'name': name,
            'current': price,
            'change': change,
            'change_pct': change_pct,
            'open': float(today_row['Open']),
            'high': high,
            'low': low,
            'prev_close': prev_close,
            'volume': float(today_row['Volume']),
            'amount': 0.0,  # Yahoo Finance 不提供准确成交额
            'amplitude': amplitude,
            'trade_date': pd.Timestamp(hist.index[-1]).date().isoformat(),
            'previous_trade_date': pd.Timestamp(hist.index[-2]).date().isoformat(),
            'source': 'Yahoo Finance/yfinance',
        }

    def get_main_indices(self, region: str = "cn") -> Optional[List[Dict[str, Any]]]:
        """
        获取主要指数行情 (Yahoo Finance)，支持 A 股、美股、港股、日股、韩股与台股。
        region=us 时委托给 _get_us_main_indices。
        region=hk 时委托给 _get_hk_main_indices。
        region=jp/kr/tw 时分别委托给对应市场指数方法。
        """
        import yfinance as yf

        if region == "us":
            return self._get_us_main_indices(yf)
        if region == "hk":
            return self._get_hk_main_indices(yf)
        if region == "jp":
            return self._get_jp_main_indices(yf)
        if region == "kr":
            return self._get_kr_main_indices(yf)
        if region == "tw":
            return self._get_tw_main_indices(yf)

        # A 股指数：akshare 代码 -> (yfinance 代码, 显示名称)
        yf_mapping = {
            'sh000001': ('000001.SS', '上证指数'),
            'sz399001': ('399001.SZ', '深证成指'),
            'sz399006': ('399006.SZ', '创业板指'),
            'sh000688': ('000688.SS', '科创50'),
            'sh000016': ('000016.SS', '上证50'),
            'sh000300': ('000300.SS', '沪深300'),
        }

        results = []
        try:
            for ak_code, (yf_code, name) in yf_mapping.items():
                try:
                    item = self._fetch_yf_ticker_data(yf, yf_code, name, ak_code)
                    if item:
                        results.append(item)
                        logger.debug(f"[Yfinance] 获取指数 {name} 成功")
                except Exception as e:
                    logger.warning(f"[Yfinance] 获取指数 {name} 失败: {e}")

            if results:
                logger.info(f"[Yfinance] 成功获取 {len(results)} 个 A 股指数行情")
                return results

        except Exception as e:
            logger.error(f"[Yfinance] 获取 A 股指数行情失败: {e}")

        return None

    def _get_us_main_indices(self, yf) -> Optional[List[Dict[str, Any]]]:
        """获取美股主要指数行情，覆盖大盘、科技、小盘与波动率。"""
        # 大盘复盘所需核心美股指数
        us_indices = ['SPX', 'IXIC', 'NDX', 'DJI', 'RUT', 'VIX']
        results = []
        try:
            for code in us_indices:
                yf_symbol, name = get_us_index_yf_symbol(code)
                if not yf_symbol:
                    continue
                try:
                    item = self._fetch_yf_ticker_data(yf, yf_symbol, name, code)
                    if item:
                        results.append(item)
                        logger.debug(f"[Yfinance] 获取美股指数 {name} 成功")
                except Exception as e:
                    logger.warning(f"[Yfinance] 获取美股指数 {name} 失败: {e}")

            if results:
                logger.info(f"[Yfinance] 成功获取 {len(results)} 个美股指数行情")
                return results

        except Exception as e:
            logger.error(f"[Yfinance] 获取美股指数行情失败: {e}")

        return None

    @staticmethod
    def _extract_yf_symbol_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Extract one symbol from a grouped yfinance batch response."""
        if raw is None or raw.empty:
            return pd.DataFrame()
        if not isinstance(raw.columns, pd.MultiIndex):
            return raw.copy()
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if symbol in level0:
            return raw[symbol].copy()
        if symbol in level1:
            return raw.xs(symbol, axis=1, level=1).copy()
        return pd.DataFrame()

    @staticmethod
    def _build_us_price_metric(
        frame: pd.DataFrame,
        *,
        symbol: str,
        name: str,
        completed_through: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Normalize a verified daily ETF series into the US context contract."""
        if frame is None or frame.empty or "Close" not in frame.columns:
            return None
        normalized = frame.copy()
        normalized["Close"] = pd.to_numeric(normalized["Close"], errors="coerce")
        normalized = normalized.dropna(subset=["Close"])
        if completed_through:
            cutoff = datetime.fromisoformat(completed_through).date()
            index_dates = pd.to_datetime(normalized.index, errors="coerce")
            dates = pd.Series(index_dates, index=normalized.index).dt.date
            normalized = normalized.loc[dates <= cutoff]
        if len(normalized) < 2:
            return None
        latest = normalized.iloc[-1]
        previous = normalized.iloc[-2]
        close = float(latest["Close"])
        previous_close = float(previous["Close"])
        if close <= 0 or previous_close <= 0:
            return None

        volume = None
        average_volume_20 = None
        volume_ratio_20d = None
        volume_observations = 0
        if "Volume" in normalized.columns:
            volumes = pd.to_numeric(normalized["Volume"], errors="coerce")
            latest_volume = volumes.iloc[-1]
            history_volume = volumes.iloc[:-1].dropna()
            history_volume = history_volume[history_volume > 0].tail(20)
            volume_observations = len(history_volume)
            if pd.notna(latest_volume) and float(latest_volume) > 0:
                volume = float(latest_volume)
            if len(history_volume) >= 15 and float(history_volume.mean()) > 0:
                average_volume_20 = float(history_volume.mean())
                if volume is not None:
                    volume_ratio_20d = volume / average_volume_20

        return {
            "symbol": symbol,
            "name": name,
            "last": close,
            "previous_close": previous_close,
            "change_pct": (close / previous_close - 1.0) * 100.0,
            "volume": volume,
            "average_volume_20": average_volume_20,
            "volume_ratio_20d": volume_ratio_20d,
            "volume_observations": volume_observations,
            "as_of": pd.Timestamp(normalized.index[-1]).date().isoformat(),
            "source": "Yahoo Finance/yfinance",
        }

    def _fetch_us_etf_metrics(self, yf) -> Dict[str, Dict[str, Any]]:
        try:
            from src.core.trading_calendar import get_effective_trading_date

            completed_through = get_effective_trading_date("us").isoformat()
        except Exception:
            completed_through = ""
        symbols = list(self._US_MARKET_PROXIES) + list(self._US_SECTOR_ETFS)
        raw = yf.download(
            symbols,
            period="3mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        metrics: Dict[str, Dict[str, Any]] = {}
        names = {**self._US_MARKET_PROXIES, **self._US_SECTOR_ETFS}
        for symbol in symbols:
            metric = self._build_us_price_metric(
                self._extract_yf_symbol_frame(raw, symbol),
                symbol=symbol,
                name=names[symbol],
                completed_through=completed_through,
            )
            if metric:
                metrics[symbol] = metric
        return metrics

    @classmethod
    def _fetch_fred_metric(
        cls,
        series_id: str,
        as_of: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Fetch recent official FRED observations without requiring an API key."""
        name, unit = cls._US_FRED_SERIES[series_id]
        start_date = (datetime.utcnow() - timedelta(days=21)).date().isoformat()
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={series_id}&cosd={start_date}"
        )
        request = Request(
            url,
            headers={"User-Agent": "daily-stock-analysis/market-review"},
        )
        with urlopen(request, timeout=8) as response:
            payload = response.read().decode("utf-8")
        rows = []
        for row in csv.DictReader(StringIO(payload)):
            raw_value = str(row.get(series_id) or "").strip()
            if not raw_value or raw_value == ".":
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            observation_date = str(
                row.get("observation_date")
                or row.get("DATE")
                or ""
            ).strip()
            if not observation_date:
                continue
            rows.append((observation_date, value))
        if as_of:
            rows = [row for row in rows if row[0] <= as_of]
        if len(rows) < 2:
            return None
        latest_date, latest_value = rows[-1]
        previous_date, previous_value = rows[-2]
        return {
            "series_id": series_id,
            "name": name,
            "value": latest_value,
            "previous_value": previous_value,
            "change": latest_value - previous_value,
            "unit": unit,
            "as_of": latest_date,
            "previous_as_of": previous_date,
            "source": "Federal Reserve Bank of St. Louis (FRED)",
            "url": f"https://fred.stlouisfed.org/series/{series_id}",
        }

    @classmethod
    def _fetch_treasury_yield_metrics(
        cls,
        as_of: str = "",
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch official 2Y/10Y par yields as a FRED-independent fallback.

        The Treasury feed is used only when a required FRED yield series is
        unavailable.  Values newer than the market evidence date are excluded
        so a delayed report cannot accidentally consume future observations.
        """
        try:
            cutoff = (
                datetime.fromisoformat(as_of).date()
                if as_of
                else datetime.utcnow().date()
            )
        except ValueError:
            cutoff = datetime.utcnow().date()

        rows: List[Dict[str, Any]] = []
        data_namespace = "http://schemas.microsoft.com/ado/2007/08/dataservices"
        metadata_namespace = (
            "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
        )

        first_of_month = cutoff.replace(day=1)
        previous_month = first_of_month - timedelta(days=1)
        for month in (cutoff, previous_month):
            month_key = month.strftime("%Y%m")
            url = (
                "https://home.treasury.gov/resource-center/data-chart-center/"
                "interest-rates/pages/xml?data=daily_treasury_yield_curve"
                f"&field_tdr_date_value_month={month_key}"
            )
            request = Request(
                url,
                headers={
                    "User-Agent": "daily-stock-analysis/market-review",
                    "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
                },
            )
            with urlopen(request, timeout=12) as response:
                payload = response.read()
            root = ElementTree.fromstring(payload)
            for properties in root.findall(
                f".//{{{metadata_namespace}}}properties"
            ):
                date_node = properties.find(f"{{{data_namespace}}}NEW_DATE")
                two_year_node = properties.find(f"{{{data_namespace}}}BC_2YEAR")
                ten_year_node = properties.find(f"{{{data_namespace}}}BC_10YEAR")
                if (
                    date_node is None
                    or two_year_node is None
                    or ten_year_node is None
                    or not date_node.text
                    or not two_year_node.text
                    or not ten_year_node.text
                ):
                    continue
                observation_date = date_node.text.strip()[:10]
                try:
                    parsed_date = datetime.fromisoformat(observation_date).date()
                    two_year = float(two_year_node.text.strip())
                    ten_year = float(ten_year_node.text.strip())
                except (TypeError, ValueError):
                    continue
                if parsed_date <= cutoff:
                    rows.append(
                        {
                            "date": parsed_date,
                            "DGS2": two_year,
                            "DGS10": ten_year,
                        }
                    )
            if len(rows) >= 2:
                break

        rows.sort(key=lambda item: item["date"])
        if len(rows) < 2:
            return {}

        previous = rows[-2]
        latest = rows[-1]
        source_url = (
            "https://home.treasury.gov/resource-center/data-chart-center/"
            "interest-rates/TextView"
            f"?type=daily_treasury_yield_curve&field_tdr_date_value={latest['date'].year}"
        )
        metrics: Dict[str, Dict[str, Any]] = {}
        for series_id in ("DGS2", "DGS10"):
            name, unit = cls._US_FRED_SERIES[series_id]
            latest_value = float(latest[series_id])
            previous_value = float(previous[series_id])
            metrics[series_id] = {
                "series_id": series_id,
                "name": name,
                "value": latest_value,
                "previous_value": previous_value,
                "change": latest_value - previous_value,
                "unit": unit,
                "as_of": latest["date"].isoformat(),
                "previous_as_of": previous["date"].isoformat(),
                "source": "U.S. Department of the Treasury",
                "url": source_url,
            }
        return metrics

    def get_us_market_context(self) -> Optional[Dict[str, Any]]:
        """Build strict US participation, sector, liquidity, and macro context."""
        import yfinance as yf

        metrics: Dict[str, Dict[str, Any]] = {}
        try:
            metrics = self._fetch_us_etf_metrics(yf)
        except Exception as exc:
            logger.warning("[Yfinance] 美股ETF批量行情失败: %s", exc)

        proxies = {
            symbol: metrics[symbol]
            for symbol in self._US_MARKET_PROXIES
            if symbol in metrics
        }
        sector_metrics = [
            metrics[symbol]
            for symbol in self._US_SECTOR_ETFS
            if symbol in metrics
        ]
        sector_metrics.sort(key=lambda item: float(item["change_pct"]), reverse=True)
        proxy_dates = {item["as_of"] for item in proxies.values()}
        latest_proxy_date = max(proxy_dates) if proxy_dates else ""

        macro: Dict[str, Dict[str, Any]] = {}
        for series_id in self._US_FRED_SERIES:
            try:
                metric = self._fetch_fred_metric(series_id, latest_proxy_date)
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                logger.warning("[FRED] 获取 %s 失败: %s", series_id, exc)
                metric = None
            if metric:
                macro[series_id] = metric

        missing_yields = [
            series_id for series_id in ("DGS2", "DGS10") if series_id not in macro
        ]
        if missing_yields:
            try:
                treasury_metrics = self._fetch_treasury_yield_metrics(latest_proxy_date)
            except (
                HTTPError,
                URLError,
                TimeoutError,
                OSError,
                ValueError,
                ElementTree.ParseError,
            ) as exc:
                logger.warning("[U.S. Treasury] 获取官方收益率曲线失败: %s", exc)
                treasury_metrics = {}
            for series_id in missing_yields:
                metric = treasury_metrics.get(series_id)
                if metric:
                    macro[series_id] = metric

        aligned_proxies = {
            symbol: item
            for symbol, item in proxies.items()
            if item["as_of"] == latest_proxy_date
        }
        aligned_sector_metrics = [
            item for item in sector_metrics if item["as_of"] == latest_proxy_date
        ]
        sector_advancers = sum(float(item["change_pct"]) > 0 for item in aligned_sector_metrics)
        sector_decliners = sum(float(item["change_pct"]) < 0 for item in aligned_sector_metrics)
        sector_flat = len(aligned_sector_metrics) - sector_advancers - sector_decliners

        spy = aligned_proxies.get("SPY")
        rsp = aligned_proxies.get("RSP")
        iwm = aligned_proxies.get("IWM")
        qqq = aligned_proxies.get("QQQ")
        participation = {
            "equal_weight_vs_cap_weight_pct": (
                float(rsp["change_pct"]) - float(spy["change_pct"])
                if rsp and spy else None
            ),
            "small_cap_vs_large_cap_pct": (
                float(iwm["change_pct"]) - float(spy["change_pct"])
                if iwm and spy else None
            ),
            "nasdaq100_vs_large_cap_pct": (
                float(qqq["change_pct"]) - float(spy["change_pct"])
                if qqq and spy else None
            ),
            "sector_advancers": sector_advancers,
            "sector_decliners": sector_decliners,
            "sector_flat": sector_flat,
            "sector_coverage": len(aligned_sector_metrics),
            "spy_volume_ratio_20d": spy.get("volume_ratio_20d") if spy else None,
            "as_of": latest_proxy_date,
            "method": (
                "SPY/RSP/IWM/QQQ relative performance plus 11 S&P sector ETFs; "
                "this is a transparent participation proxy, not exchange advance/decline counts."
            ),
        }

        proxy_ok = len(aligned_proxies) == len(self._US_MARKET_PROXIES)
        sector_ok = len(aligned_sector_metrics) >= 8
        macro_ok = "DGS2" in macro and "DGS10" in macro
        liquidity_ok = bool(
            spy
            and spy.get("volume_ratio_20d") is not None
            and int(spy.get("volume_observations") or 0) >= 15
            and 0.25 <= float(spy["volume_ratio_20d"]) <= 4.0
        )
        missing = []
        if not proxy_ok:
            missing.append("participation_proxies")
        if not liquidity_ok:
            missing.append("liquidity_proxy")
        if not sector_ok:
            missing.append("sector_etf_coverage")
        if not macro_ok:
            missing.append("treasury_yields")

        macro_sources = []
        for source_name in (
            "Federal Reserve Bank of St. Louis (FRED)",
            "U.S. Department of the Treasury",
        ):
            source_metrics = [
                item for item in macro.values() if item.get("source") == source_name
            ]
            if not source_metrics:
                continue
            macro_sources.append(
                {
                    "name": source_name,
                    "scope": ", ".join(
                        str(item.get("series_id") or "")
                        for item in source_metrics
                        if item.get("series_id")
                    ),
                    "as_of": max(
                        str(item.get("as_of") or "") for item in source_metrics
                    ),
                }
            )

        return {
            "as_of": latest_proxy_date,
            "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "proxies": aligned_proxies,
            "participation": participation,
            "sector_rankings": {
                "top": aligned_sector_metrics[:5],
                "bottom": list(reversed(aligned_sector_metrics[-5:])),
                "coverage": len(aligned_sector_metrics),
                "universe": len(self._US_SECTOR_ETFS),
            },
            "macro": macro,
            "quality": {
                "status": "ok" if not missing else "unavailable",
                "core_ready": not missing,
                "proxy_ready": proxy_ok,
                "liquidity_ready": liquidity_ok,
                "sector_ready": sector_ok,
                "macro_ready": macro_ok,
                "missing_core_fields": missing,
            },
            "sources": [
                {
                    "name": "Yahoo Finance/yfinance",
                    "scope": "ETF closes, volume, and sector rotation",
                    "as_of": latest_proxy_date,
                },
                *macro_sources,
            ],
        }

    def _get_hk_main_indices(self, yf) -> Optional[List[Dict[str, Any]]]:
        """获取港股主要指数行情（HSI、HSTECH、HSCEI），复用 _fetch_yf_ticker_data"""
        # Yahoo Finance 港股指数符号映射：
        # - HSI -> ^HSI
        # - HSTECH -> HSTECH.HK（不是 ^HSTECH）
        # - HSCEI -> ^HSCE（不是 ^HSCEI）
        # 该映射由离线单测 tests/test_yfinance_hk_indices.py 固化，避免在线依赖导致非确定性失败。
        hk_indices = {
            'HSI': ('^HSI', '恒生指数'),
            'HSTECH': ('HSTECH.HK', '恒生科技指数'),
            'HSCEI': ('^HSCE', '国企指数'),
        }
        results = []
        try:
            for code, (yf_symbol, name) in hk_indices.items():
                try:
                    item = self._fetch_yf_ticker_data(yf, yf_symbol, name, code)
                    if item:
                        results.append(item)
                        logger.debug(f"[Yfinance] 获取港股指数 {name} 成功")
                except Exception as e:
                    logger.warning(f"[Yfinance] 获取港股指数 {name} 失败: {e}")

            if results:
                logger.info(f"[Yfinance] 成功获取 {len(results)} 个港股指数行情")
                return results

        except Exception as e:
            logger.error(f"[Yfinance] 获取港股指数行情失败: {e}")

        return None

    def _get_jp_main_indices(self, yf) -> Optional[List[Dict[str, Any]]]:
        """获取日本主要指数行情（日经225、TOPIX），复用 _fetch_yf_ticker_data。"""
        jp_indices = {
            'N225': ('^N225', '日经225'),
            'TOPX': ('^TOPX', '东证指数'),
        }
        results = []
        try:
            for code, (yf_symbol, name) in jp_indices.items():
                try:
                    item = self._fetch_yf_ticker_data(yf, yf_symbol, name, code)
                    if item:
                        results.append(item)
                        logger.debug(f"[Yfinance] 获取日本指数 {name} 成功")
                except Exception as e:
                    logger.warning(f"[Yfinance] 获取日本指数 {name} 失败: {e}")
            if results:
                logger.info(f"[Yfinance] 成功获取 {len(results)} 个日本指数行情")
                return results
        except Exception as e:
            logger.error(f"[Yfinance] 获取日本指数行情失败: {e}")
        return None

    def _get_kr_main_indices(self, yf) -> Optional[List[Dict[str, Any]]]:
        """获取韩国主要指数行情（KOSPI、KOSDAQ），复用 _fetch_yf_ticker_data。"""
        kr_indices = {
            'KS11': ('^KS11', 'KOSPI'),
            'KQ11': ('^KQ11', 'KOSDAQ'),
        }
        results = []
        try:
            for code, (yf_symbol, name) in kr_indices.items():
                try:
                    item = self._fetch_yf_ticker_data(yf, yf_symbol, name, code)
                    if item:
                        results.append(item)
                        logger.debug(f"[Yfinance] 获取韩国指数 {name} 成功")
                except Exception as e:
                    logger.warning(f"[Yfinance] 获取韩国指数 {name} 失败: {e}")
            if results:
                logger.info(f"[Yfinance] 成功获取 {len(results)} 个韩国指数行情")
                return results
        except Exception as e:
            logger.error(f"[Yfinance] 获取韩国指数行情失败: {e}")
        return None

    def _get_tw_main_indices(self, yf) -> Optional[List[Dict[str, Any]]]:
        """获取台湾主要指数行情（加权指数 ^TWII、柜买指数 ^TWOII），复用 _fetch_yf_ticker_data。"""
        tw_indices = {
            'TWII': ('^TWII', '台湾加权指数'),
            'TWOII': ('^TWOII', '台湾柜买指数'),
        }
        results = []
        try:
            for code, (yf_symbol, name) in tw_indices.items():
                try:
                    item = self._fetch_yf_ticker_data(yf, yf_symbol, name, code)
                    if item:
                        results.append(item)
                        logger.debug(f"[Yfinance] 获取台湾指数 {name} 成功")
                except Exception as e:
                    logger.warning(f"[Yfinance] 获取台湾指数 {name} 失败: {e}")
            if results:
                logger.info(f"[Yfinance] 成功获取 {len(results)} 个台湾指数行情")
                return results
        except Exception as e:
            logger.error(f"[Yfinance] 获取台湾指数行情失败: {e}")
        return None

    def _is_us_stock(self, stock_code: str) -> bool:
        """
        判断代码是否为美股股票（排除美股指数）。

        委托给 us_index_mapping 模块的 is_us_stock_code()。
        """
        return is_us_stock_code(stock_code)

    def _get_us_stock_quote_from_stooq(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """
        使用 Stooq 为美股实时行情提供免密钥兜底。

        Stooq 提供的是最新交易日行情，精度不如分时实时接口，但在 Yahoo / yfinance
        被限流时，至少能为 Web UI 提供可用价格；若可获取到昨收价，则同时提供涨跌幅等衍生指标。
        """
        symbol = stock_code.strip().upper()
        stooq_symbol = f"{symbol.lower()}.us"
        url = f"https://stooq.com/q/l/?s={stooq_symbol}"
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; DSA/1.0; +https://github.com/ZhuLinsen/daily_stock_analysis)",
                "Accept": "text/plain,text/csv,*/*",
            },
        )

        try:
            with urlopen(request, timeout=15) as response:
                payload = response.read().decode("utf-8", "ignore").strip()
        except (HTTPError, URLError, TimeoutError) as exc:
            logger.warning(f"[Stooq] 获取美股 {symbol} 实时行情失败: {exc}")
            return None

        if not payload or payload.upper().startswith("NO DATA"):
            logger.warning(f"[Stooq] 无法获取 {symbol} 的行情数据")
            return None

        def _fetch_prev_close() -> Optional[float]:
            history_url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
            history_request = Request(
                history_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; DSA/1.0; +https://github.com/ZhuLinsen/daily_stock_analysis)",
                    "Accept": "text/plain,text/csv,*/*",
                },
            )
            try:
                with urlopen(history_request, timeout=15) as response:
                    history_payload = response.read().decode("utf-8", "ignore").strip()
            except (HTTPError, URLError, TimeoutError) as exc:
                logger.debug(f"[Stooq] 获取美股 {symbol} 日线历史失败: {exc}")
                return None

            if not history_payload or history_payload.upper().startswith("NO DATA"):
                return None

            try:
                reader = csv.reader(StringIO(history_payload))
                header = next(reader, None)
                if not header:
                    return None

                header_tokens = [cell.strip().lower() for cell in header]
                has_header = "close" in header_tokens and "date" in header_tokens
                if not has_header:
                    return None

                date_index = header_tokens.index("date")
                close_index = header_tokens.index("close")

                daily_rows: list[tuple[datetime, float]] = []
                for row in reader:
                    if not row:
                        continue
                    date_text = row[date_index].strip() if len(row) > date_index else ""
                    close_text = row[close_index].strip() if len(row) > close_index else ""
                    if not date_text or not close_text:
                        continue
                    try:
                        dt = datetime.strptime(date_text, "%Y-%m-%d")
                        close_val = float(close_text)
                    except Exception:
                        continue
                    daily_rows.append((dt, close_val))

                if len(daily_rows) < 2:
                    return None

                daily_rows.sort(key=lambda item: item[0])
                return daily_rows[-2][1]
            except Exception:
                return None

        try:
            reader = csv.reader(StringIO(payload))
            first_row = next(reader, None)
            if first_row is None:
                raise ValueError(f"unexpected Stooq payload: {payload}")

            normalized_first_row = [cell.strip() for cell in first_row]
            header_tokens = {cell.lower() for cell in normalized_first_row if cell}
            has_header = 'open' in header_tokens and 'close' in header_tokens
            row = next(reader, None) if has_header else first_row
            if row is None:
                raise ValueError(f"unexpected Stooq payload: {payload}")

            normalized_row = [cell.strip() for cell in row]
            while normalized_row and normalized_row[-1] == '':
                normalized_row.pop()

            if len(normalized_row) >= 8:
                open_index, high_index, low_index, price_index, volume_index = 3, 4, 5, 6, 7
            elif len(normalized_row) >= 7:
                open_index, high_index, low_index, price_index, volume_index = 2, 3, 4, 5, 6
            else:
                raise ValueError(f"unexpected Stooq payload: {payload}")

            open_price = float(normalized_row[open_index])
            high = float(normalized_row[high_index])
            low = float(normalized_row[low_index])
            price = float(normalized_row[price_index])
            volume = int(float(normalized_row[volume_index]))

            prev_close = _fetch_prev_close()
            change_amount = None
            change_pct = None
            amplitude = None
            if prev_close is not None and prev_close > 0:
                change_amount = price - prev_close
                change_pct = (change_amount / prev_close) * 100
                amplitude = ((high - low) / prev_close) * 100

            quote = UnifiedRealtimeQuote(
                code=symbol,
                name=STOCK_NAME_MAP.get(symbol, ''),
                source=RealtimeSource.STOOQ,
                price=price,
                change_pct=round(change_pct, 2) if change_pct is not None else None,
                change_amount=round(change_amount, 4) if change_amount is not None else None,
                volume=volume,
                amount=None,
                volume_ratio=None,
                turnover_rate=None,
                amplitude=round(amplitude, 2) if amplitude is not None else None,
                open_price=open_price,
                high=high,
                low=low,
                pre_close=prev_close,
                pe_ratio=None,
                pb_ratio=None,
                total_mv=None,
                circ_mv=None,
            )
            logger.info(f"[Stooq] 获取美股 {symbol} 兜底行情成功: 价格={price}")
            return quote
        except Exception as exc:
            logger.warning(f"[Stooq] 解析美股 {symbol} 行情失败: {exc}")
            return None

    def _get_us_index_realtime_quote(
        self,
        user_code: str,
        yf_symbol: str,
        index_name: str,
    ) -> Optional[UnifiedRealtimeQuote]:
        """
        Get realtime quote for US index (e.g. SPX -> ^GSPC).

        Args:
            user_code: User input code (e.g. SPX)
            yf_symbol: Yahoo Finance symbol (e.g. ^GSPC)
            index_name: Chinese name for the index

        Returns:
            UnifiedRealtimeQuote or None
        """
        import yfinance as yf

        try:
            logger.debug(f"[Yfinance] 获取美股指数 {user_code} ({yf_symbol}) 实时行情")
            ticker = yf.Ticker(yf_symbol)

            try:
                info = ticker.fast_info
                if info is None:
                    raise ValueError("fast_info is None")
                price = getattr(info, 'lastPrice', None) or getattr(info, 'last_price', None)
                prev_close = getattr(info, 'previousClose', None) or getattr(info, 'previous_close', None)
                open_price = getattr(info, 'open', None)
                high = getattr(info, 'dayHigh', None) or getattr(info, 'day_high', None)
                low = getattr(info, 'dayLow', None) or getattr(info, 'day_low', None)
                volume = getattr(info, 'lastVolume', None) or getattr(info, 'last_volume', None)
            except Exception:
                logger.debug("[Yfinance] fast_info 失败，尝试 history 方法")
                hist = ticker.history(period='2d')
                if hist.empty:
                    logger.warning(f"[Yfinance] 无法获取 {yf_symbol} 的数据")
                    return None
                today = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else today
                price = float(today['Close'])
                prev_close = float(prev['Close'])
                open_price = float(today['Open'])
                high = float(today['High'])
                low = float(today['Low'])
                volume = int(today['Volume'])

            change_amount = None
            change_pct = None
            if price is not None and prev_close is not None and prev_close > 0:
                change_amount = price - prev_close
                change_pct = (change_amount / prev_close) * 100

            amplitude = None
            if high is not None and low is not None and prev_close is not None and prev_close > 0:
                amplitude = ((high - low) / prev_close) * 100

            try:
                ticker_info = ticker.info or {}
            except Exception:
                ticker_info = {}
            missing_fields = [
                field
                for field, value in {
                    "price": price,
                    "prev_close": prev_close,
                    "volume": volume,
                    "amount": None,
                    "pe_ratio": None,
                    "pb_ratio": None,
                }.items()
                if value is None
            ]

            quote = UnifiedRealtimeQuote(
                code=user_code,
                name=index_name or user_code,
                source=RealtimeSource.FALLBACK,
                market="us",
                currency=str(ticker_info.get("currency") or "").upper() or None,
                data_quality="partial" if missing_fields else "ok",
                missing_fields=missing_fields or None,
                price=price,
                change_pct=round(change_pct, 2) if change_pct is not None else None,
                change_amount=round(change_amount, 4) if change_amount is not None else None,
                volume=volume,
                amount=None,
                volume_ratio=None,
                turnover_rate=None,
                amplitude=round(amplitude, 2) if amplitude is not None else None,
                open_price=open_price,
                high=high,
                low=low,
                pre_close=prev_close,
                pe_ratio=None,
                pb_ratio=None,
                total_mv=None,
                circ_mv=None,
            )
            logger.info(f"[Yfinance] 获取美股指数 {user_code} 实时行情成功: 价格={price}")
            return quote
        except Exception as e:
            logger.warning(f"[Yfinance] 获取美股指数 {user_code} 实时行情失败: {e}")
            return None

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """
        获取美股/美股指数实时行情数据

        支持美股股票（AAPL、TSLA）和美股指数（SPX、DJI 等）。
        数据来源：yfinance Ticker.info

        Args:
            stock_code: 美股代码或指数代码，如 'AMD', 'AAPL', 'SPX', 'DJI'

        Returns:
            UnifiedRealtimeQuote 对象，获取失败返回 None
        """
        import yfinance as yf

        # 美股指数：使用映射（SPX -> ^GSPC）
        yf_symbol, index_name = get_us_index_yf_symbol(stock_code)
        if yf_symbol:
            return self._get_us_index_realtime_quote(
                user_code=stock_code.strip().upper(),
                yf_symbol=yf_symbol,
                index_name=index_name,
            )

        # 仅处理美股股票或 JP/KR/TW suffix-only 股票
        if not (
            self._is_us_stock(stock_code)
            or self._is_jp_kr_suffix_stock(stock_code)
            or self._is_tw_suffix_stock(stock_code)
        ):
            logger.debug(f"[Yfinance] {stock_code} 不是美股或日韩 suffix 代码，跳过")
            return None

        try:
            symbol = self._convert_stock_code(stock_code)
            is_us_symbol = self._is_us_stock(symbol)
            suffix_market = get_suffix_market(symbol)
            logger.debug(f"[Yfinance] 获取 {symbol} 实时行情")

            ticker = yf.Ticker(symbol)

            # 尝试获取 fast_info（更快，但字段较少）
            try:
                info = ticker.fast_info
                if info is None:
                    raise ValueError("fast_info is None")

                price = getattr(info, 'lastPrice', None) or getattr(info, 'last_price', None)
                prev_close = getattr(info, 'previousClose', None) or getattr(info, 'previous_close', None)
                open_price = getattr(info, 'open', None)
                high = getattr(info, 'dayHigh', None) or getattr(info, 'day_high', None)
                low = getattr(info, 'dayLow', None) or getattr(info, 'day_low', None)
                volume = getattr(info, 'lastVolume', None) or getattr(info, 'last_volume', None)
                market_cap = getattr(info, 'marketCap', None) or getattr(info, 'market_cap', None)

            except Exception:
                # 回退到 history 方法获取最新数据
                logger.debug("[Yfinance] fast_info 失败，尝试 history 方法")
                hist = ticker.history(period='2d')
                if hist.empty:
                    if is_us_symbol:
                        logger.warning(f"[Yfinance] 无法获取 {symbol} 的数据，尝试 Stooq 兜底")
                        return self._get_us_stock_quote_from_stooq(symbol)
                    logger.warning(f"[Yfinance] 无法获取 {symbol} 的数据")
                    return None

                today = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else today

                price = float(today['Close'])
                prev_close = float(prev['Close'])
                open_price = float(today['Open'])
                high = float(today['High'])
                low = float(today['Low'])
                volume = int(today['Volume'])
                market_cap = None

            # 计算涨跌幅
            change_amount = None
            change_pct = None
            if price is not None and prev_close is not None and prev_close > 0:
                change_amount = price - prev_close
                change_pct = (change_amount / prev_close) * 100

            # 计算振幅
            amplitude = None
            if high is not None and low is not None and prev_close is not None and prev_close > 0:
                amplitude = ((high - low) / prev_close) * 100

            # 获取股票名称与 provider 元数据
            try:
                ticker_info = ticker.info or {}
            except Exception:
                ticker_info = {}
            try:
                info_name = ticker_info.get('shortName', '') or ticker_info.get('longName', '') or ''
                name = info_name if is_meaningful_stock_name(info_name, symbol) else STOCK_NAME_MAP.get(symbol, '')
            except Exception:
                name = STOCK_NAME_MAP.get(symbol, '')

            # 复用上方已获取的 ticker_info，无额外请求
            pe_ratio = _safe_float(ticker_info.get('trailingPE'))
            pb_ratio = _safe_float(ticker_info.get('priceToBook'))

            missing_fields = [
                field
                for field, value in {
                    "price": price,
                    "prev_close": prev_close,
                    "volume": volume,
                    "amount": None,
                    "pe_ratio": pe_ratio,
                    "pb_ratio": pb_ratio,
                }.items()
                if value is None
            ]
            quote = UnifiedRealtimeQuote(
                code=symbol,
                name=name,
                source=RealtimeSource.FALLBACK,
                market=suffix_market or ("us" if is_us_symbol else None),
                currency=str(ticker_info.get("currency") or "").upper() or None,
                data_quality="partial" if missing_fields else "ok",
                missing_fields=missing_fields or None,
                price=price,
                change_pct=round(change_pct, 2) if change_pct is not None else None,
                change_amount=round(change_amount, 4) if change_amount is not None else None,
                volume=volume,
                amount=None,  # yfinance 不直接提供成交额
                volume_ratio=None,
                turnover_rate=None,
                amplitude=round(amplitude, 2) if amplitude is not None else None,
                open_price=open_price,
                high=high,
                low=low,
                pre_close=prev_close,
                pe_ratio=pe_ratio,
                pb_ratio=pb_ratio,
                total_mv=market_cap,
                circ_mv=None,
            )

            logger.info(f"[Yfinance] 获取 {symbol} 实时行情成功: 价格={price}")
            return quote

        except Exception as e:
            if self._is_us_stock(stock_code):
                logger.warning(f"[Yfinance] 获取美股 {stock_code} 实时行情失败: {e}，尝试 Stooq 兜底")
                return self._get_us_stock_quote_from_stooq(stock_code)
            logger.warning(f"[Yfinance] 获取 {stock_code} 实时行情失败: {e}")
            return None


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)

    fetcher = YfinanceFetcher()

    try:
        df = fetcher.get_daily_data('600519')  # 茅台
        print(f"获取成功，共 {len(df)} 条数据")
        print(df.tail())
    except Exception as e:
        print(f"获取失败: {e}")
