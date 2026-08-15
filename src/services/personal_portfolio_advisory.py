"""Private, deterministic portfolio advice for the final WeChat report.

The holdings payload is read from an encrypted environment variable.  Exact
holdings are deliberately kept out of model prompts, repository reports and
logs.  Public fund NAV history is used only for deterministic factor metrics.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import pandas as pd
import requests


class PersonalPortfolioError(ValueError):
    """Raised when the encrypted private holdings payload is invalid."""


@dataclass
class FundMetric:
    asset_id: str
    fund_code: str
    name: str
    market_scope: str
    exposure_group: str
    current_value_cny: float
    current_pnl_cny: Optional[float]
    nav_date: str
    latest_nav: float
    ma20: Optional[float]
    ma60: Optional[float]
    momentum20_pct: Optional[float]
    momentum60_pct: Optional[float]
    volatility60_pct: Optional[float]
    sharpe120: Optional[float]
    max_drawdown120_pct: Optional[float]
    factor_score: Optional[float] = None
    factor_bucket: str = "insufficient"
    adjusted_nav: Optional[pd.Series] = None


def load_private_portfolio_config(encoded: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Decode the private JSON payload without logging its contents."""

    encoded = (encoded if encoded is not None else os.getenv("PERSONAL_PORTFOLIO_B64", "")).strip()
    if not encoded:
        return None
    try:
        compact = re.sub(r"\s+", "", encoded)
        payload = json.loads(base64.b64decode(compact, validate=True).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PersonalPortfolioError("个人组合加密配置无法解析") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise PersonalPortfolioError("个人组合配置版本无效")
    try:
        date.fromisoformat(str(payload["snapshot_date"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise PersonalPortfolioError("个人组合缺少有效 snapshot_date") from exc
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise PersonalPortfolioError("个人组合 assets 为空")
    seen: set[str] = set()
    for item in assets:
        if not isinstance(item, dict):
            raise PersonalPortfolioError("个人组合资产记录格式无效")
        asset_id = str(item.get("id") or "").strip()
        if not asset_id or asset_id in seen:
            raise PersonalPortfolioError("个人组合资产 id 缺失或重复")
        seen.add(asset_id)
        value = _number(item.get("base_value_cny"))
        if value is None or value < 0:
            raise PersonalPortfolioError("个人组合资产人民币市值无效")
        if item.get("asset_type") == "fund" and not re.fullmatch(
            r"\d{6}", str(item.get("fund_code") or "")
        ):
            raise PersonalPortfolioError("基金资产缺少有效六位基金代码")
    return payload


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _eastmoney_fund_nav(fund_code: str, as_of: date, timeout: float = 15.0) -> Dict[str, pd.Series]:
    """Fetch NAV with bounded retries and an independent Eastmoney endpoint fallback."""

    headers = {"Referer": "https://fund.eastmoney.com/", "User-Agent": "Mozilla/5.0"}
    text = ""
    primary_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = requests.get(
                f"https://fund.eastmoney.com/pingzhongdata/{fund_code}.js",
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            text = response.text
            break
        except requests.RequestException as exc:
            primary_error = exc
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))

    def parse(variable: str, value_index: int) -> pd.Series:
        match = re.search(rf"var\s+{variable}\s*=\s*(\[.*?\]);", text, re.S)
        if match is None:
            return pd.Series(dtype=float)
        rows = json.loads(match.group(1))
        values: Dict[date, float] = {}
        for row in rows:
            if isinstance(row, dict):
                timestamp = row.get("x")
                raw_value = row.get("y")
            elif isinstance(row, list) and len(row) > value_index:
                timestamp = row[0]
                raw_value = row[value_index]
            else:
                continue
            value = _number(raw_value)
            stamp = _number(timestamp)
            if value is None or stamp is None or value <= 0:
                continue
            item_date = pd.to_datetime(stamp, unit="ms", utc=True).tz_convert("Asia/Shanghai").date()
            if item_date <= as_of:
                values[item_date] = value
        if not values:
            return pd.Series(dtype=float)
        return pd.Series(values, dtype=float).sort_index()

    unit = parse("Data_netWorthTrend", 1)
    # Data_ACWorthTrend is a sparse dividend/cumulative-value chart for many
    # funds, not a daily total-return series.  Build a daily wealth index from
    # the provider's equityReturn field and fall back to unit-NAV pct_change
    # only for an individual missing day.
    daily_returns: Dict[date, float] = {}
    trend_match = re.search(r"var\s+Data_netWorthTrend\s*=\s*(\[.*?\]);", text, re.S)
    if trend_match is not None:
        for row in json.loads(trend_match.group(1)):
            if not isinstance(row, dict):
                continue
            stamp = _number(row.get("x"))
            item_return = _number(row.get("equityReturn"))
            if stamp is None or item_return is None:
                continue
            item_date = pd.to_datetime(stamp, unit="ms", utc=True).tz_convert(
                "Asia/Shanghai"
            ).date()
            if item_date <= as_of:
                daily_returns[item_date] = item_return / 100.0
    if len(unit) >= 60:
        unit_returns = unit.pct_change()
        total_returns = pd.Series(daily_returns, dtype=float).sort_index().reindex(unit.index)
        total_returns = total_returns.fillna(unit_returns).fillna(0.0)
        adjusted = (1.0 + total_returns).cumprod()
        return {"unit": unit, "adjusted": adjusted}

    # The historical-NAV API is structurally independent of the chart JavaScript.
    try:
        response = requests.get(
            "https://api.fund.eastmoney.com/f10/lsjz",
            params={
                "fundCode": fund_code,
                "pageIndex": 1,
                "pageSize": 200,
                "startDate": "",
                "endDate": as_of.isoformat(),
            },
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        rows = ((response.json().get("Data") or {}).get("LSJZList") or [])
        unit_values: Dict[date, float] = {}
        adjusted_values: Dict[date, float] = {}
        for row in rows:
            try:
                item_date = date.fromisoformat(str(row.get("FSRQ")))
            except (TypeError, ValueError):
                continue
            unit_value = _number(row.get("DWJZ"))
            adjusted_value = _number(row.get("LJJZ"))
            if unit_value is not None and unit_value > 0:
                unit_values[item_date] = unit_value
            if adjusted_value is not None and adjusted_value > 0:
                adjusted_values[item_date] = adjusted_value
        fallback_unit = pd.Series(unit_values, dtype=float).sort_index()
        fallback_adjusted = pd.Series(adjusted_values, dtype=float).sort_index()
        if len(fallback_unit) >= 60:
            return {
                "unit": fallback_unit,
                "adjusted": fallback_adjusted if len(fallback_adjusted) >= 2 else fallback_unit.copy(),
            }
    except (requests.RequestException, ValueError):
        pass
    if primary_error is not None:
        raise PersonalPortfolioError(f"基金 {fund_code} 两条净值链路均失败") from primary_error
    raise PersonalPortfolioError(f"基金 {fund_code} 两条净值链路数据不足")


def _trailing_return(series: pd.Series, periods: int) -> Optional[float]:
    if len(series) <= periods or series.iloc[-periods - 1] <= 0:
        return None
    return float(series.iloc[-1] / series.iloc[-periods - 1] - 1.0)


def _metric_for_fund(
    asset: Mapping[str, Any],
    snapshot_date: date,
    as_of: date,
    fetcher: Callable[[str, date], Dict[str, pd.Series]],
) -> FundMetric:
    code = str(asset["fund_code"])
    history = fetcher(code, as_of)
    unit = history.get("unit", pd.Series(dtype=float)).dropna().sort_index()
    adjusted = history.get("adjusted", unit).dropna().sort_index()
    if unit.empty or adjusted.empty:
        raise PersonalPortfolioError(f"基金 {code} 净值为空")
    snapshot_rows = unit[unit.index <= snapshot_date]
    if snapshot_rows.empty:
        raise PersonalPortfolioError(f"基金 {code} 缺少持仓快照日之前净值")
    snapshot_nav = float(snapshot_rows.iloc[-1])
    latest_nav = float(unit.iloc[-1])
    snapshot_value = float(asset["base_value_cny"])
    estimated_units = snapshot_value / snapshot_nav
    current_value = estimated_units * latest_nav
    snapshot_pnl = _number(asset.get("holding_pnl_cny"))
    cost = snapshot_value - snapshot_pnl if snapshot_pnl is not None else None
    returns = adjusted.pct_change().replace([math.inf, -math.inf], pd.NA).dropna()
    returns60 = returns.iloc[-60:]
    returns120 = returns.iloc[-120:]
    volatility = (
        float(returns60.std(ddof=1) * math.sqrt(252.0))
        if len(returns60) >= 40 and returns60.std(ddof=1) > 0
        else None
    )
    sharpe = None
    if len(returns120) >= 60 and returns120.std(ddof=1) > 0:
        daily_rf = 0.015 / 252.0
        sharpe = float((returns120.mean() - daily_rf) / returns120.std(ddof=1) * math.sqrt(252.0))
    window = adjusted.iloc[-120:]
    drawdown = None
    if len(window) >= 40:
        drawdown = float((window / window.cummax() - 1.0).min())
    return FundMetric(
        asset_id=str(asset["id"]),
        fund_code=code,
        name=str(asset.get("name") or code),
        market_scope=str(asset.get("market_scope") or "cn"),
        exposure_group=str(asset.get("exposure_group") or "other"),
        current_value_cny=round(current_value, 2),
        current_pnl_cny=round(current_value - cost, 2) if cost is not None else None,
        nav_date=unit.index[-1].isoformat(),
        latest_nav=latest_nav,
        ma20=float(unit.iloc[-20:].mean()) if len(unit) >= 20 else None,
        ma60=float(unit.iloc[-60:].mean()) if len(unit) >= 60 else None,
        momentum20_pct=(lambda value: value * 100.0 if value is not None else None)(
            _trailing_return(adjusted, 20)
        ),
        momentum60_pct=(lambda value: value * 100.0 if value is not None else None)(
            _trailing_return(adjusted, 60)
        ),
        volatility60_pct=volatility * 100.0 if volatility is not None else None,
        sharpe120=sharpe,
        max_drawdown120_pct=drawdown * 100.0 if drawdown is not None else None,
        adjusted_nav=adjusted,
    )


def _rank_metrics(metrics: List[FundMetric]) -> None:
    eligible = [
        item
        for item in metrics
        if all(
            value is not None
            for value in (
                item.momentum20_pct,
                item.momentum60_pct,
                item.volatility60_pct,
                item.sharpe120,
                item.max_drawdown120_pct,
            )
        )
    ]
    if not eligible:
        return

    def percentiles(values: Iterable[float], higher_is_better: bool = True) -> Dict[int, float]:
        series = pd.Series(list(values), dtype=float)
        ranked = series.rank(method="average", pct=True, ascending=higher_is_better) * 100.0
        return {index: float(value) for index, value in ranked.items()}

    mom20 = percentiles(item.momentum20_pct for item in eligible)
    mom60 = percentiles(item.momentum60_pct for item in eligible)
    sharpe = percentiles(item.sharpe120 for item in eligible)
    drawdown = percentiles(item.max_drawdown120_pct for item in eligible)
    low_vol = percentiles(
        (item.volatility60_pct for item in eligible), higher_is_better=False
    )
    for index, item in enumerate(eligible):
        item.factor_score = round(
            0.25 * mom20[index]
            + 0.25 * mom60[index]
            + 0.25 * sharpe[index]
            + 0.15 * drawdown[index]
            + 0.10 * low_vol[index],
            1,
        )
    ordered = sorted(eligible, key=lambda item: item.factor_score or 0.0, reverse=True)
    bucket_size = max(1, math.ceil(len(ordered) * 0.20))
    for item in ordered[:bucket_size]:
        item.factor_bucket = "top20"
    for item in ordered[-bucket_size:]:
        item.factor_bucket = "bottom20"
    for item in ordered[bucket_size:-bucket_size]:
        item.factor_bucket = "middle60"


def _long_short_diagnostic(metrics: List[FundMetric]) -> Dict[str, Any]:
    series = {
        item.fund_code: item.adjusted_nav
        for item in metrics
        if item.adjusted_nav is not None and len(item.adjusted_nav) >= 80
    }
    if len(series) < 5:
        return {"status": "insufficient", "observations": 0}
    frame = pd.concat(series, axis=1).sort_index()
    frame.index = pd.to_datetime(frame.index)
    weekly = frame.resample("W-FRI").last().ffill(limit=1)
    spreads: List[float] = []
    rebalance_dates: List[str] = []
    for index in range(26, len(weekly) - 1):
        history = weekly.iloc[: index + 1]
        next_returns = weekly.iloc[index + 1] / weekly.iloc[index] - 1.0
        rows: List[Dict[str, float | str]] = []
        for code in weekly.columns:
            values = history[code].dropna()
            if len(values) < 27 or code not in next_returns.index:
                continue
            future_return = _number(next_returns.get(code))
            if future_return is None:
                continue
            returns = values.pct_change().dropna()
            returns12 = returns.iloc[-12:]
            returns26 = returns.iloc[-26:]
            if len(returns12) < 10 or len(returns26) < 20:
                continue
            std12 = float(returns12.std(ddof=1))
            std26 = float(returns26.std(ddof=1))
            if std12 <= 0 or std26 <= 0:
                continue
            window = values.iloc[-26:]
            rows.append(
                {
                    "code": str(code),
                    "future_return": future_return,
                    "momentum4": float(values.iloc[-1] / values.iloc[-5] - 1.0),
                    "momentum12": float(values.iloc[-1] / values.iloc[-13] - 1.0),
                    "sharpe26": float(
                        (returns26.mean() - 0.015 / 52.0) / std26 * math.sqrt(52.0)
                    ),
                    "drawdown26": float((window / window.cummax() - 1.0).min()),
                    "volatility12": std12 * math.sqrt(52.0),
                }
            )
        if len(rows) < 5:
            continue
        ranking = pd.DataFrame(rows)
        ranking["score"] = (
            ranking["momentum4"].rank(pct=True) * 0.25
            + ranking["momentum12"].rank(pct=True) * 0.25
            + ranking["sharpe26"].rank(pct=True) * 0.25
            + ranking["drawdown26"].rank(pct=True) * 0.15
            + ranking["volatility12"].rank(pct=True, ascending=False) * 0.10
        )
        ranking = ranking.sort_values("score", ascending=False)
        bucket_size = max(1, math.ceil(len(ranking) * 0.20))
        top_return = float(ranking.head(bucket_size)["future_return"].mean())
        bottom_return = float(ranking.tail(bucket_size)["future_return"].mean())
        if math.isfinite(top_return) and math.isfinite(bottom_return):
            spreads.append(top_return - bottom_return)
            rebalance_dates.append(weekly.index[index].date().isoformat())
    if len(spreads) < 12:
        return {"status": "insufficient", "observations": len(spreads)}
    spread = pd.Series(spreads, index=rebalance_dates, dtype=float).iloc[-104:]
    std = float(spread.std(ddof=1))
    sharpe = float(spread.mean() / std * math.sqrt(52.0)) if std > 0 else None
    return {
        "status": "ok" if sharpe is not None else "insufficient",
        "observations": len(spread),
        "annualized_sharpe": round(sharpe, 2) if sharpe is not None else None,
        "annualized_return_pct": round(float(spread.mean() * 52.0 * 100.0), 2),
        "annualized_volatility_pct": round(std * math.sqrt(52.0) * 100.0, 2),
        "method": "每周仅用当时可见的净值因子排序，计算下一周前20%减后20%的实际涨跌差；最多保留近104周滚动样本外诊断，不是真实做空收益",
    }


def build_personal_portfolio_advisory(
    config: Mapping[str, Any],
    *,
    as_of: date,
    fetcher: Callable[[str, date], Dict[str, pd.Series]] = _eastmoney_fund_nav,
    max_workers: int = 6,
) -> Dict[str, Any]:
    """Refresh public NAVs and build a deterministic account-level decision."""

    snapshot_date = date.fromisoformat(str(config["snapshot_date"]))
    if as_of < snapshot_date:
        raise PersonalPortfolioError("报告交易日早于持仓快照日，禁止倒推个人仓位")
    assets = list(config["assets"])
    fund_assets = [item for item in assets if item.get("asset_type") == "fund"]
    metrics: List[FundMetric] = []
    failed_assets: List[Mapping[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, 8))) as executor:
        futures = {
            executor.submit(_metric_for_fund, item, snapshot_date, as_of, fetcher): item
            for item in fund_assets
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                metrics.append(future.result())
            except Exception:
                failed_assets.append(item)
    # A failed concurrent batch is retried serially once.  This is deliberately
    # inside the same run so a transient provider throttle does not become a
    # permanent "data missing" item in the WeChat report.
    failures: List[str] = []
    for item in failed_assets:
        try:
            metrics.append(_metric_for_fund(item, snapshot_date, as_of, fetcher))
        except Exception:
            failures.append(str(item.get("fund_code") or item.get("id")))
    _rank_metrics(metrics)
    factor_diagnostic = _long_short_diagnostic(metrics)
    factor_sharpe = _number(factor_diagnostic.get("annualized_sharpe"))
    factor_add_enabled = factor_diagnostic.get("status") == "ok" and bool(
        factor_sharpe is not None and factor_sharpe > 0
    )

    refreshed_values = {item.asset_id: item.current_value_cny for item in metrics}
    total = sum(
        refreshed_values.get(str(item["id"]), float(item["base_value_cny"]))
        for item in assets
    )
    if total <= 0:
        raise PersonalPortfolioError("个人组合可核验总额必须大于0")
    category_values: Dict[str, float] = {}
    for item in assets:
        category = str(item.get("allocation_bucket") or item.get("asset_type") or "other")
        value = refreshed_values.get(str(item["id"]), float(item["base_value_cny"]))
        category_values[category] = category_values.get(category, 0.0) + value
    allocations = {
        key: {"value_cny": round(value, 2), "weight_pct": round(value / total * 100.0, 2)}
        for key, value in category_values.items()
    }

    cash_known = bool(config.get("cash_balance_known"))
    usd_fixed_value = category_values.get("usd_fixed_income", 0.0)
    usd_cap_pct = float(config.get("usd_fixed_income_cap_pct", 50.0))
    usd_reduce = max(0.0, usd_fixed_value - total * usd_cap_pct / 100.0)
    redeemable_dates = sorted(
        str(item.get("redeemable_date"))
        for item in assets
        if item.get("allocation_bucket") == "usd_fixed_income" and item.get("redeemable_date")
    )
    gold_fund_value = sum(
        item.current_value_cny for item in metrics if item.exposure_group == "gold"
    )
    gold_exposure_value = category_values.get("gold", 0.0) + gold_fund_value
    gold_weight = gold_exposure_value / total * 100.0
    fund_weight = allocations.get("funds", {}).get("weight_pct", 0.0)

    metric_groups: Dict[str, List[FundMetric]] = {}
    for item in metrics:
        metric_groups.setdefault(item.exposure_group, []).append(item)
    preferred_by_group = {
        group: max(
            rows,
            key=lambda row: row.factor_score if row.factor_score is not None else -1,
        ).asset_id
        for group, rows in metric_groups.items()
        if len(rows) > 1
    }

    metric_rows = []
    for item in sorted(
        metrics,
        key=lambda row: row.factor_score if row.factor_score is not None else -1,
        reverse=True,
    ):
        upper_buy = item.ma20 * 1.02 if item.ma20 is not None else None
        risk_line = item.ma60 * 0.97 if item.ma60 is not None else None
        if item.factor_bucket == "bottom20":
            amount = round(item.current_value_cny * 0.25)
            action = "减少/合并"
            trigger = (
                f"下一可赎回窗口先减少约{amount:.0f}元；若净值连续2个披露日低于{risk_line:.4f}，再减少同额（累计约50%）"
                if risk_line is not None
                else f"下一可赎回窗口先减少约{amount:.0f}元；趋势数据恢复前不新增"
            )
        elif item.factor_bucket == "top20":
            planned_amount = min(total * 0.005, 500.0)
            is_duplicate_secondary = (
                item.exposure_group in preferred_by_group
                and preferred_by_group[item.exposure_group] != item.asset_id
            )
            amount = (
                0.0
                if not cash_known or not factor_add_enabled or is_duplicate_secondary
                else planned_amount
            )
            action = "持有/合并" if is_duplicate_secondary else "持有/条件加仓"
            if is_duplicate_secondary:
                preferred = next(
                    row.name
                    for row in metric_groups[item.exposure_group]
                    if row.asset_id == preferred_by_group[item.exposure_group]
                )
                trigger = f"当前新增0元；同类暴露优先保留“{preferred}”，核对赎回费后再合并"
            elif upper_buy is not None:
                if not factor_add_enabled:
                    trigger = "因子样本外Sharpe未为正，当前新增0元，仅持有观察"
                elif not cash_known:
                    trigger = (
                        f"当前新增0元；现金缓冲确认后，净值站稳MA20且不高于{upper_buy:.4f}时每批上限{planned_amount:.0f}元"
                    )
                else:
                    trigger = f"净值连续2个披露日站稳MA20且不高于{upper_buy:.4f}时每批最多{amount:.0f}元"
            else:
                trigger = "缺少MA20，不执行新增"
        else:
            amount = 0.0
            action = "持有观察"
            trigger = (
                f"净值跌破{risk_line:.4f}后重新评估，不因浮亏机械补仓"
                if risk_line is not None
                else "数据不足，不新增"
            )
        metric_rows.append(
            {
                "fund_code": item.fund_code,
                "name": item.name,
                "market_scope": item.market_scope,
                "exposure_group": item.exposure_group,
                "value_cny": item.current_value_cny,
                "pnl_cny": item.current_pnl_cny,
                "nav": round(item.latest_nav, 4),
                "nav_date": item.nav_date,
                "ma20": round(item.ma20, 4) if item.ma20 is not None else None,
                "ma60": round(item.ma60, 4) if item.ma60 is not None else None,
                "score": item.factor_score,
                "bucket": item.factor_bucket,
                "action": action,
                "amount_cny": amount,
                "trigger": trigger,
            }
        )

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in metric_rows:
        groups.setdefault(row["exposure_group"], []).append(row)
    duplicates = []
    for group, rows in groups.items():
        if len(rows) < 2:
            continue
        ordered = sorted(rows, key=lambda row: row.get("score") or -1, reverse=True)
        duplicates.append(
            {
                "group": group,
                "count": len(rows),
                "preferred": ordered[0]["name"],
                "others": [row["name"] for row in ordered[1:]],
                "market_scopes": sorted({str(row["market_scope"]) for row in rows}),
            }
        )

    return {
        "as_of": as_of.isoformat(),
        "snapshot_date": snapshot_date.isoformat(),
        "snapshot_age_days": (as_of - snapshot_date).days,
        "known_total_cny": round(total, 2),
        "cash_balance_known": cash_known,
        "risk_profile": str(config.get("risk_profile") or "balanced_provisional"),
        "allocations": allocations,
        "account_actions": {
            "usd_fixed_income": {
                "current_weight_pct": round(usd_fixed_value / total * 100.0, 2),
                "cap_pct": usd_cap_pct,
                "reduce_cny": round(usd_reduce, -2),
                "earliest_redeemable_date": redeemable_dates[0] if redeemable_dates else "未取得",
            },
            "gold": {
                "current_weight_pct": gold_weight,
                "action": "不加仓，持有并再平衡" if gold_weight >= 10.0 else "持有观察",
            },
            "funds": {
                "current_weight_pct": fund_weight,
                "action": "现金余额未确认前不扩大基金总仓位" if not cash_known else "按因子分批再平衡",
            },
        },
        "fund_metrics": metric_rows,
        "factor_diagnostic": factor_diagnostic,
        "duplicates": duplicates,
        "fund_coverage": {
            "configured": len(fund_assets),
            "valid": len(metrics),
            "failed_codes": sorted(failures),
        },
        "data_gaps": list(config.get("data_gaps") or []),
    }


def render_private_portfolio_advisory(advisory: Mapping[str, Any], market: str) -> str:
    """Render the private section that is sent only through PushPlus."""

    total = float(advisory.get("known_total_cny") or 0.0)
    allocations = advisory.get("allocations") if isinstance(advisory.get("allocations"), Mapping) else {}
    actions = advisory.get("account_actions") if isinstance(advisory.get("account_actions"), Mapping) else {}
    usd = actions.get("usd_fixed_income") if isinstance(actions.get("usd_fixed_income"), Mapping) else {}
    gold = actions.get("gold") if isinstance(actions.get("gold"), Mapping) else {}
    funds = actions.get("funds") if isinstance(actions.get("funds"), Mapping) else {}
    lines = [
        "# 🔐 你的账户级量化操作单",
        "",
        "> 本节来自 GitHub 加密持仓，只进入本次微信正文；不进入公开报告附件，也不发送给千问。",
        "",
        "## 🎯 现在最重要的三件事",
        "",
        f"1. **美元固收先降集中度**：当前约占 {usd.get('current_weight_pct', 0):.2f}%。在 {usd.get('earliest_redeemable_date', '未取得')} 可赎回时，建议至少不续作约 **{float(usd.get('reduce_cny') or 0):,.0f} 元**等值份额，使其回到不高于 {usd.get('cap_pct', 50):.0f}% 的临时上限；释放资金先留作人民币现金/短久期低风险资产，不直接追涨基金。",
        f"2. **黄金暂不补仓**：已知黄金约占 {float(gold.get('current_weight_pct') or 0):.2f}%，接近组合上限。亏损本身不是补仓理由，当前动作是“{gold.get('action', '持有观察')}”。",
        f"3. **基金先做减法**：基金合计约占 {float(funds.get('current_weight_pct') or 0):.2f}%。{funds.get('action', '持有观察')}；新增资金优先给高分、低重复的宽基，主题基金只作卫星仓。",
        "",
        "## 🧮 可核验资产结构",
        "",
        f"- 可核验总额：**{total:,.2f} 元**（不含未展示现金及未估值的额外1克如意金）。",
        f"- 持仓快照：**{advisory.get('snapshot_date')}**；本次数据日：**{advisory.get('as_of')}**，期间默认未发生申购、赎回或分红再投。",
    ]
    labels = {
        "cny_cash_management": "人民币现金管理理财",
        "usd_fixed_income": "美元固定收益理财",
        "funds": "公募基金",
        "gold": "积存金",
    }
    for key in ("cny_cash_management", "usd_fixed_income", "funds", "gold"):
        row = allocations.get(key) if isinstance(allocations.get(key), Mapping) else {}
        lines.append(
            f"- {labels[key]}：{float(row.get('value_cny') or 0):,.2f} 元｜{float(row.get('weight_pct') or 0):.2f}%"
        )

    scope = "us" if market == "us" else "cn"
    rows = [
        row
        for row in advisory.get("fund_metrics", [])
        if isinstance(row, Mapping) and str(row.get("market_scope")) in {scope, "all"}
    ]
    priority = {"bottom20": 0, "top20": 1, "middle60": 2, "insufficient": 3}
    rows.sort(key=lambda row: (priority.get(str(row.get("bucket")), 9), -(row.get("score") or 0)))
    lines.extend(["", f"## 📋 {'美股/QDII' if scope == 'us' else 'A股/商品'}基金操作", ""])
    for row in rows[:12]:
        score = "未评分" if row.get("score") is None else f"{float(row['score']):.1f}分"
        lines.append(
            f"- **{row.get('name')}（{row.get('fund_code')}）**：{row.get('action')}｜因子 {score}｜"
            f"最新净值 {float(row.get('nav') or 0):.4f}（{row.get('nav_date')}）｜{row.get('trigger')}"
        )

    duplicate_rows = [
        row
        for row in advisory.get("duplicates", [])
        if isinstance(row, Mapping)
        and bool({scope, "all"}.intersection(set(row.get("market_scopes") or [])))
    ]
    if duplicate_rows:
        lines.extend(["", "## 🧹 重复暴露合并清单", ""])
        for row in duplicate_rows[:5]:
            lines.append(
                f"- **{row.get('group')}** 共{row.get('count')}只：因子暂时领先为“{row.get('preferred')}”；"
                f"其余 {', '.join(row.get('others') or [])} 停止新增。A/C份额或同指数合并前先核对赎回费和持有期。"
            )

    diagnostic = advisory.get("factor_diagnostic") if isinstance(advisory.get("factor_diagnostic"), Mapping) else {}
    lines.extend(["", "## 📐 图中方法的实际检验", ""])
    if diagnostic.get("status") == "ok":
        lines.append(
            f"- 滚动周度前20%减后20%的样本外净值涨跌差：年化 Sharpe **{diagnostic.get('annualized_sharpe')}**，"
            f"年化价差收益 {diagnostic.get('annualized_return_pct')}%，年化波动 {diagnostic.get('annualized_volatility_pct')}%，"
            f"样本 {diagnostic.get('observations')} 个交易日。"
        )
        lines.append(f"- 口径：{diagnostic.get('method')}。")
    else:
        lines.append("- 有效共同净值样本不足，暂不展示组合 Sharpe，更不据此加仓。")

    coverage = advisory.get("fund_coverage") if isinstance(advisory.get("fund_coverage"), Mapping) else {}
    lines.extend(["", "## ✅ 数据护栏", ""])
    lines.append(
        f"- 基金净值覆盖：{coverage.get('valid', 0)}/{coverage.get('configured', 0)}；"
        f"失败代码：{', '.join(coverage.get('failed_codes') or []) or '无'}。"
    )
    if not advisory.get("cash_balance_known"):
        lines.append("- **现金余额未知，因此所有基金即时加仓金额固定为0；补齐现金与应急金目标后才会解锁。**")
    if int(advisory.get("snapshot_age_days") or 0) > 7:
        lines.append("- **持仓快照已超过7天；如期间有任何交易，必须先更新加密持仓，否则金额建议仅作失真预警。**")
    for gap in advisory.get("data_gaps", []):
        lines.append(f"- {gap}")
    lines.extend(
        [
            "",
            "> 操作金额是组合风险上限，不是自动交易指令；赎回费、持有期限、税务和产品说明书未核对前不执行。",
        ]
    )
    return "\n".join(lines).strip() + "\n"
