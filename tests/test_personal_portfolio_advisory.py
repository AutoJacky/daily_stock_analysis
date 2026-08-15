import base64
import json
import math
from datetime import date

import pandas as pd
import pytest

from src.services.personal_portfolio_advisory import (
    PersonalPortfolioError,
    build_personal_portfolio_advisory,
    load_private_portfolio_config,
    render_private_portfolio_advisory,
)


AS_OF = date(2026, 8, 14)


def _config() -> dict:
    assets = [
        {
            "id": "cash",
            "asset_type": "cash_management",
            "allocation_bucket": "cny_cash_management",
            "base_value_cny": 100,
        },
        {
            "id": "usd",
            "asset_type": "wealth_management",
            "allocation_bucket": "usd_fixed_income",
            "base_value_cny": 600,
            "redeemable_date": "2026-08-17",
        },
        {
            "id": "gold",
            "asset_type": "gold",
            "allocation_bucket": "gold",
            "base_value_cny": 100,
        },
    ]
    for index in range(10):
        assets.append(
            {
                "id": f"fund-{index}",
                "asset_type": "fund",
                "allocation_bucket": "funds",
                "fund_code": f"{index + 1:06d}",
                "name": f"{'US' if index % 2 == 0 else 'CN'} Fund {index}",
                "market_scope": "us" if index % 2 == 0 else "cn",
                "exposure_group": "nasdaq100" if index in {0, 2} else f"group-{index}",
                "base_value_cny": 20,
                "holding_pnl_cny": -2,
            }
        )
    return {
        "version": 1,
        "snapshot_date": AS_OF.isoformat(),
        "cash_balance_known": False,
        "risk_profile": "balanced_provisional",
        "usd_fixed_income_cap_pct": 50,
        "assets": assets,
        "data_gaps": ["活期现金余额未知。"],
    }


def _fetcher(code: str, as_of: date) -> dict:
    index = int(code) - 1
    # Different inception dates reproduce the real cross-fund alignment case.
    dates = pd.bdate_range(end=as_of.isoformat(), periods=300 + index * 3).date
    values = []
    level = 1.0
    drift = (index - 4.5) * 0.00025
    for step in range(len(dates)):
        daily_return = drift + 0.003 * math.sin(step * 0.31 + index)
        level *= 1 + daily_return
        values.append(level)
    series = pd.Series(values, index=dates, dtype=float)
    return {"unit": series, "adjusted": series}


def test_loader_accepts_wrapped_base64_and_rejects_duplicate_ids():
    encoded = base64.b64encode(json.dumps(_config()).encode()).decode()
    wrapped = "\n".join(encoded[index : index + 60] for index in range(0, len(encoded), 60))
    assert load_private_portfolio_config(wrapped)["version"] == 1

    invalid = _config()
    invalid["assets"][1]["id"] = "cash"
    encoded_invalid = base64.b64encode(json.dumps(invalid).encode()).decode()
    with pytest.raises(PersonalPortfolioError, match="id"):
        load_private_portfolio_config(encoded_invalid)


def test_builds_specific_account_actions_and_factor_diagnostic():
    advisory = build_personal_portfolio_advisory(
        _config(), as_of=AS_OF, fetcher=_fetcher, max_workers=2
    )

    assert advisory["known_total_cny"] == pytest.approx(1000.0)
    assert advisory["account_actions"]["usd_fixed_income"]["reduce_cny"] == 100
    assert advisory["fund_coverage"] == {
        "configured": 10,
        "valid": 10,
        "failed_codes": [],
    }
    buckets = {row["bucket"] for row in advisory["fund_metrics"]}
    assert {"top20", "middle60", "bottom20"} <= buckets
    assert advisory["factor_diagnostic"]["status"] == "ok"
    assert advisory["factor_diagnostic"]["observations"] >= 12
    assert advisory["factor_diagnostic"]["observations"] <= 104
    assert all(
        row["amount_cny"] == 0
        for row in advisory["fund_metrics"]
        if row["bucket"] == "top20"
    )
    assert all(
        row["amount_cny"] > 0
        for row in advisory["fund_metrics"]
        if row["bucket"] == "bottom20"
    )


def test_report_scope_and_private_data_do_not_cross_market_sections():
    advisory = build_personal_portfolio_advisory(
        _config(), as_of=AS_OF, fetcher=_fetcher, max_workers=2
    )

    us_report = render_private_portfolio_advisory(advisory, "us")
    cn_report = render_private_portfolio_advisory(advisory, "cn")
    assert "US Fund 0" in us_report
    assert "CN Fund 1" not in us_report
    assert "CN Fund 1" in cn_report
    assert "US Fund 0" not in cn_report
    assert "即时加仓金额固定为0" in us_report
    assert "重复暴露合并清单" in us_report


def test_refuses_to_backcast_holdings_before_snapshot():
    with pytest.raises(PersonalPortfolioError, match="早于持仓快照日"):
        build_personal_portfolio_advisory(
            _config(), as_of=date(2026, 8, 13), fetcher=_fetcher, max_workers=1
        )


def test_allows_weekend_snapshot_for_previous_friday_report():
    config = _config()
    config["snapshot_date"] = "2026-08-15"
    advisory = build_personal_portfolio_advisory(
        config, as_of=date(2026, 8, 14), fetcher=_fetcher, max_workers=2
    )
    assert advisory["snapshot_date"] == "2026-08-15"
    assert advisory["as_of"] == "2026-08-14"
