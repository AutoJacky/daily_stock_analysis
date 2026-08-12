from datetime import date

import pandas as pd

from src.services.institutional_market_context import (
    InstitutionalMarketContextCollector,
    _percentile,
)


class FakeAk:
    def stock_index_pe_lg(self, symbol):
        return pd.DataFrame(
            {
                "日期": ["2021-08-11", "2026-08-10", "2026-08-11"],
                "滚动市盈率": [10.0, 12.0, 14.0],
            }
        )

    def stock_index_pb_lg(self, symbol):
        return pd.DataFrame(
            {
                "日期": ["2021-08-11", "2026-08-10", "2026-08-11"],
                "市净率": [1.0, 1.2, 1.4],
            }
        )

    def bond_zh_us_rate(self, start_date):
        return pd.DataFrame(
            {
                "日期": ["2026-08-11"],
                "中国国债收益率10年": [1.7],
                "美国国债收益率10年": [4.7],
            }
        )

    def macro_china_market_margin_sh(self):
        return pd.DataFrame(
            {"日期": ["2026-08-10", "2026-08-11"], "融资余额": [100e9, 110e9]}
        )

    def macro_china_market_margin_sz(self):
        return pd.DataFrame(
            {"日期": ["2026-08-10", "2026-08-11"], "融资余额": [200e9, 220e9]}
        )

    def stock_hsgt_hist_em(self, symbol):
        return pd.DataFrame(
            {"日期": ["2026-08-11"], "当日成交净买额": [float("nan")]}
        )

    def stock_industry_pe_ratio_cninfo(self, symbol, date):
        return pd.DataFrame(
            {
                "行业层级": [1.0, 1.0],
                "行业名称": ["制造业", "金融业"],
                "静态市盈率-加权平均": [20.0, 8.0],
            }
        )


def test_percentile_is_deterministic():
    assert _percentile([1, 2, 3, 4], 3) == 75.0


def test_cn_context_never_treats_missing_northbound_as_zero(monkeypatch):
    collector = InstitutionalMarketContextCollector(FakeAk())
    monkeypatch.setattr(
        collector,
        "_global_linkage",
        lambda as_of: {"status": "missing", "source": "test", "as_of": "", "data": {}},
    )
    context = collector.collect("cn", date(2026, 8, 12))

    assert context["valuation"]["hs300"]["data"]["pe_ttm"] == 14.0
    assert context["valuation"]["rates_erp"]["data"]["equity_risk_premium_proxy"] == 5.44
    assert context["capital_flow"]["margin"]["data"]["change_100m_cny"] == 300.0
    assert context["capital_flow"]["northbound"]["status"] == "not_supported"
    assert context["industry_valuation"]["data"]["lowest"][0]["name"] == "金融业"


def test_sina_global_fallback_preserves_dates_and_missing_changes(monkeypatch):
    payload = "\n".join(
        [
            'var hq_str_gb_$inx="S&P,7728.20,-0.32,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,Aug 11 04:34PM EDT,7753.11,0,1,2026";',
            'var hq_str_DINIW="17:19:26,99.8187,99.8187,99.8112,1116,99.8243,99.9018,99.7902,99.8187,DXY,2026-08-12";',
            'var hq_str_hf_GC="4474.397,,4474.000,4474.400,4477.400,4421.400,17:19:28,4441.100,4430.000,0,1,5,2026-08-12,Gold,0";',
        ]
    ).encode("gb18030")

    class Response:
        content = payload

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: Response())
    block = InstitutionalMarketContextCollector._sina_global_linkage(
        date(2026, 8, 12)
    )

    assert block["status"] == "partial"
    assert block["data"]["sp500"]["as_of"] == "2026-08-11"
    assert block["data"]["sp500"]["change_pct"] == -0.32
    assert block["data"]["dxy"]["change_pct"] is None
    assert block["data"]["gold"]["as_of"] == "2026-08-12"
