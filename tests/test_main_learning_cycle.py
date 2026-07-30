# -*- coding: utf-8 -*-
"""Tests for the deterministic daily prediction-feedback cycle."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import main


def _config(**overrides):
    values = {
        "backtest_enabled": True,
        "backtest_eval_window_days": 10,
        "backtest_min_age_days": 14,
        "agent_memory_enabled": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_auto_backtest_runs_multi_horizon_outcomes_and_builds_audit_report() -> None:
    backtest = MagicMock()
    backtest.run_backtest.return_value = {
        "processed": 4,
        "saved": 4,
        "completed": 4,
        "insufficient": 0,
        "errors": 0,
    }
    outcomes = MagicMock()
    outcomes.run_outcomes.return_value = {"evaluated": 8}
    outcomes.get_stats.return_value = {
        "total": 40,
        "completed": 32,
        "unable": 8,
        "hit": 20,
        "miss": 10,
        "hit_rate_pct": 66.67,
    }

    with patch(
        "src.services.backtest_service.BacktestService",
        return_value=backtest,
    ), patch(
        "src.services.decision_signal_outcome_service.DecisionSignalOutcomeService",
        return_value=outcomes,
    ):
        report = main._run_auto_backtest(_config())

    backtest.run_backtest.assert_called_once_with(
        force=False,
        eval_window_days=10,
        min_age_days=14,
        limit=200,
    )
    outcomes.run_outcomes.assert_called_once_with(
        horizons=["1d", "3d", "5d", "10d"],
        force=False,
        limit=500,
    )
    assert "66.7%" in report
    assert "模型不参与给自己打分" in report
    assert "仅依据客观后验下调/校准置信度" in report


def test_auto_backtest_is_fail_open() -> None:
    with patch(
        "src.services.backtest_service.BacktestService",
        side_effect=RuntimeError("database unavailable"),
    ):
        assert main._run_auto_backtest(_config()) == ""


def test_auto_backtest_persists_adaptive_governance_when_enabled() -> None:
    backtest = MagicMock()
    backtest.run_backtest.return_value = {
        "processed": 0,
        "saved": 0,
        "completed": 0,
        "insufficient": 0,
        "errors": 0,
    }
    backtest.get_global_summary.return_value = {
        "total_evaluations": 40,
        "direction_accuracy": 0.48,
    }
    outcomes = MagicMock()
    outcomes.run_outcomes.return_value = {"evaluated": 4}
    outcomes.get_stats.return_value = {
        "total": 40,
        "completed": 35,
        "unable": 5,
        "hit": 17,
        "miss": 18,
        "hit_rate_pct": 48.57,
    }
    governor = MagicMock()
    governor.run_daily.return_value = {
        "state": "guarded",
        "confidence_factor": 0.85,
        "shadow_champion_profile": None,
        "live_trading_allowed": False,
    }

    with patch(
        "src.services.backtest_service.BacktestService",
        return_value=backtest,
    ), patch(
        "src.services.decision_signal_outcome_service.DecisionSignalOutcomeService",
        return_value=outcomes,
    ), patch(
        "src.services.adaptive_learning_service.AdaptiveLearningService",
        return_value=governor,
    ):
        report = main._run_auto_backtest(
            _config(adaptive_learning_enabled=True)
        )

    governor.run_daily.assert_called_once_with(
        outcome_stats=outcomes.get_stats.return_value,
        backtest_summary=backtest.get_global_summary.return_value,
    )
    assert "自主治理状态" in report
    assert "下一轮全局置信度系数上限 0.85" in report
    assert "真实下单" in report
