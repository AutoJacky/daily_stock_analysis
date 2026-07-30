# -*- coding: utf-8 -*-
"""Tests for objective adaptive-learning governance."""

from __future__ import annotations

from datetime import date
import os

import pytest

from src.config import Config
from src.services.adaptive_learning_service import AdaptiveLearningService
from src.storage import DatabaseManager


@pytest.fixture()
def isolated_db(tmp_path):
    old_database_path = os.environ.get("DATABASE_PATH")
    os.environ["DATABASE_PATH"] = str(tmp_path / "adaptive-learning.db")
    Config.reset_instance()
    DatabaseManager.reset_instance()
    db = DatabaseManager.get_instance()
    try:
        yield db
    finally:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        if old_database_path is None:
            os.environ.pop("DATABASE_PATH", None)
        else:
            os.environ["DATABASE_PATH"] = old_database_path


def test_collecting_state_never_changes_confidence_or_enables_trading() -> None:
    result = AdaptiveLearningService.evaluate(
        outcome_stats={
            "total": 20,
            "completed": 18,
            "unable": 2,
            "hit_rate_pct": 80,
        },
        backtest_summary={
            "total_evaluations": 18,
            "direction_accuracy": 0.8,
        },
    )

    assert result["state"] == "collecting"
    assert result["confidence_factor"] == 1.0
    assert result["live_trading_allowed"] is False


@pytest.mark.parametrize(
    ("outcome_stats", "backtest_summary", "expected_state", "expected_factor"),
    [
        (
            {"total": 100, "completed": 60, "unable": 40, "hit_rate_pct": 70},
            {"total_evaluations": 60, "direction_accuracy": 0.7},
            "data_blocked",
            0.65,
        ),
        (
            {"total": 60, "completed": 60, "unable": 0, "hit_rate_pct": 42},
            {
                "total_evaluations": 60,
                "direction_accuracy": 0.42,
                "avg_return": -0.02,
            },
            "restricted",
            0.65,
        ),
        (
            {"total": 60, "completed": 60, "unable": 0, "hit_rate_pct": 49},
            {"total_evaluations": 60, "direction_accuracy": 0.51},
            "guarded",
            0.85,
        ),
        (
            {"total": 80, "completed": 76, "unable": 4, "hit_rate_pct": 58},
            {"total_evaluations": 76, "direction_accuracy": 0.56},
            "stable",
            1.0,
        ),
    ],
)
def test_governor_only_keeps_or_reduces_confidence(
    outcome_stats,
    backtest_summary,
    expected_state,
    expected_factor,
) -> None:
    result = AdaptiveLearningService.evaluate(
        outcome_stats=outcome_stats,
        backtest_summary=backtest_summary,
    )

    assert result["state"] == expected_state
    assert result["confidence_factor"] == expected_factor
    assert result["confidence_factor"] <= 1.0
    assert result["live_trading_allowed"] is False


def test_profile_can_only_be_promoted_to_shadow_champion() -> None:
    result = AdaptiveLearningService.evaluate(
        outcome_stats={
            "total": 140,
            "completed": 130,
            "unable": 10,
            "hit_rate_pct": 57,
            "profile_calibration": {
                "breakdowns": {
                    "decision_profile": [
                        {
                            "dimensions": {"decision_profile": "balanced"},
                            "completed": 70,
                            "hit_rate_pct": 60,
                            "unable_rate_pct": 5,
                        },
                        {
                            "dimensions": {"decision_profile": "aggressive"},
                            "completed": 60,
                            "hit_rate_pct": 58,
                            "unable_rate_pct": 5,
                        },
                    ]
                }
            },
        },
        backtest_summary={
            "total_evaluations": 80,
            "direction_accuracy": 0.56,
        },
    )

    assert result["shadow_champion_profile"] == "balanced"
    assert result["live_trading_allowed"] is False


def test_daily_snapshot_is_idempotent_and_persistent(isolated_db) -> None:
    service = AdaptiveLearningService(db_manager=isolated_db)
    kwargs = {
        "outcome_stats": {
            "total": 50,
            "completed": 45,
            "unable": 5,
            "hit_rate_pct": 48,
        },
        "backtest_summary": {
            "total_evaluations": 45,
            "direction_accuracy": 0.49,
        },
        "snapshot_date": date(2026, 7, 30),
    }

    first = service.run_daily(**kwargs)
    second = service.run_daily(**kwargs)
    latest = service.get_latest_snapshot()

    assert first["id"] == second["id"]
    assert latest is not None
    assert latest["snapshot_date"] == "2026-07-30"
    assert latest["state"] == "guarded"
    assert latest["confidence_factor"] == 0.85
    assert latest["live_trading_allowed"] is False
