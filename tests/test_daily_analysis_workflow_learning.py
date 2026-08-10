# -*- coding: utf-8 -*-
"""Static safeguards for the cloud prediction-feedback loop."""

from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = ROOT_DIR / ".github/workflows/00-daily-analysis.yml"


def _steps() -> list[dict]:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    return workflow["jobs"]["analyze"]["steps"]


def test_learning_database_is_restored_before_analysis() -> None:
    steps = _steps()
    names = [step.get("name") for step in steps]
    restore_index = names.index("恢复历史预测与校准数据库")
    analysis_index = names.index("执行股票分析")

    assert restore_index < analysis_index
    restore = steps[restore_index]
    assert restore["uses"] == "actions/cache/restore@v4"
    cache_path = restore["with"]["path"]
    assert "data/stock_analysis.db" in cache_path
    assert "restore-keys" in restore["with"]
    assert "github.event_name" in restore["with"]["key"]
    assert "github.run_attempt" in restore["with"]["key"]


def test_only_scheduled_runs_persist_a_valid_learning_database() -> None:
    steps = _steps()
    validate = next(step for step in steps if step.get("name") == "校验每日学习数据库")
    save = next(step for step in steps if step.get("name") == "保存每日预测与校准数据库")

    assert "github.event_name == 'schedule'" in validate["if"]
    assert "PRAGMA integrity_check" in validate["run"]
    assert save["uses"] == "actions/cache/save@v4"
    assert "github.event_name == 'schedule'" in save["if"]
    assert "steps.learning_db.outputs.ready == 'true'" in save["if"]
    assert "github.run_attempt" in save["with"]["key"]


def test_daily_analysis_enables_objective_backtest_and_agent_calibration() -> None:
    analyze = next(step for step in _steps() if step.get("name") == "执行股票分析")
    env = analyze["env"]

    assert env["DATABASE_PATH"] == "./data/stock_analysis.db"
    assert "BACKTEST_ENABLED" in env
    assert "BACKTEST_EVAL_WINDOW_DAYS" in env
    assert "BACKTEST_MIN_AGE_DAYS" in env
    assert "AGENT_MEMORY_ENABLED" in env
    assert "ADAPTIVE_LEARNING_ENABLED" in env
    assert "ADAPTIVE_LEARNING_NOTIFY_ENABLED" in env
    assert "'false'" in str(env["ADAPTIVE_LEARNING_NOTIFY_ENABLED"])
    assert "DSA_TRIGGER_SOURCE" in env
