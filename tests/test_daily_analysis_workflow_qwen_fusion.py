"""Static safeguards for the free-only Qwen fusion workflow."""

from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = ROOT_DIR / ".github/workflows/00-daily-analysis.yml"


def _steps() -> list[dict]:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    return workflow["jobs"]["analyze"]["steps"]


def test_fusion_is_opt_in_and_requires_encrypted_modelscope_token() -> None:
    steps = _steps()
    check = next(step for step in steps if step.get("name") == "检查免费千问融合条件")
    assert "QWEN_FREE_FUSION_ENABLED" in check["env"]
    assert "secrets.MODELSCOPE_ACCESS_TOKEN" in check["env"]["MODELSCOPE_ACCESS_TOKEN"]
    assert "enabled=false" in check["run"]


def test_source_notifications_are_suppressed_only_for_ready_scheduled_fusion() -> None:
    analysis = next(step for step in _steps() if step.get("name") == "执行股票分析")
    script = analysis["run"]
    assert 'FUSION_MARKET="cn"' in script
    assert 'FUSION_MARKET="us"' in script
    assert "steps.qwen_fusion.outputs.enabled" in script
    assert 'NOTIFY_ARG="--no-notify"' in script


def test_fusion_step_uses_local_free_only_script_and_pushplus() -> None:
    fusion = next(step for step in _steps() if step.get("name") == "生成并推送双 AI 融合终稿")
    assert "steps.qwen_fusion.outputs.enabled == 'true'" in fusion["if"]
    assert "scripts/fuse_qwen_market_report.py" in fusion["run"]
    assert "--send" in fusion["run"]
    assert "github.event.inputs.send_notification" in fusion["run"]
    assert fusion["env"]["NOTIFICATION_REPORT_CHANNELS"] == "pushplus"
    assert "secrets.MODELSCOPE_ACCESS_TOKEN" in fusion["env"]["MODELSCOPE_ACCESS_TOKEN"]


def test_manual_dispatch_can_choose_fusion_market_for_no_notify_validation() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    dispatch = workflow[True]["workflow_dispatch"]["inputs"]
    assert dispatch["fusion_market"]["options"] == ["none", "cn", "us"]
    analysis = next(step for step in _steps() if step.get("name") == "执行股票分析")
    assert "github.event.inputs.fusion_market" in analysis["run"]
