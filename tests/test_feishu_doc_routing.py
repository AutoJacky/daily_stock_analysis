"""Feishu cloud documents must obey the report-channel allow-list."""

from types import SimpleNamespace

from src.feishu_doc import FeishuDocManager


def _manager(routes):
    manager = FeishuDocManager.__new__(FeishuDocManager)
    manager.config = SimpleNamespace(notification_report_channels=routes)
    manager.app_id = "app-id"
    manager.app_secret = "app-secret"
    manager.folder_token = "folder-token"
    return manager


def test_feishu_doc_is_disabled_for_pushplus_only_reports():
    assert _manager(["pushplus"]).is_configured() is False


def test_feishu_doc_remains_available_when_report_route_includes_feishu():
    assert _manager(["pushplus", "feishu"]).is_configured() is True
