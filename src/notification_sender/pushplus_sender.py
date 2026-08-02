# -*- coding: utf-8 -*-
"""
PushPlus 发送提醒服务

职责：
1. 通过 PushPlus API 发送 PushPlus 消息
"""
import logging
import re
import time
from collections import deque
from html import unescape
from typing import Optional
from datetime import datetime
import requests

from src.config import Config
from src.formatters import (
    compact_markdown_single_document,
    markdown_to_pushplus_compact_html,
    markdown_to_pushplus_html,
)


logger = logging.getLogger(__name__)
_MARKDOWN_TITLE_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_PUSHPLUS_RATE_LIMIT = 5
_PUSHPLUS_RATE_WINDOW_SECONDS = 60.5
_PUSHPLUS_TRANSIENT_RETRY_SECONDS = 3.0
_PUSHPLUS_CONTENT_SAFETY_RATIO = 0.90


class PushplusSender:
    
    def __init__(self, config: Config):
        """
        初始化 PushPlus 配置

        Args:
            config: 配置对象
        """
        self._pushplus_token = getattr(config, 'pushplus_token', None)
        self._pushplus_topic = getattr(config, 'pushplus_topic', None)
        # PushPlus documents its normal-account limit as 20,000 *characters*,
        # not UTF-8 bytes.  The historical byte-based check split a short
        # Chinese report into three separate WeChat notifications because one
        # Han character occupies three bytes.  Keep the old, undocumented
        # config name as a compatibility fallback, but use character semantics.
        self._pushplus_max_chars = getattr(
            config,
            'pushplus_max_chars',
            getattr(config, 'pushplus_max_bytes', 20000),
        )
        # PushPlus enforces a five-requests-per-minute account quota.  This
        # instance is shared by NotificationService, so the queue also covers
        # a long stock report followed immediately by a market review.
        self._pushplus_request_slots = deque()

    def _effective_html_limit(self) -> int:
        """Return a conservative final-HTML character limit."""

        return max(
            1,
            int(self._pushplus_max_chars * _PUSHPLUS_CONTENT_SAFETY_RATIO),
        )
        
    def send_to_pushplus(
        self,
        content: str,
        title: Optional[str] = None,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        """
        推送消息到 PushPlus

        PushPlus API 格式：
        POST http://www.pushplus.plus/send
        {
            "token": "用户令牌",
            "title": "消息标题",
            "content": "消息内容",
            "template": "html/txt/json/markdown"
        }

        PushPlus 特点：
        - 国内推送服务，免费额度充足
        - 支持微信公众号推送
        - 支持多种消息格式

        Args:
            content: 消息内容（Markdown 格式）
            title: 消息标题（可选）

        Returns:
            是否发送成功
        """
        if not self._pushplus_token:
            logger.warning("PushPlus Token 未配置，跳过推送")
            return False

        api_url = "https://www.pushplus.plus/send"

        if title is None:
            title = self._mobile_title(content)

        try:
            html_content = markdown_to_pushplus_html(content)
            template = "html"
            content_chars = len(html_content)
            if content_chars > self._effective_html_limit():
                compact_html = markdown_to_pushplus_compact_html(content)
                logger.info(
                    "PushPlus 精美HTML=%s字符超限，已自动切换单页精简HTML=%s字符",
                    content_chars,
                    len(compact_html),
                )
                html_content = compact_html

            if len(html_content) > self._effective_html_limit():
                # Markdown has dramatically lower formatting overhead while
                # retaining headings, emphasis, tables and links.  It is the
                # last normal fallback before proportional section compaction.
                html_content = compact_markdown_single_document(
                    content,
                    self._effective_html_limit(),
                )
                template = "markdown"
                logger.warning(
                    "PushPlus 精简HTML仍超限，改用单条Markdown：%s字符",
                    len(html_content),
                )

            return self._send_pushplus_message(
                api_url,
                html_content,
                title,
                timeout_seconds=timeout_seconds,
                template=template,
            )
        except Exception as e:
            logger.error(f"发送 PushPlus 消息失败: {e}")
            return False

    @staticmethod
    def _mobile_title(content: str) -> str:
        """Build a short title that does not wrap into two lines on phones."""

        date_str = datetime.now().strftime("%m-%d")
        title_probe = (content or "")[:1200]
        if "A股复盘·自选决策" in title_probe:
            return f"🎯 A股自选决策 · {date_str}"
        if "美股复盘·自选决策" in title_probe:
            return f"🎯 美股自选决策 · {date_str}"
        if "港股/日股复盘·自选决策" in title_probe:
            return f"🎯 港日股自选决策 · {date_str}"
        if "港股复盘·自选决策" in title_probe:
            return f"🎯 港股自选决策 · {date_str}"
        if "日股复盘·自选决策" in title_probe:
            return f"🎯 日股自选决策 · {date_str}"
        if "大盘复盘" in title_probe or "市场复盘" in title_probe:
            if "美股" in title_probe:
                return f"📊 美股收盘复盘 · {date_str}"
            if "A股" in title_probe:
                return f"📊 A股收盘复盘 · {date_str}"
            return f"📊 收盘复盘 · {date_str}"
        if any(token in title_probe for token in ("模型复核", "预测复核", "置信度校准")):
            return f"🧭 每日模型复核 · {date_str}"

        match = _MARKDOWN_TITLE_RE.search(content or "")
        heading = re.sub(r"[*_`#]", "", match.group(1)).strip() if match else ""
        if "大盘复盘" in heading or "市场复盘" in heading:
            return f"📊 收盘复盘 · {date_str}"
        if any(token in heading for token in ("复核", "校准", "回测")):
            return f"🧭 每日模型复核 · {date_str}"
        stock_match = re.search(
            r"(?:[^一-鿿A-Za-z0-9*]*)(.+?)\s*[\(（]([A-Za-z0-9.\-]+)[\)）]",
            heading,
        )
        if stock_match:
            stock_name = stock_match.group(1).strip(" ·-|")
            stock_code = stock_match.group(2).strip().upper()
            if stock_name and stock_code:
                return f"📈 {stock_name} · {stock_code} · {date_str}"
        return f"📈 自选股研报 · {date_str}"

    def _send_pushplus_message(
        self,
        api_url: str,
        content: str,
        title: str,
        *,
        timeout_seconds: Optional[float] = None,
        template: str = "html",
    ) -> bool:
        content_chars = len(content)
        effective_limit = self._effective_html_limit()
        if content_chars > effective_limit:
            content, template = self._force_single_document_fit(
                content,
                template,
                effective_limit,
            )
            content_chars = len(content)
            logger.warning(
                "PushPlus 载荷在发送前仍超限，已强制瘦身为单文档：%s字符/%s字符",
                content_chars,
                effective_limit,
            )

        payload = {
            "token": self._pushplus_token,
            "title": title,
            "content": content,
            "template": template,
        }

        if self._pushplus_topic:
            payload["topic"] = self._pushplus_topic

        for attempt in range(3):
            self._wait_for_rate_slot()
            try:
                response = requests.post(
                    api_url,
                    json=payload,
                    timeout=timeout_seconds or 10,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt == 0:
                    logger.warning(
                        "PushPlus 瞬时网络失败，将在 %.1f 秒后重试一次: %s",
                        _PUSHPLUS_TRANSIENT_RETRY_SECONDS,
                        exc,
                    )
                    time.sleep(_PUSHPLUS_TRANSIENT_RETRY_SECONDS)
                    continue
                logger.error("PushPlus 网络重试仍失败: %s", exc)
                return False

            result = {}
            if response.status_code == 200:
                try:
                    result = response.json()
                except (TypeError, ValueError):
                    result = {}
                if result.get('code') == 200:
                    # PushPlus handles delivery asynchronously.  A 200 response
                    # means that the request was accepted, not that WeChat has
                    # already completed the final delivery.
                    logger.info("PushPlus 消息请求已受理")
                    return True

            retry_delay = self._retry_delay_seconds(response.status_code, result)
            if retry_delay is not None and attempt == 0:
                logger.warning(
                    "PushPlus 服务端暂时拒绝请求，将在 %.1f 秒后重试一次",
                    retry_delay,
                )
                time.sleep(retry_delay)
                continue

            if self._is_content_limit_response(response.status_code, result) and attempt < 2:
                # The documented limit is character based, but channel/account
                # policy can still be stricter.  A rejected request creates no
                # WeChat notification, so retry the same document at a smaller
                # size instead of dropping it or splitting it into many pushes.
                retry_limit = max(1, int(len(payload["content"]) * 0.60))
                retry_content, retry_template = self._force_single_document_fit(
                    payload["content"],
                    payload["template"],
                    retry_limit,
                )
                payload = {
                    **payload,
                    "content": retry_content,
                    "template": retry_template,
                }
                content_chars = len(retry_content)
                logger.warning(
                    "PushPlus 服务端判定内容超限，已自动瘦身后重投：%s字符",
                    content_chars,
                )
                continue

            if response.status_code == 200:
                error_msg = result.get('msg', '未知错误')
                logger.error(
                    "PushPlus 返回错误(code=%s, html_chars=%s): %s",
                    result.get("code"),
                    content_chars,
                    error_msg,
                )
            else:
                logger.error(f"PushPlus 请求失败: HTTP {response.status_code}")
            return False

        return False

    @staticmethod
    def _force_single_document_fit(
        content: str,
        template: str,
        max_chars: int,
    ) -> tuple[str, str]:
        """Return one readable payload that cannot exceed ``max_chars``."""

        source = str(content or "")
        if template == "html":
            source = re.sub(r"<[^>]+>", "\n", source)
            source = unescape(source)
            source = re.sub(r"\n{3,}", "\n\n", source).strip()
        fitted = compact_markdown_single_document(source, max(1, max_chars))
        return fitted[:max(1, max_chars)], "markdown"

    @staticmethod
    def _is_content_limit_response(status_code: int, result: dict) -> bool:
        """Detect a server-side content-size rejection eligible for compaction."""

        message = str(result.get("msg") or "").strip().lower()
        markers = (
            "内容超限",
            "内容长度",
            "字符上限",
            "字数超限",
            "content too long",
            "content length",
            "payload too large",
        )
        return status_code == 413 or any(marker in message for marker in markers)

    @staticmethod
    def _retry_delay_seconds(status_code: int, result: dict) -> Optional[float]:
        """Return one conservative retry delay for rate/transient failures."""

        message = str(result.get("msg") or "").strip().lower()
        business_code = result.get("code")
        rate_markers = (
            "频率",
            "限流",
            "太频繁",
            "一分钟",
            "每分钟",
            "too many",
            "rate limit",
        )
        if status_code == 429 or business_code == 429 or any(
            marker in message for marker in rate_markers
        ):
            return _PUSHPLUS_RATE_WINDOW_SECONDS
        if status_code in {408, 425} or status_code >= 500:
            return _PUSHPLUS_TRANSIENT_RETRY_SECONDS
        if status_code == 200 and not result:
            return _PUSHPLUS_TRANSIENT_RETRY_SECONDS
        return None

    def _wait_for_rate_slot(self) -> None:
        """Reserve a rolling-window request slot before calling PushPlus."""

        now = time.monotonic()
        while (
            self._pushplus_request_slots
            and self._pushplus_request_slots[0] <= now - _PUSHPLUS_RATE_WINDOW_SECONDS
        ):
            self._pushplus_request_slots.popleft()

        scheduled_at = now
        if len(self._pushplus_request_slots) >= _PUSHPLUS_RATE_LIMIT:
            scheduled_at = max(
                now,
                self._pushplus_request_slots[-_PUSHPLUS_RATE_LIMIT]
                + _PUSHPLUS_RATE_WINDOW_SECONDS,
            )
        self._pushplus_request_slots.append(scheduled_at)

        delay = scheduled_at - now
        if delay > 0:
            logger.info("PushPlus 每分钟限额保护：等待 %.1f 秒后继续发送", delay)
            time.sleep(delay)
