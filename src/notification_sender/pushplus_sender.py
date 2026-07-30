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
from typing import Optional
from datetime import datetime
import requests

from src.config import Config
from src.formatters import chunk_markdown_preserving_blocks, markdown_to_pushplus_html


logger = logging.getLogger(__name__)
_MARKDOWN_TITLE_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
_PUSHPLUS_RATE_LIMIT = 5
_PUSHPLUS_RATE_WINDOW_SECONDS = 60.5
_PUSHPLUS_TRANSIENT_RETRY_SECONDS = 3.0


class PushplusSender:
    
    def __init__(self, config: Config):
        """
        初始化 PushPlus 配置

        Args:
            config: 配置对象
        """
        self._pushplus_token = getattr(config, 'pushplus_token', None)
        self._pushplus_topic = getattr(config, 'pushplus_topic', None)
        self._pushplus_max_bytes = getattr(config, 'pushplus_max_bytes', 20000)
        # PushPlus enforces a five-requests-per-minute account quota.  This
        # instance is shared by NotificationService, so the queue also covers
        # a long stock report followed immediately by a market review.
        self._pushplus_request_slots = deque()
        
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
            content_bytes = len(html_content.encode('utf-8'))
            if content_bytes > self._pushplus_max_bytes:
                logger.info(
                    "PushPlus HTML 内容超长(%s字节/%s字符)，将分批发送",
                    content_bytes,
                    len(content),
                )
                return self._send_pushplus_chunked(
                    api_url,
                    content,
                    title,
                    self._pushplus_max_bytes,
                    timeout_seconds=timeout_seconds,
                )

            return self._send_pushplus_message(
                api_url,
                html_content,
                title,
                timeout_seconds=timeout_seconds,
            )
        except Exception as e:
            logger.error(f"发送 PushPlus 消息失败: {e}")
            return False

    @staticmethod
    def _mobile_title(content: str) -> str:
        """Build a short title that does not wrap into two lines on phones."""

        date_str = datetime.now().strftime("%m-%d")
        title_probe = (content or "")[:1200]
        if "大盘复盘" in title_probe or "市场复盘" in title_probe:
            return f"📊 收盘复盘 · {date_str}"
        if any(token in title_probe for token in ("模型复核", "预测复核", "置信度校准")):
            return f"🧭 每日模型复核 · {date_str}"

        match = _MARKDOWN_TITLE_RE.search(content or "")
        heading = re.sub(r"[*_`#]", "", match.group(1)).strip() if match else ""
        if "大盘复盘" in heading or "市场复盘" in heading:
            return f"📊 收盘复盘 · {date_str}"
        if any(token in heading for token in ("复核", "校准", "回测")):
            return f"🧭 每日模型复核 · {date_str}"
        return f"📈 自选股研报 · {date_str}"

    def _send_pushplus_message(
        self,
        api_url: str,
        content: str,
        title: str,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        payload = {
            "token": self._pushplus_token,
            "title": title,
            "content": content,
            "template": "html",
        }

        if self._pushplus_topic:
            payload["topic"] = self._pushplus_topic

        for attempt in range(2):
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
                    logger.info("PushPlus 消息发送成功")
                    return True

            retry_delay = self._retry_delay_seconds(response.status_code, result)
            if retry_delay is not None and attempt == 0:
                logger.warning(
                    "PushPlus 服务端暂时拒绝请求，将在 %.1f 秒后重试一次",
                    retry_delay,
                )
                time.sleep(retry_delay)
                continue

            if response.status_code == 200:
                error_msg = result.get('msg', '未知错误')
                logger.error(f"PushPlus 返回错误: {error_msg}")
            else:
                logger.error(f"PushPlus 请求失败: HTTP {response.status_code}")
            return False

        return False

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
            # This is the response observed when the account-wide 5/minute
            # validation gate rejected pages 6–8.
            "服务端验证错误",
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

    def _send_pushplus_chunked(
        self,
        api_url: str,
        content: str,
        title: str,
        max_bytes: int,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        """Split Markdown first, then render every page as standalone mobile HTML."""

        wrapper_bytes = len(markdown_to_pushplus_html("").encode("utf-8"))
        budget = max(500, max_bytes - wrapper_bytes - 2500)
        byte_len = lambda value: len(value.encode("utf-8"))
        chunks = chunk_markdown_preserving_blocks(
            content,
            budget,
            len_fn=byte_len,
            add_page_marker=False,
        )

        # Escaped source text may expand after Markdown-to-HTML conversion. Tighten
        # the source budget until every rendered page stays below the API limit.
        for _ in range(6):
            rendered_chunks = [markdown_to_pushplus_html(chunk) for chunk in chunks]
            if all(len(chunk.encode("utf-8")) <= max_bytes for chunk in rendered_chunks):
                break
            budget = max(500, int(budget * 0.72))
            chunks = chunk_markdown_preserving_blocks(
                content,
                budget,
                len_fn=byte_len,
                add_page_marker=False,
            )
        else:
            rendered_chunks = [markdown_to_pushplus_html(chunk) for chunk in chunks]

        total_chunks = len(chunks)
        success_count = 0

        logger.info(f"PushPlus 分批发送：共 {total_chunks} 批")

        for i, html_chunk in enumerate(rendered_chunks):
            chunk_title = f"{title} · {i+1}/{total_chunks}" if total_chunks > 1 else title
            if self._send_pushplus_message(
                api_url,
                html_chunk,
                chunk_title,
                timeout_seconds=timeout_seconds,
            ):
                success_count += 1
                logger.info(f"PushPlus 第 {i+1}/{total_chunks} 批发送成功")
            else:
                logger.error(f"PushPlus 第 {i+1}/{total_chunks} 批发送失败")

            if i < total_chunks - 1:
                time.sleep(1)

        return success_count == total_chunks
