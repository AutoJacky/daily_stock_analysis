# -*- coding: utf-8 -*-
"""
===================================
格式化工具模块
===================================

提供各种内容格式化工具函数，用于将通用格式转换为平台特定格式。
"""

import re
from typing import Callable, List, Optional

import markdown2

TRUNCATION_SUFFIX = "\n\n...(本段内容过长已截断)"
PAGE_MARKER_PREFIX = f"\n\n📄"
PAGE_MARKER_SAFE_BYTES = 16 # "\n\n📄 9999/9999"
PAGE_MARKER_SAFE_LEN = 13   # "\n\n📄 9999/9999"
MIN_MAX_WORDS = 10
MIN_MAX_BYTES = 40
FENCED_CODE_BLOCK_RE = re.compile(r"(^```[^\n]*\n.*?^```[ \t]*$)", re.MULTILINE | re.DOTALL)
FENCED_CODE_BLOCK_PLACEHOLDER = "@@DSA_FENCED_CODE_BLOCK_{}@@"
MARKDOWN_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
RAW_HTTP_URL_RE = re.compile(r"https?://[^\s<>\]）)。，；、]+")

# Unicode code point ranges for special characters.
_SPECIAL_CHAR_RANGE = (0x10000, 0xFFFFF)
_SPECIAL_CHAR_REGEX = re.compile(r'[\U00010000-\U000FFFFF]')


def _page_marker(i: int, total: int) -> str:
    return f"{PAGE_MARKER_PREFIX} {i+1}/{total}"


def _is_special_char(c: str) -> bool:
    """判断字符是否为特殊字符
    
    Args:
        c: 字符
        
    Returns:
        True 如果字符为特殊字符，False 否则
    """
    if len(c) != 1:
        return False
    cp = ord(c)
    return _SPECIAL_CHAR_RANGE[0] <= cp <= _SPECIAL_CHAR_RANGE[1]


def _count_special_chars(s: str) -> int:
    """
    计算字符串中的特殊字符数量
    
    Args:
        s: 字符串
    """
    # reg find all (0x10000, 0xFFFFF)
    match = _SPECIAL_CHAR_REGEX.findall(s)
    return len(match)


def _effective_len(s: str, special_char_len: int = 2) -> int:
    """
    计算字符串的有效长度
    
    Args:
        s: 字符串
        special_char_len: 每个特殊字符的长度，默认为 2
        
    Returns:
        s 的有效长度
    """
    n = len(s)
    n += _count_special_chars(s) * (special_char_len - 1)
    return n


def _slice_at_effective_len(s: str, effective_len: int, special_char_len: int = 2) -> tuple[str, str]:
    """
    按有效长度分割字符串
    
    Args:
        s: 字符串
        effective_len: 有效长度
        special_char_len: 每个特殊字符的长度，默认为 2
        
    Returns:
        分割后的前、后部分字符串
    """
    if _effective_len(s, special_char_len) <= effective_len:
        return s, ""
    
    s_ = s[:effective_len]
    n_special_chars = _count_special_chars(s_)
    residual_lens = n_special_chars * (special_char_len - 1) + len(s_) - effective_len
    while residual_lens > 0:
        residual_lens -= special_char_len if _is_special_char(s_[-1]) else 1
        s_ = s_[:-1]
    return s_, s[len(s_):]


def markdown_to_html_document(markdown_text: str) -> str:
    """
    Convert Markdown to a complete HTML document (for email, md2img, etc.).

    Uses markdown2 with table and code block support, wraps with inline CSS
    for compact, readable layout. Reused by notification email and md2img.

    Args:
        markdown_text: Raw Markdown content.

    Returns:
        Full HTML document string with DOCTYPE, head, and body.
    """
    html_content = markdown2.markdown(
        markdown_text,
        extras=["tables", "fenced-code-blocks", "break-on-newline", "cuddled-lists"],
    )

    css_style = """
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                line-height: 1.5;
                color: #24292e;
                font-size: 14px;
                padding: 15px;
                max-width: 900px;
                margin: 0 auto;
            }
            h1 {
                font-size: 20px;
                border-bottom: 1px solid #eaecef;
                padding-bottom: 0.3em;
                margin-top: 1.2em;
                margin-bottom: 0.8em;
                color: #0366d6;
            }
            h2 {
                font-size: 18px;
                border-bottom: 1px solid #eaecef;
                padding-bottom: 0.3em;
                margin-top: 1.0em;
                margin-bottom: 0.6em;
            }
            h3 {
                font-size: 16px;
                margin-top: 0.8em;
                margin-bottom: 0.4em;
            }
            p {
                margin-top: 0;
                margin-bottom: 8px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 12px 0;
                display: block;
                overflow-x: auto;
                font-size: 13px;
            }
            th, td {
                border: 1px solid #dfe2e5;
                padding: 6px 10px;
                text-align: left;
            }
            th {
                background-color: #f6f8fa;
                font-weight: 600;
            }
            tr:nth-child(2n) {
                background-color: #f8f8f8;
            }
            tr:hover {
                background-color: #f1f8ff;
            }
            blockquote {
                color: #6a737d;
                border-left: 0.25em solid #dfe2e5;
                padding: 0 1em;
                margin: 0 0 10px 0;
            }
            code {
                padding: 0.2em 0.4em;
                margin: 0;
                font-size: 85%;
                background-color: rgba(27,31,35,0.05);
                border-radius: 3px;
                font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
            }
            pre {
                padding: 12px;
                overflow: auto;
                line-height: 1.45;
                background-color: #f6f8fa;
                border-radius: 3px;
                margin-bottom: 10px;
            }
            hr {
                height: 0.25em;
                padding: 0;
                margin: 16px 0;
                background-color: #e1e4e8;
                border: 0;
            }
            ul, ol {
                padding-left: 20px;
                margin-bottom: 10px;
            }
            li {
                margin: 2px 0;
            }
        """

    return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                {css_style}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """


def _compact_report_links(markdown_text: str) -> str:
    """Hide long raw URLs while keeping every source clickable.

    LLM/search evidence often arrives as ``标题（https://...）``.  PushPlus
    renders that URL verbatim, which makes a narrow WeChat WebView almost
    unreadable.  Protect normal Markdown links first, convert raw URLs to a
    compact source pill, then restore the protected links.
    """

    protected_links: List[str] = []

    def _protect_link(match: re.Match) -> str:
        label = match.group(1).strip()
        if label.lower().startswith(("http://", "https://")):
            label = "查看来源 ↗"
        protected_links.append(f"[{label}]({match.group(2)})")
        return f"@@DSA_PUSHPLUS_LINK_{len(protected_links) - 1}@@"

    compact = MARKDOWN_LINK_RE.sub(_protect_link, markdown_text or "")
    compact = RAW_HTTP_URL_RE.sub(
        lambda match: f"[查看来源 ↗]({match.group(0)})",
        compact,
    )
    for index, link in enumerate(protected_links):
        compact = compact.replace(f"@@DSA_PUSHPLUS_LINK_{index}@@", link)
    return compact


def _pushplus_highlight_strong(match: re.Match) -> str:
    """Apply high-contrast semantic colour to bold report labels."""

    inner = match.group(1)
    plain = re.sub(r"<[^>]+>", "", inner).strip().lower()
    style = "font-weight:800;color:#172033;"
    if any(
        word in plain
        for word in ("风险", "警报", "缺失", "异常", "不可用", "止损", "失效")
    ):
        style += (
            "display:inline-block;background:#ffe4e8;color:#b42318;"
            "padding:3px 7px;border-radius:6px;"
        )
    elif any(word in plain for word in ("买入", "加仓", "强势", "多头", "机会")):
        style += (
            "display:inline-block;background:#ffebe7;color:#c4320a;"
            "padding:3px 7px;border-radius:6px;"
        )
    elif any(word in plain for word in ("减仓", "卖出", "回避", "空头")):
        style += (
            "display:inline-block;background:#dcfae6;color:#067647;"
            "padding:3px 7px;border-radius:6px;"
        )
    elif any(
        word in plain
        for word in ("结论", "核心", "策略", "计划", "重点", "建议", "怎么做")
    ):
        style += (
            "display:inline-block;background:#dbeafe;color:#174ea6;"
            "padding:3px 7px;border-radius:6px;"
        )
    elif any(
        word in plain
        for word in ("数据", "事实", "验证", "财报", "质量", "依据", "动态")
    ):
        style += (
            "display:inline-block;background:#dcfae6;color:#067647;"
            "padding:3px 7px;border-radius:6px;"
        )
    elif any(
        word in plain
        for word in ("观察", "等待", "关注", "中性", "持有", "观望", "明天")
    ):
        style += (
            "display:inline-block;background:#fff0c2;color:#93370d;"
            "padding:3px 7px;border-radius:6px;"
        )
    return f'<strong style="{style}">{inner}</strong>'


def _pushplus_section_style(match: re.Match) -> str:
    """Colour section bars by meaning without changing report content."""

    attrs = match.group("attrs") or ""
    inner = match.group("inner")
    plain = re.sub(r"<[^>]+>", "", inner).strip().lower()
    background = "#174ea6"
    border = "#7fb0ff"
    if any(word in plain for word in ("风险", "警报", "失败", "未完成")):
        background = "#b42318"
        border = "#ff9b9b"
    elif any(word in plain for word in ("数据", "可信", "术语", "看懂")):
        background = "#9a4e00"
        border = "#ffc46b"
    elif any(word in plain for word in ("机会", "公司", "财报", "动态")):
        background = "#067647"
        border = "#75d99a"
    elif any(word in plain for word in ("结论", "重点", "计划", "自选")):
        background = "#4938a2"
        border = "#a99cff"
    style = (
        "font-size:19px;line-height:1.4;margin:22px 0 12px;padding:11px 13px;"
        f"color:#ffffff;background:{background};border-left:5px solid {border};"
        "border-radius:9px;font-weight:800;box-shadow:0 3px 8px "
        "rgba(16,24,40,0.12);"
    )
    return f'<h2{attrs} style="{style}">{inner}</h2>'


def _pushplus_semantic_paragraph(match: re.Match) -> str:
    """Turn decision, risk, and data lines into scannable slide-like callouts."""

    attrs = match.group("attrs") or ""
    inner = match.group("inner")
    plain = re.sub(r"<[^>]+>", "", inner).strip().lower()
    style = "font-size:16px;line-height:1.72;margin:0 0 11px;color:#344054;"
    palette = None
    if any(word in plain[:28] for word in ("风险", "警报", "失效", "数据缺口")):
        palette = ("#fff1f3", "#e11d48")
    elif any(word in plain[:28] for word in ("一句话结论", "核心结论", "你现在怎么做")):
        palette = ("#eff6ff", "#2563eb")
    elif any(word in plain[:28] for word in ("机会观察", "买入候选", "加仓")):
        palette = ("#fff4ed", "#f04438")
    elif any(word in plain[:28] for word in ("明天只盯", "观察条件", "等待")):
        palette = ("#fffbeb", "#f59e0b")
    elif any(word in plain[:28] for word in ("主要依据", "公司动态", "数据可信")):
        palette = ("#ecfdf3", "#12b76a")
    if palette:
        style += (
            f"padding:10px 11px;background:{palette[0]};"
            f"border-left:4px solid {palette[1]};border-radius:7px;"
        )
    return f'<p{attrs} style="{style}">{inner}</p>'


def _pushplus_glossary_html(markdown_text: str) -> str:
    """Add a short, deterministic glossary when a mobile report uses jargon."""

    source = markdown_text or ""
    if "不懂术语就看这里" in source:
        return ""
    candidates = (
        (
            r"(?<![A-Za-z])PE(?![A-Za-z])|市盈率",
            "市盈率（PE）",
            "股价相对公司利润的倍数；高低要结合行业和增长看。",
        ),
        (
            r"(?<![A-Za-z])PB(?![A-Za-z])|市净率",
            "市净率（PB）",
            "股价相对净资产的倍数；低于同行不一定代表被低估。",
        ),
        (
            r"(?<![A-Za-z])RSI(?![A-Za-z])",
            "RSI",
            "反映短期涨跌强弱，只是超买超卖参考，不是买卖指令。",
        ),
        (
            r"(?<![A-Za-z])MACD(?![A-Za-z])",
            "MACD",
            "观察趋势和动能变化，单独的金叉或死叉不能决定交易。",
        ),
        (
            r"均线|MA\d+",
            "均线",
            "一段时间的平均价格，用来观察趋势和可能的支撑压力。",
        ),
        (
            r"北向资金",
            "北向资金",
            "通过互联互通进入A股的资金流；单日流入流出不等于趋势。",
        ),
        (
            r"成交额|成交量|量能",
            "成交量/成交额",
            "反映交易活跃度；放量要结合价格方向和所处位置判断。",
        ),
    )
    matched = [
        (label, explanation)
        for pattern, label, explanation in candidates
        if re.search(pattern, source, flags=re.IGNORECASE)
    ][:4]
    if not matched:
        return ""
    items = "".join(
        (
            '<li style="font-size:14px;line-height:1.65;margin:5px 0;">'
            f'<strong style="color:#7a2e0e;">{label}：</strong>{explanation}</li>'
        )
        for label, explanation in matched
    )
    return (
        '<aside style="margin:20px 0 4px;padding:12px 13px;background:#fff7d6;'
        'border:1px solid #ffd36a;border-radius:9px;color:#5f3b12;">'
        '<div style="font-size:16px;font-weight:800;margin-bottom:6px;">'
        "🧭 新手术语小抄</div>"
        f'<ul style="margin:0;padding-left:20px;">{items}</ul></aside>'
    )


def markdown_to_pushplus_html(markdown_text: str) -> str:
    """Render a compact, mobile-first HTML document for PushPlus/WeChat.

    PushPlus' Markdown template uses browser defaults, producing oversized
    headings and raw, line-wrapping URLs.  This renderer keeps the source
    Markdown intact for other channels and supplies inline styles that survive
    the sanitisation performed by PushPlus and mobile WeChat WebViews.
    """

    compact_markdown = _compact_report_links(markdown_text)
    glossary_html = _pushplus_glossary_html(compact_markdown)
    body = markdown2.markdown(
        compact_markdown,
        extras=["tables", "fenced-code-blocks", "break-on-newline", "cuddled-lists"],
        safe_mode="escape",
    )

    tag_styles = {
        "h1": (
            "font-size:23px;line-height:1.35;margin:0 0 16px;padding:17px 15px;"
            "color:#ffffff;background:#102a56;border-radius:12px;font-weight:850;"
            "letter-spacing:-0.2px;box-shadow:0 5px 14px rgba(16,42,86,0.22);"
        ),
        "h3": (
            "font-size:18px;line-height:1.45;margin:0 0 11px;padding:10px 11px;"
            "color:#172033;background:#eef2ff;border-left:4px solid #6366f1;"
            "border-radius:7px;font-weight:800;"
        ),
        "ul": "margin:7px 0 14px;padding-left:22px;color:#344054;",
        "ol": "margin:7px 0 14px;padding-left:24px;color:#344054;",
        "li": "font-size:16px;line-height:1.68;margin:6px 0;padding-left:2px;",
        "blockquote": (
            "margin:12px 0 16px;padding:13px 14px;background:#fff7d6;"
            "border:1px solid #ffd36a;border-left:5px solid #f59e0b;"
            "border-radius:9px;color:#713b12;font-weight:650;"
        ),
        "hr": "border:0;border-top:1px solid #e4e7ec;margin:22px 0;",
        "table": (
            "box-sizing:border-box;border-collapse:collapse;width:100%;max-width:100%;"
            "table-layout:fixed;"
            "font-size:13px;line-height:1.5;"
        ),
        "th": (
            "box-sizing:border-box;border:1px solid #d0d5dd;padding:6px 5px;"
            "background:#f2f4f7;color:#344054;text-align:left;font-weight:700;"
            "word-break:break-word;overflow-wrap:anywhere;white-space:normal;"
        ),
        "td": (
            "box-sizing:border-box;border:1px solid #e4e7ec;padding:6px 5px;"
            "color:#475467;vertical-align:top;word-break:break-word;"
            "overflow-wrap:anywhere;white-space:normal;"
        ),
        "a": (
            "color:#175cd3;text-decoration:none;font-weight:650;"
            "overflow-wrap:anywhere;"
        ),
        "code": (
            "font-size:13px;background:#f2f4f7;color:#344054;padding:2px 4px;"
            "border-radius:4px;overflow-wrap:anywhere;"
        ),
        "pre": (
            "font-size:13px;line-height:1.55;background:#f8fafc;border:1px solid #e4e7ec;"
            "padding:10px;overflow:auto;border-radius:6px;"
        ),
    }
    for tag, style in tag_styles.items():
        body = re.sub(
            rf"<{tag}(?P<attrs>\s[^>]*)?>",
            lambda match, tag=tag, style=style: (
                f'<{tag}{match.group("attrs") or ""} style="{style}">'
            ),
            body,
        )

    body = re.sub(
        r"<h2(?P<attrs>\s[^>]*)?>(?P<inner>.*?)</h2>",
        _pushplus_section_style,
        body,
        flags=re.DOTALL,
    )
    body = re.sub(
        r"<p(?P<attrs>\s[^>]*)?>(?P<inner>.*?)</p>",
        _pushplus_semantic_paragraph,
        body,
        flags=re.DOTALL,
    )
    body = re.sub(
        r"<strong>(.*?)</strong>",
        _pushplus_highlight_strong,
        body,
        flags=re.DOTALL,
    )
    body = re.sub(
        r"(<h3\b.*?</h3>)(.*?)(?=<h[23]\b|$)",
        (
            '<section style="box-sizing:border-box;margin:12px 0 16px;padding:12px;'
            'background:#ffffff;border:1px solid #dfe3eb;border-radius:11px;'
            'box-shadow:0 3px 10px rgba(16,24,40,0.08);">\\1\\2</section>'
        ),
        body,
        flags=re.DOTALL,
    )
    body = re.sub(
        r"(<table\b.*?</table>)",
        (
            '<div style="width:100%;overflow:hidden;margin:12px 0 18px;'
            'border-radius:6px;">\\1</div>'
        ),
        body,
        flags=re.DOTALL,
    )

    return (
        '<div style="margin:0;background:#eef1f6;padding:10px 0;">'
        '<article style="box-sizing:border-box;max-width:760px;margin:0 auto;'
        'padding:14px 12px 24px;background:#f8f9fc;color:#344054;border-radius:12px;'
        'box-shadow:0 3px 14px rgba(16,24,40,0.08);'
        'font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,'
        'Helvetica Neue,Arial,PingFang SC,Hiragino Sans GB,Microsoft YaHei,sans-serif;'
        'word-break:break-word;-webkit-text-size-adjust:100%;">'
        f"{body}{glossary_html}"
        '<div style="margin-top:24px;padding-top:12px;border-top:1px solid #e4e7ec;'
        'font-size:12px;line-height:1.6;color:#98a2b3;">'
        "数据与结论均应结合交易所公告及最新行情复核；数据不足时以风险控制优先。"
        "</div></article></div>"
    )


def markdown_to_plain_text(markdown_text: str) -> str:
    """
    将 Markdown 转换为纯文本
    
    移除 Markdown 格式标记，保留可读性
    """
    text = markdown_text
    
    # 移除标题标记 # ## ###
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # 移除加粗 **text** -> text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    
    # 移除斜体 *text* -> text
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    
    # 移除引用 > text -> text
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # 移除列表标记 - item -> item
    text = re.sub(r'^[-*]\s+', '• ', text, flags=re.MULTILINE)
    
    # 移除分隔线 ---
    text = re.sub(r'^---+$', '────────', text, flags=re.MULTILINE)
    
    # 移除表格语法 |---|---|
    text = re.sub(r'\|[-:]+\|[-:|\s]+\|', '', text)
    text = re.sub(r'^\|(.+)\|$', r'\1', text, flags=re.MULTILINE)
    
    # 清理多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def _bytes(s: str) -> int:
    return len(s.encode('utf-8'))


def utf8_len(s: str) -> int:
    """Return the number of UTF-8 bytes used by ``s``."""

    return len(s.encode("utf-8"))


def utf16_len(s: str) -> int:
    """Return the number of UTF-16 code units used by ``s``.

    Telegram's 4096-character message limit is effectively counted in UTF-16
    units, so astral-plane characters such as emoji consume two units.
    """

    return len(s.encode("utf-16-le")) // 2


def _custom_unit_to_index(text: str, budget: int, len_fn: Callable[[str], int]) -> int:
    """Map a custom-unit budget to the largest safe Python string index."""

    if len_fn(text) <= budget:
        return len(text)
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len_fn(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _has_unclosed_inline_code(text: str) -> bool:
    """Return whether ``text`` ends inside a single-backtick inline code span."""

    escaped = False
    count = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and not escaped:
            escaped = True
            i += 1
            continue
        if ch == "`" and not escaped:
            # Triple backticks are handled as fenced code by the caller; skip
            # them here so fence delimiters do not look like inline spans.
            if text[i:i + 3] == "```":
                i += 3
                escaped = False
                continue
            count += 1
        escaped = False
        i += 1
    return count % 2 == 1


def _last_unclosed_markdown_link_start(text: str) -> int:
    """Return the start index of an inline Markdown link split in progress."""

    last_open_paren = text.rfind("](")
    last_close_paren = text.rfind(")")
    if last_open_paren > last_close_paren:
        label_start = text.rfind("[", 0, last_open_paren)
        return label_start if label_start >= 0 else last_open_paren

    last_open_bracket = text.rfind("[")
    last_close_bracket = text.rfind("]")
    if last_open_bracket > last_close_bracket:
        return last_open_bracket

    return -1


def chunk_markdown_preserving_blocks(
    content: str,
    max_units: int,
    *,
    len_fn: Optional[Callable[[str], int]] = None,
    add_page_marker: bool = False,
) -> List[str]:
    """Split Markdown while preserving common formatting boundaries.

    The splitter is intentionally conservative and does not alter report
    semantics.  If a split lands inside a fenced code block, the current chunk is
    closed and the next chunk reopens the same fence language.  It also avoids
    splitting inside inline code spans and Markdown links, and supports custom
    length functions such as :func:`utf16_len`.
    """

    measure = len_fn or len
    if max_units < MIN_MAX_WORDS:
        raise ValueError(f"max_units={max_units} < {MIN_MAX_WORDS}, 可能陷入无限递归。")
    if measure(content) <= max_units:
        return [content]

    marker_reserve = measure(_page_marker(9998, 9998)) if add_page_marker else 0
    indicator_reserve = measure("\n\n(9999/9999)")
    fence_close = "\n```"
    chunks: List[str] = []
    remaining = content
    carry_lang: Optional[str] = None

    while remaining:
        prefix = f"```{carry_lang}\n" if carry_lang is not None else ""
        headroom = max_units - marker_reserve - indicator_reserve - measure(prefix) - measure(fence_close)
        if headroom < MIN_MAX_WORDS:
            headroom = max(MIN_MAX_WORDS, max_units - marker_reserve - indicator_reserve - measure(prefix))
        if headroom <= 0:
            raise ValueError("max_units is too small for markdown-preserving chunking")

        if measure(prefix) + measure(remaining) <= max_units - marker_reserve - indicator_reserve:
            chunks.append(prefix + remaining)
            break

        cp_limit = (
            _custom_unit_to_index(remaining, headroom, measure)
            if measure is not len else min(headroom, len(remaining))
        )
        region = remaining[:cp_limit]
        split_at = region.rfind("\n\n")
        if split_at < cp_limit // 2:
            split_at = region.rfind("\n")
        if split_at < cp_limit // 2:
            split_at = region.rfind(" ")
        if split_at < 1:
            split_at = cp_limit

        candidate = remaining[:split_at]
        unsafe_start = len(candidate)
        if _has_unclosed_inline_code(candidate):
            last_tick = candidate.rfind("`")
            if last_tick >= 0:
                unsafe_start = min(unsafe_start, last_tick)

        link_start = _last_unclosed_markdown_link_start(candidate)
        if link_start >= 0:
            unsafe_start = min(unsafe_start, link_start)

        if unsafe_start < len(candidate):
            safe_split = max(candidate.rfind(" ", 0, unsafe_start), candidate.rfind("\n", 0, unsafe_start))
            if safe_split > 0:
                split_at = safe_split

        chunk_body = remaining[:split_at].rstrip()
        in_code = carry_lang is not None
        lang = carry_lang or ""
        for line in chunk_body.split("\n"):
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_code:
                    in_code = False
                    lang = ""
                else:
                    in_code = True
                    tag = stripped[3:].strip()
                    lang = tag.split()[0] if tag else ""

        next_remaining = remaining[split_at:]
        if next_remaining.startswith("\n"):
            next_remaining = next_remaining[1:]
        elif not in_code and next_remaining.startswith(" "):
            line_start = remaining.rfind("\n", 0, split_at) + 1
            if remaining[line_start:split_at].strip(" \t"):
                next_remaining = next_remaining[1:]
        remaining = next_remaining
        full_chunk = prefix + chunk_body

        if in_code:
            full_chunk += fence_close
            carry_lang = lang
        else:
            carry_lang = None

        chunks.append(full_chunk)

    if len(chunks) > 1:
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            suffix = f"\n\n({i + 1}/{total})"
            if add_page_marker:
                suffix += _page_marker(i, total)
            chunks[i] = chunk + suffix
    elif add_page_marker:
        chunks[0] = chunks[0] + _page_marker(0, 1)
    return chunks


def _is_markdown_table_separator(row: str) -> bool:
    return bool(re.match(r'^\s*\|?\s*[:-]+\s*(\|\s*[:-]+\s*)+\|?\s*$', row))


def _parse_markdown_table_row(row: str) -> List[str]:
    stripped = row.strip()
    if stripped.startswith('|'):
        stripped = stripped[1:]
    if stripped.endswith('|'):
        stripped = stripped[:-1]
    return [c.strip() for c in stripped.split('|')]


def _strip_inline_markdown(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^\*\*(.+)\*\*$', r'\1', text)
    text = re.sub(r'^\*(.+)\*$', r'\1', text)
    text = text.replace("**", "")
    return text.strip()


def _format_two_column_table_row(header: List[str], row: List[str]) -> str:
    key = _strip_inline_markdown(row[0]) if row else ""
    value = _strip_inline_markdown(row[1]) if len(row) > 1 else ""
    value_header = _strip_inline_markdown(header[1]) if len(header) > 1 else ""

    if not value:
        return key
    if value.upper() == "N/A" and value_header in {"类型", "Type"}:
        return key
    return f"{key}：{value}"


def _flush_table_as_key_value_rows(buffer: List[str], output: List[str], *, bullet: str) -> None:
    if not buffer:
        return

    rows = []
    for raw in buffer:
        if _is_markdown_table_separator(raw):
            continue
        parsed = _parse_markdown_table_row(raw)
        if parsed:
            rows.append(parsed)

    if not rows:
        return

    header = rows[0]
    data_rows = rows[1:] if len(rows) > 1 else []
    for row in data_rows:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        if len(header) == 2 and len(row) >= 2:
            output.append(f"{bullet} {_format_two_column_table_row(header, row)}")
            continue

        pairs = []
        for idx, cell in enumerate(row):
            key = _strip_inline_markdown(header[idx]) if idx < len(header) else f"列{idx + 1}"
            pairs.append(f"{key}：{_strip_inline_markdown(cell)}")
        output.append(f"{bullet} {' | '.join(pairs)}")


def _protect_fenced_code_blocks(content: str) -> tuple[str, List[str]]:
    blocks: List[str] = []

    def _replace(match: re.Match) -> str:
        blocks.append(match.group(0))
        return FENCED_CODE_BLOCK_PLACEHOLDER.format(len(blocks) - 1)

    return FENCED_CODE_BLOCK_RE.sub(_replace, content), blocks


def _restore_fenced_code_blocks(content: str, blocks: List[str]) -> str:
    restored = content
    for idx, block in enumerate(blocks):
        restored = restored.replace(FENCED_CODE_BLOCK_PLACEHOLDER.format(idx), block)
    return restored


def _transform_outside_fenced_code_blocks(content: str, transform: Callable[[str], str]) -> str:
    protected, blocks = _protect_fenced_code_blocks(content)
    return _restore_fenced_code_blocks(transform(protected), blocks)


def _markdown_tables_to_key_value_rows_unprotected(content: str, *, bullet: str) -> str:
    """Convert pipe tables to compact key-value rows for chat clients."""

    lines: List[str] = []
    table_buffer: List[str] = []

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith('|'):
            table_buffer.append(line)
            continue

        if table_buffer:
            _flush_table_as_key_value_rows(table_buffer, lines, bullet=bullet)
            table_buffer = []

        lines.append(line)

    if table_buffer:
        _flush_table_as_key_value_rows(table_buffer, lines, bullet=bullet)

    return "\n".join(lines).strip()


def markdown_tables_to_key_value_rows(content: str, *, bullet: str = "•") -> str:
    """Convert pipe tables to compact key-value rows outside fenced code blocks."""

    return _transform_outside_fenced_code_blocks(
        content,
        lambda text: _markdown_tables_to_key_value_rows_unprotected(text, bullet=bullet),
    ).strip()


def _chunk_by_max_bytes(content: str, max_bytes: int) -> List[str]:
    if _bytes(content) <= max_bytes:
        return [content]
    if max_bytes < MIN_MAX_BYTES:
        raise ValueError(f"max_bytes={max_bytes} < {MIN_MAX_BYTES}, 可能陷入无限递归。")
    
    sections: List[str] = []
    suffix = TRUNCATION_SUFFIX
    effective_max_bytes = max_bytes - _bytes(suffix)
    if effective_max_bytes <= 0:
        effective_max_bytes = max_bytes
        suffix = ""
        
    while True:
        chunk, content = slice_at_max_bytes(content, effective_max_bytes)
        if content.strip() != "":
            sections.append(chunk + suffix)
        else:
            # 最后一段了，直接添加并离开循环
            sections.append(chunk)
            break
    return sections


def chunk_content_by_max_bytes(content: str, max_bytes: int, add_page_marker: bool = False) -> List[str]:
    """
    按字节数智能分割消息内容
    
    Args:
        content: 完整消息内容
        max_bytes: 单条消息最大字节数
        add_page_marker: 是否添加分页标记
        
    Returns:
        分割后的区块列表
    """
    def _chunk(content: str, max_bytes: int) -> List[str]:
        # 优先按分隔线/标题分割，保证分页自然
        if max_bytes < MIN_MAX_BYTES:
            raise ValueError(f"max_bytes={max_bytes} < {MIN_MAX_BYTES}, 可能陷入无限递归。")
        
        if _bytes(content) <= max_bytes:
            return [content]
        
        sections, separator = _chunk_by_separators(content)
        if separator == "" and len(sections) == 1:
            # 无法智能分割，则强制按字数分割
            return _chunk_by_max_bytes(content, max_bytes)
        
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_bytes = 0
        separator_bytes = _bytes(separator) if separator else 0
        effective_max_bytes = max_bytes - separator_bytes

        for section in sections:
            section += separator
            section_bytes = _bytes(section)
            
            # 如果单个 section 就超长，需要强制截断
            if section_bytes > effective_max_bytes:
                # 先保存当前积累的内容
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_bytes = 0

                # 强制按字节截断，避免整段被截断丢失
                section_chunks = _chunk(
                    section[:-separator_bytes], effective_max_bytes
                )
                section_chunks[-1] = section_chunks[-1] + separator
                chunks.extend(section_chunks)
                continue

            # 检查加入后是否超长
            if current_bytes + section_bytes > effective_max_bytes:
                # 保存当前块，开始新块
                if current_chunk:
                    chunks.append("".join(current_chunk))
                current_chunk = [section]
                current_bytes = section_bytes
            else:
                current_chunk.append(section)
                current_bytes += section_bytes
                
        # 添加最后一块
        if current_chunk:
            chunks.append("".join(current_chunk))
            
        # 移除最后一个块的分割符
        if (chunks and 
            len(chunks[-1]) > separator_bytes and 
            chunks[-1][-separator_bytes:] == separator
        ):
            chunks[-1] = chunks[-1][:-separator_bytes]
        
        return chunks
    
    if add_page_marker:
        max_bytes = max_bytes - PAGE_MARKER_SAFE_BYTES
    
    chunks = _chunk(content, max_bytes)
    if add_page_marker:
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunks[i] = chunk + _page_marker(i, total_chunks)
    return chunks


def slice_at_max_bytes(text: str, max_bytes: int) -> tuple[str, str]:
    """
    按字节数截断字符串，确保不会在多字节字符中间截断

    Args:
        text: 要截断的字符串
        max_bytes: 最大字节数

    Returns:
        (截断后的字符串, 剩余未截断内容)
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, ""

    # 从最大字节数开始向前查找，找到完整的 UTF-8 字符边界
    truncated = encoded[:max_bytes]
    while truncated and (truncated[-1] & 0xC0) == 0x80:
        truncated = truncated[:-1]

    truncated = truncated.decode('utf-8', errors='ignore')
    return truncated, text[len(truncated):]


def _format_feishu_markdown_unprotected(content: str) -> str:
    lines = []
    table_buffer: List[str] = []

    for raw_line in content.splitlines():
        line = raw_line.rstrip()

        # 处理表格行
        if line.strip().startswith('|'):
            table_buffer.append(line)
            continue

        # 刷新表格缓冲区
        if table_buffer:
            _flush_table_as_key_value_rows(table_buffer, lines, bullet="•")
            table_buffer = []

        # 转换标题（# ## ### 等）
        if re.match(r'^#{1,6}\s+', line):
            title = re.sub(r'^#{1,6}\s+', '', line).strip()
            line = f"**{title}**" if title else ""
        # 转换引用块
        elif line.startswith('> '):
            quote = line[2:].strip()
            line = quote
        # 转换分隔线
        elif line.strip() == '---':
            line = '────────'
        # 转换列表项
        elif line.startswith('- '):
            line = f"• {line[2:].strip()}"

        lines.append(line)

    # 处理末尾的表格
    if table_buffer:
        _flush_table_as_key_value_rows(table_buffer, lines, bullet="•")

    return "\n".join(lines).strip()


def format_feishu_markdown(content: str) -> str:
    """
    将通用 Markdown 转换为飞书 lark_md 更友好的格式

    转换规则：
    - 飞书不支持 Markdown 标题（# / ## / ###），用加粗代替
    - 引用块使用前缀替代
    - 分隔线统一为细线
    - 表格转换为条目列表

    Args:
        content: 原始 Markdown 内容

    Returns:
        转换后的飞书 Markdown 格式内容

    Example:
        >>> markdown = "# 标题\\n> 引用\\n| 列1 | 列2 |"
        >>> formatted = format_feishu_markdown(markdown)
        >>> print(formatted)
        **标题**
        💬 引用
        • 列1：值1 | 列2：值2
    """
    def _flush_table_rows(buffer: List[str], output: List[str]) -> None:
        """将表格缓冲区中的行转换为飞书格式"""
        if not buffer:
            return

        def _parse_row(row: str) -> List[str]:
            """解析表格行，提取单元格"""
            return _parse_markdown_table_row(row)

        rows = []
        for raw in buffer:
            # 跳过分隔行（如 |---|---|）
            if re.match(r'^\s*\|?\s*[:-]+\s*(\|\s*[:-]+\s*)+\|?\s*$', raw):
                continue
            parsed = _parse_row(raw)
            if parsed:
                rows.append(parsed)

        if not rows:
            return

        header = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []
        for row in data_rows:
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            pairs = []
            for idx, cell in enumerate(row):
                key = header[idx] if idx < len(header) else f"列{idx + 1}"
                pairs.append(f"{key}：{cell}")
            output.append(f"• {' | '.join(pairs)}")

    lines = []
    table_buffer: List[str] = []

    for raw_line in content.splitlines():
        line = raw_line.rstrip()

        # 处理表格行
        if line.strip().startswith('|'):
            table_buffer.append(line)
            continue

        # 刷新表格缓冲区
        if table_buffer:
            _flush_table_rows(table_buffer, lines)
            table_buffer = []

        # 转换标题（# ## ### 等）
        if re.match(r'^#{1,6}\s+', line):
            title = re.sub(r'^#{1,6}\s+', '', line).strip()
            line = f"**{title}**" if title else ""
        # 转换引用块
        elif line.startswith('> '):
            quote = line[2:].strip()
            line = f"💬 {quote}" if quote else ""
        # 转换分隔线
        elif line.strip() == '---':
            line = '────────'
        # 转换列表项
        elif line.startswith('- '):
            line = f"• {line[2:].strip()}"

        lines.append(line)

    # 处理末尾的表格
    if table_buffer:
        _flush_table_rows(table_buffer, lines)

    return "\n".join(lines).strip()


def _format_telegram_markdown_unprotected(content: str) -> str:
    """Convert common report Markdown to Telegram legacy Markdown."""

    result = _markdown_tables_to_key_value_rows_unprotected(content, bullet="-")
    result = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', result, flags=re.MULTILINE)
    result = re.sub(r'\*\*(.+?)\*\*', r'*\1*', result)
    result = re.sub(r'^\s*---+\s*$', '────────', result, flags=re.MULTILINE)
    result = _escape_telegram_non_link_markdown_chars(result)
    return result.strip()


def _escape_telegram_non_link_markdown_chars(content: str) -> str:
    """Escape Telegram Markdown link metacharacters outside valid links."""

    links: list[str] = []

    def _save_link(match: re.Match) -> str:
        links.append(match.group(0))
        return f"@@DSA_TELEGRAM_LINK_{len(links) - 1}@@"

    result = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _save_link, content)
    for char in ("[", "]", "(", ")"):
        result = result.replace(char, f"\\{char}")

    for index, link in enumerate(links):
        result = result.replace(f"@@DSA_TELEGRAM_LINK_{index}@@", link)
    return result


def format_telegram_markdown(content: str) -> str:
    """Convert common report Markdown to Telegram legacy Markdown."""

    return _transform_outside_fenced_code_blocks(
        content,
        _format_telegram_markdown_unprotected,
    ).strip()


def format_wechat_markdown(content: str) -> str:
    """Keep WeChat Markdown style while making pipe tables mobile-readable."""

    result = markdown_tables_to_key_value_rows(content, bullet="•")
    result = re.sub(r'^\s*---+\s*$', '────────', result, flags=re.MULTILINE)
    return result.strip()


def _format_slack_mrkdwn_unprotected(content: str) -> str:
    """Convert common report Markdown to Slack mrkdwn."""

    result = _markdown_tables_to_key_value_rows_unprotected(content, bullet="•")
    result = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<\2|\1>', result)
    result = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', result)
    result = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', result, flags=re.MULTILINE)
    result = re.sub(r'\*\*(.+?)\*\*', r'*\1*', result)
    result = re.sub(r'^\s*---+\s*$', '────────', result, flags=re.MULTILINE)
    return result.strip()


def format_slack_mrkdwn(content: str) -> str:
    """Convert common report Markdown to Slack mrkdwn."""

    return _transform_outside_fenced_code_blocks(
        content,
        _format_slack_mrkdwn_unprotected,
    ).strip()


def _chunk_by_separators(content: str) -> tuple[list[str], str]:
    """
    通过分割线等特殊字符将消息内容分割为多个区块
    
    Args:
        content: 完整消息内容
        
    Returns:
        sections: 分割后的区块列表
        separator: 区块之间的分隔符，None 表示无法分割
    """
    # 智能分割：优先按 "---" 分隔（股票之间的分隔线）
    # 其次尝试各级标题分割
    if "\n---\n" in content:
        sections = content.split("\n---\n")
        separator = "\n---\n"
    elif "\n# " in content:
        # 按 # 分割 (兼容一级标题)
        parts = content.split("\n## ")
        sections = [parts[0]] + [f"## {p}" for p in parts[1:]]
        separator = "\n"
    elif "\n## " in content:
        # 按 ## 分割 (兼容二级标题)
        parts = content.split("\n## ")
        sections = [parts[0]] + [f"## {p}" for p in parts[1:]]
        separator = "\n"
    elif "\n### " in content:
        # 按 ### 分割
        parts = content.split("\n### ")
        sections = [parts[0]] + [f"### {p}" for p in parts[1:]]
        separator = "\n"
    elif "\n**" in content:
        # 按 ** 加粗标题分割 (兼容 AI 未输出标准 Markdown 标题的情况)
        parts = content.split("\n**")
        sections = [parts[0]] + [f"**{p}" for p in parts[1:]]
        separator = "\n"
    elif "\n" in content:
        # 按 \n 分割
        sections = content.split("\n")
        separator = "\n"
    else:
        return [content], ""
    return sections, separator


def _chunk_by_max_words(content: str, max_words: int, special_char_len: int = 2) -> list[str]:
    """
    按字数分割消息内容
    
    Args:
        content: 完整消息内容
        max_words: 单条消息最大字数
        special_char_len: 每个特殊字符的长度，默认为 2
        
    Returns:
        分割后的区块列表
    """
    if _effective_len(content, special_char_len) <= max_words:
        return [content]
    if max_words < MIN_MAX_WORDS:
        raise ValueError(
            f"max_words={max_words} < {MIN_MAX_WORDS}, 可能陷入无限递归。"
        )

    sections = []
    suffix = TRUNCATION_SUFFIX
    effective_max_words = max_words - len(suffix)  # 预留后缀，避免边界超限
    if effective_max_words <= 0:
        effective_max_words = max_words
        suffix = ""

    while True:
        chunk, content = _slice_at_effective_len(content, effective_max_words, special_char_len)
        if content.strip() != "":
            sections.append(chunk + suffix)
        else:
            # 最后一段了，直接添加并离开循环
            sections.append(chunk)
            break
    return sections


def chunk_content_by_max_words(
    content: str, 
    max_words: int, 
    special_char_len: int = 2,
    add_page_marker: bool = False
    ) -> list[str]:
    """
    按字数智能分割消息内容
    
    Args:
        content: 完整消息内容
        max_words: 单条消息最大字数
        special_char_len: 每个特殊字符的长度，默认为 2
        add_page_marker: 是否添加分页标记
        
    Returns:
        分割后的区块列表
    """
    def _chunk(content: str, max_words: int, special_char_len: int = 2) -> list[str]:
        if max_words < MIN_MAX_WORDS:
            # Safe guard，避免无限递归
            # 理论上，max_words在每次递归中可以减小到无限小，但实际中不太可能发生，
            # 除非每次_chunk_by_separators都能成功返回分隔符，且max_words初始值太小。
            raise ValueError(f"max_words={max_words} < {MIN_MAX_WORDS}, 可能陷入无限递归。")
        
        if _effective_len(content, special_char_len) <= max_words:
            return [content]

        sections, separator = _chunk_by_separators(content)
        if separator == "" and len(sections) == 1:
            # 无法智能分割，则强制按字数分割
            return _chunk_by_max_words(content, max_words, special_char_len)

        chunks = []
        current_chunk = []
        current_word_len = 0
        separator_len = len(separator) if separator else 0
        effective_max_words = max_words - separator_len # 预留分割符长度，避免边界超限

        for section in sections:
            section += separator
            section_word_len = _effective_len(section, special_char_len)

            # 如果单个 section 就超长，需要强制截断
            if section_word_len > max_words:
                # 先保存当前积累的内容
                if current_chunk:
                    chunks.append("".join(current_chunk))

                # 强制截断这个超长 section
                section_chunks = _chunk(
                    section[:-separator_len], effective_max_words, special_char_len
                    )
                section_chunks[-1] = section_chunks[-1] + separator
                chunks.extend(section_chunks)
                continue

            # 检查加入后是否超长
            if current_word_len + section_word_len > max_words:
                # 保存当前块，开始新块
                if current_chunk:
                    chunks.append("".join(current_chunk))
                current_chunk = [section]
                current_word_len = section_word_len
            else:
                current_chunk.append(section)
                current_word_len += section_word_len

        # 添加最后一块
        if current_chunk:
            chunks.append("".join(current_chunk))

        # 移除最后一个块的分割符
        if (chunks and
            len(chunks[-1]) > separator_len and
            chunks[-1][-separator_len:] == separator
        ):
            chunks[-1] = chunks[-1][:-separator_len]
        return chunks
    
    
    if add_page_marker:
        max_words = max_words - PAGE_MARKER_SAFE_LEN
    
    chunks = _chunk(content, max_words, special_char_len)
    if add_page_marker:
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunks[i] = chunk + _page_marker(i, total_chunks)
    return chunks
