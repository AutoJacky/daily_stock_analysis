# 免费千问双 AI 复盘融合

本功能把每日 A股或美股的严格数据复盘、自选股决策摘要交给魔搭 ModelScope 的免费千问做独立复核，再生成一份适合微信阅读的终稿。

## 数据与结论边界

- 程序校验的严格数据报告是指数、涨跌家数、成交额、价格和比例的唯一权威来源。
- 千问只负责指出一致项、分歧、风险动作、机会观察和数据缺口，不能覆盖已校验行情事实。
- 千问输出中出现但两份源报告都不存在的数字会被自动标记为未采用。
- 发送到魔搭的内容只包含市场报告和股票分析，不包含 ModelScope Token、PushPlus Token、账号或其他密钥。
- 仅调用 `https://api-inference.modelscope.cn/v1/chat/completions` 和 `Qwen/Qwen3.5-397B-A17B`。遇到 429、免费容量不足或接口错误会停止，不调用任何收费回退。

## 本地运行

ModelScope Token 默认从 macOS 钥匙串服务 `codex-qwen-free-modelscope` 读取：

```bash
python scripts/fuse_qwen_market_report.py --market cn
python scripts/fuse_qwen_market_report.py --market us --send
```

不加 `--send` 时只生成 `reports/fused_review_<market>_<date>.md`，不会发送微信。

## GitHub Actions 定时推送

云端启用需同时满足：

1. Repository Secret `MODELSCOPE_ACCESS_TOKEN` 已配置。
2. Repository Variable `QWEN_FREE_FUSION_ENABLED=true`。

满足后，北京时间 A股收盘批次和美股收盘批次会关闭原始报告通知，仍保存完整报告，然后生成并发送一份融合终稿。缺少任一配置时保持原有推送方式，避免正式报告静默中断。

手动验收可在 workflow dispatch 中选择 `fusion_market=cn|us`。当 `send_notification=false` 时只生成并上传当天的融合报告，不发送微信；融合器不会读取仓库里旧日期的报告作为本轮输入。
