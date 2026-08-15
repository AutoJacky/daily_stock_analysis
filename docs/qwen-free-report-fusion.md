# 免费千问双 AI 复盘融合

本功能把每日 A股或美股的严格数据复盘、自选股决策摘要交给魔搭 ModelScope 的免费千问做独立复核，再生成一份适合微信阅读的终稿。

## 数据与结论边界

- 程序校验的严格数据报告是指数、涨跌家数、成交额、价格和比例的唯一权威来源。
- 千问只负责指出一致项、分歧、风险动作、机会观察和数据缺口，不能覆盖已校验行情事实。
- 千问输出中出现但两份源报告都不存在的数字会被自动标记为未采用。
- 发送到魔搭的内容只包含市场报告和股票分析，不包含 ModelScope Token、PushPlus Token、账号或其他密钥。
- 仅调用 `https://api-inference.modelscope.cn/v1/chat/completions` 和 `Qwen/Qwen3.5-397B-A17B`。遇到 429、免费容量不足或接口错误时跳过千问审计并明确标注，程序校验报告仍照常生成；绝不调用任何收费回退。

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

三源融合启用后，A股批次为北京时间工作日 16:00，美股批次为北京时间工作日次日 05:45；港股/日股批次仍为 16:15。时间差用于等待千问原生任务完成并由本机桥接上传。

手动验收可在 workflow dispatch 中选择 `fusion_market=cn|us`。当 `send_notification=false` 时只生成并上传当天的融合报告，不发送微信；融合器不会读取仓库里旧日期的报告作为本轮输入。

## 机构框架与三源融合

终稿按以下证据优先级处理：

1. 程序校验数据：唯一权威数字源，字段附状态、来源与数据日。
2. 千问客户端原生定时报告：观点源；数字必须被程序数据再次确认。
3. 魔搭免费千问：审计源；只提炼一致项、分歧、风险和机会。

机构框架包含估值温度计、利率与 ERP 代理、全球市场联动、资金与杠杆、行业估值/景气状态、自选股机构卡及无伪概率情景。免费源未覆盖的科创50同口径估值、ETF申赎、行业高频景气、机构一致目标价等会显示 `missing` 或 `not_supported`，不会由模型补造。

## 私有账户级量化操作单

可选的 Repository Secret `PERSONAL_PORTFOLIO_B64` 保存版本化个人持仓 JSON。程序会用公开基金逐日复权收益计算 20/60 日动量、60 日波动率、120 日 Sharpe 和最大回撤，按持仓内横截面因子分数形成前 20% / 中间 60% / 后 20%。因子检验每周只使用当时已知净值排序，再计算下一周前 20% 与后 20% 的真实涨跌差及年化 Sharpe，避免用当前排名回看过去所产生的前视偏差。低分组在普通基金账户中解释为减少、停止新增或合并重复基金，绝不表述为可以实际做空。

精确持仓只在最终 PushPlus 发送前由确定性程序读取：不进入魔搭千问请求，不写入仓库报告文件和 Actions artifact，也不打印到日志。云端报告附件仍只有脱敏市场底稿。现金余额未知、净值覆盖不足或持仓快照过期时，程序禁止立即加仓；场外基金只使用每日净值与 MA20/MA60 条件，不伪造盘中买入价。持仓发生申购、赎回、分红或产品到期后需要更新 Secret，系统不能绕过银行/基金平台授权自行读取账户或执行交易。

## 接入千问客户端定时报告

完整的新建任务名称、时间与可直接复制的提示词见 [qwen-native-task-prompts.md](qwen-native-task-prompts.md)。

千问客户端没有公开稳定的定时任务结果 API。请在两个千问任务的 Prompt 末尾增加保存要求，让任务把当天最终 Markdown 同时写到固定文件，例如：

```text
完成报告后，必须将完整 Markdown 原文覆盖保存到：
/Users/yueyue/Documents/qwen-finance-reports/cn/latest.md
首行写“# A股盘后复盘报告”，正文必须包含报告日期、数据来源和数据日。
```

美股任务保存到 `.../us/latest.md`，首行写 `# 美股科技半导体分析`。美股载荷按纽约市场日期校验，避免北京时间跨日后误判为过期。任务完成后运行：

```bash
python scripts/publish_qwen_native_report.py --market cn \
  --report /Users/yueyue/Documents/qwen-finance-reports/cn/latest.md \
  --gh .tools/gh/gh_2.96.0_macOS_arm64/bin/gh

python scripts/publish_qwen_native_report.py --market us \
  --report /Users/yueyue/Documents/qwen-finance-reports/us/latest.md \
  --gh .tools/gh/gh_2.96.0_macOS_arm64/bin/gh
```

脚本只把正文写入 GitHub Encrypted Secret，不提交 Git 历史、不读取千问登录态。云端只接受当天且市场匹配的载荷；旧报告和错市场报告会被拒绝。

macOS 可安装每 5 分钟检查一次的本机桥接（免费、无常驻云服务）：

```bash
python scripts/install_qwen_report_bridge_macos.py
```

桥接只在上述固定文件当天发生变化时上传，正文通过标准输入交给 `gh secret set` 并在本机加密，不出现在命令行参数或日志中。
