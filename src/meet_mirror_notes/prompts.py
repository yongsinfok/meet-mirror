"""Qwen prompt templates for the Phase 2 summarizer.

Spec: docs/superpowers/specs/2026-04-30-meet-mirror-phase2-notes-design.md §6.2
"""

SUMMARY_SYSTEM_PROMPT = (
    "你是专业的会议纪要撰写员。任务:把英文会议转录(含说话人标签和时间戳)整理成"
    "结构化的中文会议纪要。\n"
    "规则:\n"
    "- 只输出中文,严禁输出原文、解释、引号、Markdown 代码块标记\n"
    "- 风格:商务会议纪要,客观、简洁、不带主观评价\n"
    "- 必须按以下章节顺序输出: 概要 / 关键决策 / 行动项 / 未解决问题 / 时间线\n"
    "- 时间线保留 \"Speaker 1/2/...\" 标签和原始时间戳\n"
    "- 行动项格式: \"- [ ] 内容 (负责人:..., 截止:...)\"; 信息缺失写 \"不确定\"\n"
    "- 严禁编造原文中没有的信息;不确定时写 \"不确定\"\n"
    "- 专有名词、产品名、公司名保留英文(如 Salesforce、AWS)"
)

# Map-reduce stage 1: per-window mini-summary
WINDOW_SUMMARY_USER_TEMPLATE = (
    "下面是一段会议的部分转录(时间 {start_min}–{end_min} 分钟)。"
    "请输出 3-6 条中文要点,每条 1 句话,只描述这段内容里发生的事:\n\n"
    "{transcript}"
)

# Map-reduce stage 2: compose final notes from window summaries + raw timeline
COMPOSE_USER_TEMPLATE = (
    "以下是会议各时段的要点汇总和完整时间线。请整理成最终的中文会议纪要,"
    "严格按 概要 / 关键决策 / 行动项 / 未解决问题 / 时间线 五个章节输出。\n\n"
    "## 各时段要点\n{window_summaries}\n\n"
    "## 完整时间线\n{timeline}"
)

# Single-pass prompt for short meetings (<15 min)
SINGLE_PASS_USER_TEMPLATE = (
    "下面是一段完整的英文会议转录,请整理成结构化的中文会议纪要,"
    "严格按 概要 / 关键决策 / 行动项 / 未解决问题 / 时间线 五个章节输出:\n\n"
    "{transcript}"
)
