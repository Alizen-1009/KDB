---
name: kb-query
description: 基于 wiki/ 回答研究问题——架构对比、性能归因、机制拆解、实现路线评估——默认产出 output/reports/ 下的自包含 HTML 报告（面试向的放 output/interview/，幻灯片放 output/slides/），并把高价值结论回填 wiki/。当用户说 query / 出个报告 / 写篇对比，或提出“对比 X 与 Y”“为什么 X 更快”这类研究问题时使用。
---

# Query：基于 wiki 产出研究输出

## 开始前

读 `AGENTS.md` 的《写作标准》——尤其是“区分原文说法 / 实现细节 / 实验观察 / LLM 归纳”和“不伪造 benchmark 数字”。

## 检索顺序

1. 先读 `wiki/index.md`（按主题分组），锁定主题后读 `wiki/maps/<topic>.md`——它的《导读》写了这个主题的阅读顺序和主干页面，比在 82 个概念里瞎翻快得多。
2. 读这些 wiki 页面，它们才是这个库的检索入口。
3. **只有 wiki 信息不足时**才回 `raw/` 读原始资料——这个库的设计前提就是知识已经编译过一遍，不是每次问答都重新 RAG 原文。
4. 涉及具体实现细节（kernel grid 映射、调度器行为、API 签名）时，交叉核对官方源码或文档，并在报告里写明核对来源。

## 输出格式 gate

篇幅较长、面向人阅读的技术报告、机制讲解、架构对比和性能复盘，默认写到 `output/reports/<描述性中文标题>.html`。创建前读取并遵循全局 `html-artifacts` skill；它是 HTML 信息架构、可移植性、可访问性和浏览器验证的单一来源，本 skill 不重复那些规则。

保留 Markdown 的情况：短而线性的研究备忘、明确要求在 Obsidian 中频繁 diff/edit 的源文档、机器消费格式，或用户明确指定 `.md`。不要同时创建同名 HTML 和 Markdown 副本；用户明确格式优先。**文件名不加时间戳**。

**面试备考类仍写到 `output/interview/<标题>.md`**，因为它是 Obsidian/复习卡片选料源；要 Marp 幻灯片时写到 `output/slides/<标题>.md`。

无论载体是什么，内容结构参考 `Templates/Query Report Template.md`：

- 背景
- 核心观点
- 机制拆解
- 对比分析
- 工程含义
- 可进一步研究的问题

内容上必须把这四类分开，不要混成一段散文：

- **已知事实**：有来源支撑的
- **实现差异**：框架/版本之间的具体不同
- **性能权衡**：代价和适用边界
- **待核实**：推断、超出知识截止的 API、没有一手来源的数字

HTML 内回链 wiki 使用普通相对链接，如 `../../wiki/concepts/KV Cache.md`；wiki 页面回链 HTML 报告使用普通 Markdown 链接 `[报告](../../output/reports/报告.html)`。Markdown output 才使用 `[[../../wiki/concepts/KV Cache|KV Cache]]`。

## 收尾

1. **回填**：如果产生了新的、稳定的结论，写回相关概念页或实体页——只停在报告里的结论等于没有进知识库。回填后重跑索引。
2. ```bash
   python3 scripts/kb_meta.py sync
   python3 scripts/update_index.py
   python3 scripts/lint.py
   python3 scripts/kb_log.py query "<问题标题>" \
     -b "读取概念页：\`FlashAttention\`、\`GPU执行模型\`" \
     -b "创建报告：\`output/reports/xxx.html\`" \
     -b "更新概念页：\`wiki/concepts/xxx.md\`，补充 …" \
     -b "待核实：…"
   ```
3. 向用户汇报：报告写在哪、回填了哪些页面、哪些点待核实。
