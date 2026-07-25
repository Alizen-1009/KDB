---
name: kb-query
description: 基于 wiki/ 回答研究问题——架构对比、性能归因、机制拆解、实现路线评估——产出 output/reports/ 下的报告（面试向的放 output/interview/，幻灯片放 output/slides/），并把高价值结论回填 wiki/。当用户说 query / 出个报告 / 写篇对比，或提出“对比 X 与 Y”“为什么 X 更快”这类研究问题时使用。
---

# Query：基于 wiki 产出研究输出

## 开始前

读 `AGENTS.md` 的《写作标准》——尤其是“区分原文说法 / 实现细节 / 实验观察 / LLM 归纳”和“不伪造 benchmark 数字”。

## 检索顺序

1. 先读 `wiki/index.md`，挑出相关的概念页、实体页、来源页。
2. 读这些 wiki 页面，它们才是这个库的检索入口。
3. **只有 wiki 信息不足时**才回 `raw/` 读原始资料——这个库的设计前提就是知识已经编译过一遍，不是每次问答都重新 RAG 原文。
4. 涉及具体实现细节（kernel grid 映射、调度器行为、API 签名）时，交叉核对官方源码或文档，并在报告里写明核对来源。

## 输出

技术报告写到 `output/reports/<描述性中文标题>.md`，**面试备考类写到 `output/interview/`**（这两类内容的组织方式不同，已经分开）。**文件名不加时间戳**——现有文件都是纯标题，早期脚本的 `20260722-xxx` 前缀已废弃。

要 Marp 幻灯片时写到 `output/slides/`，同样的命名规则。

推荐结构（`Templates/Query Report Template.md` 是同一套）：

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

回链 wiki 页面用 `[[../../wiki/concepts/KV Cache|KV Cache]]` 形式（`output/reports/` 和 `output/interview/` 都深两层，`../wiki/...` 是错的）。

## 收尾

1. **回填**：如果产生了新的、稳定的结论，写回相关概念页或实体页——只停在报告里的结论等于没有进知识库。回填后重跑索引。
2. ```bash
   python3 scripts/update_index.py
   python3 scripts/kb_log.py query "<问题标题>" \
     -b "读取概念页：\`FlashAttention\`、\`GPU执行模型\`" \
     -b "创建报告：\`output/reports/xxx.md\`" \
     -b "更新概念页：\`wiki/concepts/xxx.md\`，补充 …" \
     -b "待核实：…"
   ```
3. 向用户汇报：报告写在哪、回填了哪些页面、哪些点待核实。
