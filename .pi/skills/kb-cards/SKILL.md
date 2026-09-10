---
name: kb-cards
description: 从 wiki/ 的概念页、实体页和 output/interview/ 的面试备考资料里生成复习卡片——写成 output/cards/ 下的 Markdown 问答，再用 scripts/export_cards.py 转成 Anki 可导入的 TSV。当用户说导出卡片、生成 Anki、做复习卡或“我要背这个主题”时使用；普通 HTML 导出使用 kb-export。
---

# Cards：把 wiki 制作成复习卡片

> [!note] 格式边界
> 本 skill 的 Markdown 是 `scripts/export_cards.py` 的机器输入和 Anki 卡片事实源，不适用人读报告的 HTML 默认规则；产物仍是 `output/cards/<主题>.md` 与 `output/cards/anki/<主题>.txt`。

## 一次只导一个主题

问清主题（或从用户给的页面推断），一个主题一个文件：`output/cards/<主题>.md`。主题粒度参考 `attention`、`kv-cache`、`并行策略`、`cuda-优化`、`moe`，不要一次导出整个 wiki——卡片质量比数量重要得多。

## 选料

按适合出卡的程度排序：

1. **概念页的《定义》《核心机制》《关键权衡》** ——最适合，本来就是压缩过的知识点。
2. **`output/interview/` 里的面试备考资料**（现有 11 篇）——已经是问答形式，基本可以直接转。
3. 实体页的《核心信息》——适合出“X 是谁做的 / 解决什么问题”这类事实卡。

**不要出卡的内容**：页面里标了“待核实”的结论；没有一手来源的 benchmark 数字；依赖特定版本且原文没写清版本的实现细节。`AGENTS.md` 禁止伪造这类事实，卡片会被反复记诵，错了代价更大。

## 卡片文件格式

```markdown
---
type: cards
deck: AI Infra::Attention
tags: attention flashattention
source: [[../../wiki/concepts/FlashAttention|FlashAttention]]
---

### FlashAttention 为什么能降低显存占用？

- 不物化 N×N 的注意力矩阵，改为按 tile 分块计算。
- 用 online softmax 递推维护 `m`、`l`，无需保留完整 score 矩阵。
- 代价是重算：反向传播时需要重新计算部分中间量。
```

- `deck` 是 Anki 牌组名，`::` 表示子牌组。
- `tags` 空格分隔，单个 tag 内不能有空格。
- 每个 `### ` 标题是卡片正面，到下一个 `### ` 之间是背面。

## 卡片质量规则

- **一卡一个知识点**。想在一张卡里塞完整机制说明，就是没想清楚该拆成几张。
- **问题要具体**：问“FlashAttention 为什么能降低显存占用”，不要问“介绍一下 FlashAttention”。
- **背面 3-6 个 bullet**，超了就拆卡。
- **每张卡独立成立**，不能依赖上一张卡的上下文（Anki 会打乱顺序）。
- 术语首次出现用中文+英文原词，方便和 wiki 页面对上。

## 转换与导入

```bash
python3 scripts/export_cards.py output/cards/<主题>.md   # 单个主题
python3 scripts/export_cards.py                          # 全部重新导出
```

产物是 `output/cards/anki/<主题>.txt`。转换器行为：`[[链接]]` 抹成纯文本、换行变 `<br>`、空背面和重复正面会被跳过并打 warning——**看到 warning 要回去修卡片源文件**，不要只看总数。

在 Anki 里：文件 → 导入 → 选这个 `.txt`，牌组和制卡类型由文件头的 `#deck` / `#notetype` 决定，无需手选。

## 收尾

```bash
python3 scripts/kb_log.py cards "<主题>复习卡片" \
  -b "来源页面：\`wiki/concepts/FlashAttention\`、\`wiki/concepts/Online Softmax\`" \
  -b "创建卡片：\`output/cards/attention.md\`（12 张）" \
  -b "导出：\`output/cards/anki/attention.txt\`"
```

汇报卡片张数、跳过了哪些内容（尤其是因为“待核实”而没出卡的点）。
