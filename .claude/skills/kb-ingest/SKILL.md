---
name: kb-ingest
description: 把 raw/ 下的一份原始资料编译进 wiki/——写来源页、更新实体页与概念页、补交叉引用、重建索引、追加日志。当用户说 ingest / 摄入 / 收录 / “把这篇文章（这个 PDF、这段代码）加进知识库”，或直接给出一个 raw/ 下的文件路径时使用。
---

# Ingest：把原始资料编译进 wiki

## 开始前

先读 `AGENTS.md` 的《写作标准》《编辑原则》《Obsidian 约定》三节——那里是这个 vault 的 schema 单一来源，本文件不重复它。再读 `wiki/index.md` 摸清已有页面。

## 第 0 步：定位资料

有路径参数就用它。没有参数时，列出 `raw/` 里还没有对应来源页的文件，让用户选：

```bash
python3 - <<'PY'
from pathlib import Path
root = Path('.')
have = {p.stem for p in (root/'wiki/sources').glob('*.md')}
for f in sorted((root/'raw').rglob('*')):
    if f.is_file() and not f.name.startswith('.') and f.stem not in have:
        print(f)
PY
```

PDF 用 Read 的 `pages` 参数分批读完；长文章分段读到底。**不要只读开头就进入下一步**——这一步偷工，后面所有页面都是错的。

## 阶段 1：先汇报，等确认（硬性 gate）

读完后向用户汇报，然后**停下**：

- `2-3` 条核心摘要
- `1-3` 条值得关注的论断
- 推荐重点关注的对比维度
- 预计会新建 / 更新哪些页面（列出具体文件名）

**得到用户确认前不要写任何文件。** 这是这个知识库最核心的规则：用户要先校准重点，再让你落盘。

## 阶段 2：确认后落盘

1. **来源页** `wiki/sources/<标题>.md`，按 `wiki/sources/_TEMPLATE.md` 的章节结构写。必须包含 `原始文件：` 字段——`scripts/lint.py` 会检查它。命名跟随现有 40 篇的约定：
   - 文章、代码：与 raw 文件 stem 一致（`raw/articles/PageAttention代码走读.md` → `wiki/sources/PageAttention代码走读.md`）
   - 论文：用论文标题，不用 arXiv 编号（`raw/papers/2603.15031v1.pdf` → `wiki/sources/Attention Residuals.md`）
2. **实体页 / 概念页**：先 `ls wiki/concepts wiki/entities` 看已有页面。优先在现有页面里局部增补相关小节，**不重写整页**；只有确实是新概念/新实体才新建，新建时用同目录的 `_TEMPLATE.md`。
3. **冲突**：新资料与已有 wiki 结论矛盾时，在页面里显式标注两种说法及各自来源，不要静默改写旧结论。
4. **交叉引用要双向**：来源页的《关键概念》列出概念页，同时概念页的《相关来源》要回链这篇来源页。只做单向就会在 lint 里变成孤儿页。
5. **链接写法**跟随现状：同目录裸名 `[[MLA]]`，跨目录相对路径 `[[../entities/vLLM]]`。图片用 `![[raw/images/<主题>/<文件名>]]`。
6. **重建索引**（不要手改 `wiki/index.md`，它是生成物）：
   ```bash
   python3 scripts/update_index.py
   ```
7. **追加日志**（不要手编 `wiki/log.md`，格式由脚本保证）：
   ```bash
   python3 scripts/kb_log.py ingest "<资料标题>" \
     -b "读取原始资料：\`raw/papers/xxx.pdf\`" \
     -b "创建来源页：\`wiki/sources/xxx.md\`" \
     -b "更新概念页：\`wiki/concepts/xxx.md\`" \
     -b "未发现与现有 wiki 的直接冲突"
   ```
8. **汇报**：创建了哪些页面、更新了哪些页面、有没有冲突或待核实点。

## 不要做

- 不要在 `inbox/` 生成任务文件——旧的 `ingest.py` 两步流程已废弃，现在直接执行。
- 不要修改 `raw/` 里的原始资料，包括“修正”明显的笔误。
- 不要伪造 benchmark 数字、吞吐、硬件配置或版本号；原文没写就写“待核实”。
- 不要把一次问答的原始对话直接贴进概念页。
