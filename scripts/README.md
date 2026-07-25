# Scripts

这里只放**确定性的文件操作**。三个核心动作（ingest / query / export）本身是 skill，不是脚本——它们需要判断力，写在 `.claude/skills/` 下。脚本负责的是格式必须一致、不该交给 LLM 逐字手写的部分，由 skill 在流程末尾调用。

## `update_index.py`

扫描知识库并重建 `wiki/index.md`（统计 + 实体 / 概念 / 来源 / 报告 / 幻灯片列表 + 最近日志）。

`wiki/index.md` 是生成物，不要手工编辑。

```bash
python3 scripts/update_index.py
```

## `kb_log.py`

向 `wiki/log.md` 追加一条操作日志。日志格式由 `kb_utils.append_log_entry` 统一控制，所以不要手编 `wiki/log.md`。

```bash
python3 scripts/kb_log.py ingest "Attention Residuals" \
  -b "读取原始资料：\`raw/papers/2603.15031v1.pdf\`" \
  -b "创建来源页：\`wiki/sources/Attention Residuals.md\`"
```

## `export_cards.py`

把 `output/cards/*.md` 的问答卡片转换成 Anki 可导入的 TSV，输出到 `output/cards/anki/`。卡片内容由 `kb-export` skill 编写，这个脚本只做机械转换。

```bash
python3 scripts/export_cards.py                      # 全部
python3 scripts/export_cards.py output/cards/attention.md
```

空背面和重复正面会被跳过并打 warning。

## `lint.py` / `health_check.py`

对 wiki 做健康检查：断链、孤儿页、来源页缺 `原始文件：` 字段、空目录。`lint.py` 是 `health_check.py` 的入口包装。

```bash
python3 scripts/lint.py
```

已知问题：目前扫描全仓库所有 `.md`，所以“断链”一节里大部分是 `raw/articles/` 剪藏笔记 frontmatter 里的作者名（`[[方佳瑞​新知答主]]` 这类）造成的假阳性。只有 `wiki/` 和 `output/` 下的结果需要处理。

## `kb_utils.py`

共享工具：路径常量（`ROOT` / `RAW` / `WIKI` / `OUTPUT` / `INBOX`）、日期、slugify、文件遍历、日志追加、wikilink 提取。

全部脚本只用 Python 标准库，`python3` 直接跑，无需 venv 或依赖安装。相对路径参数一律以仓库根目录为基准，而不是当前 shell 的 cwd。
