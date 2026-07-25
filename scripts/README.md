# Scripts

这里只放**确定性的文件操作**。三个核心动作（ingest / query / export）本身是 skill，不是脚本——它们需要判断力，写在 `.pi/skills/` 下。脚本负责的是格式必须一致、不该交给 LLM 逐字手写的部分，由 skill 在流程末尾调用。

## `update_index.py`

重建 `wiki/index.md` 和 `wiki/maps/<主题>.md`。分组依据是页面 frontmatter 的 `topic` / `entity_type` / `source_kind`，所以页面必须先有 frontmatter。

- `wiki/index.md` 整页是生成物，不要手工编辑。
- `wiki/maps/<主题>.md` 只有 `BEGIN AUTO` 标记之后的部分是生成物；**标记之前的《导读》是手写区，重新生成时保留**。想写“这个主题按什么顺序读”，写在那里。

```bash
python3 scripts/update_index.py
```

## `kb_meta.py`

frontmatter 的词表、校验与派生字段维护。

```bash
python3 scripts/kb_meta.py check   # 校验 type / topic / entity_type / source_kind 是否在词表内
python3 scripts/kb_meta.py sync    # 从实际链接数和 git 提交日期刷新 sources / updated
```

`TOPICS` / `ENTITY_TYPES` / `SOURCE_KINDS` 三个词表定义在这个文件里，是唯一来源；`AGENTS.md` 里那份是给人看的副本，改词表要同时改。

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

对 wiki 做健康检查：断链、**链接路径写错**（名字对但相对路径错，Obsidian 里点不开）、frontmatter 是否合规、孤儿页、来源页缺 `原始文件：` 字段、空目录。`lint.py` 是 `health_check.py` 的入口包装。

```bash
python3 scripts/lint.py
```

扫描全仓库所有 `.md`，但会先剔掉 YAML frontmatter、代码块和行内代码里的 `[[...]]`——剪藏笔记的 `author:` 字段和文档里的示例写法都不是真链接。链接目标归一化时只剥 `.md` 后缀和 `#` 锚点，不用 `Path.stem`（文件名里可能带小数点，比如“提升 1.7 倍”）。

所以这份输出现在应该是零噪音的：报出来的断链就是真断链。

## `kb_utils.py`

共享工具：路径常量（`ROOT` / `RAW` / `WIKI` / `OUTPUT` / `INBOX`）、日期、slugify、文件遍历、日志追加、wikilink 提取。

全部脚本只用 Python 标准库，`python3` 直接跑，无需 venv 或依赖安装。相对路径参数一律以仓库根目录为基准，而不是当前 shell 的 cwd。
