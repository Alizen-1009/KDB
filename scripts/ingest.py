#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from kb_utils import INBOX, ROOT, append_log_entry, now_date, now_stamp, slugify
from update_index import main as update_index


def detect_kind(path: Path) -> str:
    parts = set(path.parts)
    if "articles" in parts:
        return "article"
    if "papers" in parts:
        return "paper"
    if "repos" in parts:
        return "repo"
    if "datasets" in parts:
        return "dataset"
    if "images" in parts:
        return "image"
    if "code" in parts:
        return "code"
    return "source"


def build_task(source: Path) -> str:
    kind = detect_kind(source)
    title = source.stem
    rel = source.relative_to(ROOT).as_posix()

    return f"""# Ingest 任务：{title}

## 基本信息

- 日期：{now_date()}
- 类型：{kind}
- 原始文件：[[../{rel}|{source.name}]]

## LLM 执行要求

### 阶段 1：先阅读，再与用户确认

1. 阅读完整原始资料，不要跳过正文。
2. 先向用户汇报：
   - `2-3` 条核心摘要
   - `1-3` 条值得关注的论断
   - 推荐重点关注的对比维度
3. 明确等待用户确认或补充重点后，再进入写入阶段。

### 阶段 2：确认后再落盘

4. 在 `wiki/sources/` 中创建或更新对应来源摘要页。
5. 检查是否需要更新以下页面类型：
   - `wiki/entities/`
   - `wiki/concepts/`
6. 如果该资料引入新的概念、项目、人物、系统、硬件或 benchmark，请新增相应页面或在现有页面补充。
7. 检查新来源与已有 wiki 内容之间是否有冲突，并在页面中显式标注。
8. 更新所有受影响页面的交叉引用。
9. 更新 `wiki/index.md`。
10. 追加到 `wiki/log.md`。
11. 最后向用户报告：
    - 创建了哪些页面
    - 更新了哪些页面
    - 是否发现矛盾或待核实点

## 人工关注点

- 这份资料最重要的结论是什么？
- 它改变了哪些已有认识？
- 它与现有 wiki 中哪些概念或实体最相关？
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an ingest task for a raw source.")
    parser.add_argument("source", help="Path to a file under raw/")
    args = parser.parse_args()

    source = (
        (ROOT / args.source).resolve()
        if not Path(args.source).is_absolute()
        else Path(args.source).resolve()
    )
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")
    if ROOT not in source.parents and source != ROOT:
        raise SystemExit("Source must be inside the knowledge-base root.")

    task_name = f"{now_stamp()}-ingest-{slugify(source.stem)}.md"
    task_path = INBOX / task_name
    task_path.write_text(build_task(source), encoding="utf-8")

    append_log_entry(
        "ingest",
        source.stem,
        [
            f"登记原始资料：`{source.relative_to(ROOT).as_posix()}`",
            f"创建待处理任务：`inbox/{task_name}`",
            "本次 ingest 采用先讨论后落盘的两阶段流程",
        ],
    )
    update_index()
    print(f"Created ingest task: {task_path}")


if __name__ == "__main__":
    main()
