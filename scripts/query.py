#!/usr/bin/env python3

from __future__ import annotations

import argparse

from kb_utils import OUTPUT, ROOT, append_log_entry, now_date, now_stamp, slugify
from update_index import main as update_index


def build_query_doc(question: str, fmt: str) -> str:
    return f"""# 研究问题

{question}

## 请求信息

- 日期：{now_date()}
- 输出格式：{fmt}

## LLM 执行建议

1. 先阅读 `AGENTS.md`、`wiki/index.md`、`wiki/log.md`。
2. 找出与该问题最相关的概念页、来源页、实体页。
3. 基于 wiki 进行综合分析，不直接跳回所有原始资料，除非 wiki 信息不足。
4. 输出时明确区分：
   - 已知事实
   - 实现差异
   - 性能权衡
   - 待核实问题
5. 如果形成了高价值新结论，可考虑回填到 `wiki/`。

## 结果结构建议

- 背景
- 核心观点
- 机制拆解
- 对比分析
- 工程含义
- 可进一步研究的问题
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a query task document.")
    parser.add_argument("question", help="Research question")
    parser.add_argument(
        "--format",
        default="report",
        choices=["report", "slides"],
        help="Output format",
    )
    args = parser.parse_args()

    folder = OUTPUT / ("reports" if args.format == "report" else "slides")
    name = f"{now_stamp()}-{slugify(args.question[:40])}.md"
    path = folder / name
    path.write_text(build_query_doc(args.question, args.format), encoding="utf-8")

    append_log_entry(
        "query",
        args.question[:60],
        [
            f"创建查询任务：`{path.relative_to(ROOT).as_posix()}`",
            f"输出格式：`{args.format}`",
        ],
    )
    update_index()
    print(f"Created query file: {path}")


if __name__ == "__main__":
    main()
