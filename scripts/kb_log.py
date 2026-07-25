#!/usr/bin/env python3
"""向 wiki/log.md 追加一条操作日志。

日志格式由 kb_utils.append_log_entry 统一控制，因此不要手工编辑 wiki/log.md。

用法：

    python3 scripts/kb_log.py ingest "Attention Residuals" \\
        -b "读取原始资料：\\`raw/papers/2603.15031v1.pdf\\`" \\
        -b "创建来源页：\\`wiki/sources/Attention Residuals.md\\`"
"""

from __future__ import annotations

import argparse

from kb_utils import append_log_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Append an entry to wiki/log.md.")
    parser.add_argument("action", help="动作类型，例如 ingest / query / export / lint")
    parser.add_argument("title", help="本次操作的标题")
    parser.add_argument(
        "-b",
        "--bullet",
        action="append",
        default=[],
        dest="bullets",
        help="一条日志要点，可重复传入",
    )
    args = parser.parse_args()

    if not args.bullets:
        raise SystemExit("至少需要一条 -b/--bullet 要点。")

    append_log_entry(args.action, args.title, args.bullets)
    print(f"Appended log entry: [{args.action}] {args.title}")


if __name__ == "__main__":
    main()
