#!/usr/bin/env python3
"""wiki 页面 frontmatter 的词表、读取与维护。

`type` / `topic` / `entity_type` / `source_kind` 是人写的（由 kb-ingest skill 填），
`sources` 与 `updated` 是派生的，用 `sync` 子命令从实际链接和 git 提交日期刷新，
不要手工维护。

    python3 scripts/kb_meta.py check    # 校验全部 wiki 页面的 frontmatter
    python3 scripts/kb_meta.py sync     # 刷新 sources 与 updated
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from kb_utils import ROOT, WIKI, now_date, visible_files

TOPICS = [
    "注意力机制",
    "KV Cache",
    "推理服务",
    "并行与分布式",
    "GPU 编程",
    "性能分析",
    "模型架构",
    "投机解码",
    "训练与 Scaling",
    "位置编码",
]
ENTITY_TYPES = ["项目", "框架", "模型", "公司", "组织", "人物", "课程", "硬件", "benchmark"]
SOURCE_KINDS = ["文章", "论文", "课程", "代码", "面试整理", "截图整理", "repo", "数据集"]

FOLDER_TYPES = {"concepts": "concept", "entities": "entity", "sources": "source"}
SOURCE_LINK_RE = re.compile(r"\[\[\.\./sources/([^\]|]+)")


def wiki_pages() -> list[Path]:
    pages: list[Path] = []
    for folder in FOLDER_TYPES:
        pages.extend(
            p for p in visible_files(WIKI / folder, (".md",)) if not p.name.startswith("_")
        )
    return sorted(pages)


def parse_frontmatter(text: str) -> tuple[dict[str, str], int]:
    """返回 (字段字典, frontmatter 结束后的字符下标)。没有 frontmatter 时返回 ({}, 0)。"""
    if not text.startswith("---\n"):
        return {}, 0
    end = text.find("\n---", 4)
    if end == -1:
        return {}, 0
    fields: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields, end + 5


def check() -> list[str]:
    problems: list[str] = []
    for page in wiki_pages():
        rel = page.relative_to(ROOT).as_posix()
        fields, _ = parse_frontmatter(page.read_text(encoding="utf-8"))
        if not fields:
            problems.append(f"{rel}: 缺少 frontmatter")
            continue

        expected = FOLDER_TYPES[page.parent.name]
        if fields.get("type") != expected:
            problems.append(f"{rel}: type 应为 `{expected}`，实际 `{fields.get('type')}`")
        if fields.get("topic") not in TOPICS:
            problems.append(f"{rel}: topic `{fields.get('topic')}` 不在词表中")
        if expected == "entity" and fields.get("entity_type") not in ENTITY_TYPES:
            problems.append(f"{rel}: entity_type `{fields.get('entity_type')}` 不在词表中")
        if expected == "source" and fields.get("source_kind") not in SOURCE_KINDS:
            problems.append(f"{rel}: source_kind `{fields.get('source_kind')}` 不在词表中")
        if not fields.get("updated"):
            problems.append(f"{rel}: 缺少 updated")
    return problems


def git_date(page: Path) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ad", "--date=short", "--", str(page)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return result.stdout.strip() or now_date()


def sync() -> list[str]:
    """维护派生字段。

    `sources` 每次都按实际链接重算。`updated` 只在缺失时补（用 git 最后提交日期兜底）——
    改页面的人负责把它改成当天，脚本不去覆盖：一次批量元数据变更会让所有文件在 git 里
    都变成“今天提交过”，从 git 反推日期只会把这个字段刷成一片今天，反而丢掉“哪些页面
    很久没动”这个唯一有用的信号。
    """
    changed: list[str] = []
    for page in wiki_pages():
        text = page.read_text(encoding="utf-8")
        fields, body_start = parse_frontmatter(text)
        if not fields:
            continue

        updates: dict[str, str] = {}
        if not fields.get("updated"):
            updates["updated"] = git_date(page)
        if fields.get("type") != "source":
            updates["sources"] = str(len(set(SOURCE_LINK_RE.findall(text))))

        new_fields = dict(fields)
        new_fields.update(updates)
        if new_fields == fields:
            continue

        rendered = "---\n" + "".join(f"{k}: {v}\n" for k, v in new_fields.items()) + "---\n"
        page.write_text(rendered + text[body_start:], encoding="utf-8")
        changed.append(page.relative_to(ROOT).as_posix())
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and refresh wiki frontmatter.")
    parser.add_argument("command", choices=["check", "sync"])
    args = parser.parse_args()

    if args.command == "check":
        problems = check()
        if problems:
            print("Frontmatter 问题：")
            for problem in problems:
                print(f"- {problem}")
            raise SystemExit(1)
        print(f"Frontmatter OK（{len(wiki_pages())} 个页面）")
    else:
        changed = sync()
        for path in changed:
            print(f"updated {path}")
        print(f"刷新 {len(changed)} 个页面")


if __name__ == "__main__":
    main()
