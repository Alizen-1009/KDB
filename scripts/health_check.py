#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import re

from kb_utils import ROOT, visible_files, wikilinks_in

WIKI = ROOT / "wiki"

FENCE_RE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")


def markdown_files(base: Path) -> list[Path]:
    return sorted(p for p in visible_files(base, (".md",)) if not p.name.startswith("_"))


def strip_non_prose(text: str) -> str:
    """去掉 YAML frontmatter、代码块和行内代码。

    这三处的 `[[...]]` 不是真链接：剪藏笔记的 `author:` 字段、文档里的示例写法、
    脚本用法说明。不剔掉它们，断链报告会被假阳性淹没。
    """
    lines = text.splitlines()
    start = 0
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() in {"---", "..."}:
                start = index + 1
                break

    kept: list[str] = []
    in_fence = False
    for line in lines[start:]:
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        kept.append(INLINE_CODE_RE.sub("", line))
    return "\n".join(kept)


def normalize_target(raw: str) -> str:
    """把 `[[../concepts/名字.md|别名]]` 归一成 `名字`。

    不能用 Path.stem——文件名里可能带小数点（如“提升 1.7 倍”），会被截断。
    """
    target = raw.split("|", 1)[0].split("#", 1)[0].strip()
    target = target.rsplit("/", 1)[-1]
    if target.endswith(".md"):
        target = target[:-3]
    return target


def is_placeholder_target(target: str) -> bool:
    placeholders = {
        "示例摘要",
        "相关概念A",
        "相关概念B",
        "概念A",
        "概念B",
        "相关实体",
        "相关来源摘要",
        "_TEMPLATE",
        "Wiki Links",
        "双向链接",
        "文件名",
    }
    return target in placeholders


def main() -> None:
    files = markdown_files(ROOT)
    known = {p.stem for p in files}
    problems: list[str] = []
    backlinks: dict[str, int] = {p.stem: 0 for p in files}
    source_reference_issues: list[str] = []

    for file in files:
        text = file.read_text(encoding="utf-8", errors="ignore")
        matches = wikilinks_in(strip_non_prose(text))
        for match in matches:
            target = normalize_target(match)
            if is_placeholder_target(target):
                continue
            if target and target not in known:
                rel = file.relative_to(ROOT)
                problems.append(f"{rel}: 缺失链接目标 [[{match}]]")
            elif target in backlinks:
                backlinks[target] += 1

        if file.parent == WIKI / "sources" and "原始文件：" not in text:
            source_reference_issues.append(
                f"{file.relative_to(ROOT)}: 缺少“原始文件”来源字段"
            )

    empty_dirs = []
    for relative in [
        "raw/articles",
        "raw/papers",
        "raw/repos",
        "raw/datasets",
        "raw/images",
        "raw/code",
        "wiki/sources",
        "wiki/entities",
        "wiki/concepts",
        "output/reports",
        "output/slides",
        "output/visuals",
    ]:
        path = ROOT / relative
        if path.exists() and not any(path.iterdir()):
            empty_dirs.append(relative)

    orphan_pages = []
    for file in files:
        if WIKI not in file.parents:
            continue
        if file == WIKI / "index.md" or file == WIKI / "log.md":
            continue
        if backlinks.get(file.stem, 0) == 0:
            orphan_pages.append(str(file.relative_to(ROOT)))

    print("=== Health Check ===")
    if problems:
        print("Broken wiki links:")
        for problem in problems:
            print(f"- {problem}")
    else:
        print("No broken wiki links found.")

    print("")
    if source_reference_issues:
        print("Source pages missing source references:")
        for item in source_reference_issues:
            print(f"- {item}")
    else:
        print("All source pages contain source references.")

    print("")
    if orphan_pages:
        print("Orphan pages:")
        for item in orphan_pages:
            print(f"- {item}")
    else:
        print("No orphan pages detected.")

    print("")
    if empty_dirs:
        print("Empty directories:")
        for item in empty_dirs:
            print(f"- {item}")
    else:
        print("No empty directories detected.")


if __name__ == "__main__":
    main()
