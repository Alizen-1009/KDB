#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from kb_utils import ROOT, visible_files, wikilinks_in

WIKI = ROOT / "wiki"


def markdown_files(base: Path) -> list[Path]:
    return sorted(p for p in visible_files(base, (".md",)) if not p.name.startswith("_"))


def normalize_target(raw: str) -> str:
    target = raw.split("|", 1)[0].strip()
    return Path(target).stem or Path(target).name


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
        matches = wikilinks_in(text)
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
