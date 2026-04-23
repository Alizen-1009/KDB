#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from kb_utils import ROOT, visible_files


def list_files(base: Path, exts: tuple[str, ...] | None = None) -> list[Path]:
    return visible_files(base, exts)


def section(title: str, items: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if items:
        lines.extend(items)
    else:
        lines.append("- 暂无")
    lines.append("")
    return lines


def rel_link(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    name = path.stem
    no_suffix = relative[:-3] if relative.endswith(".md") else relative
    return f"- [[../{no_suffix}|{name}]]"


def recent_log_items(log_path: Path, limit: int = 8) -> list[str]:
    if not log_path.exists():
        return []
    entries = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("## ["):
            entries.append(f"- {line.removeprefix('## ').strip()}")
    return list(reversed(entries[-limit:]))


def main() -> None:
    raw_dir = ROOT / "raw"
    wiki_dir = ROOT / "wiki"
    output_dir = ROOT / "output"

    source_files = list_files(wiki_dir / "sources", (".md",))
    entity_files = list_files(wiki_dir / "entities", (".md",))
    concept_files = list_files(wiki_dir / "concepts", (".md",))
    report_files = list_files(output_dir / "reports", (".md",))
    slide_files = list_files(output_dir / "slides", (".md",))
    recent_logs = recent_log_items(wiki_dir / "log.md")

    article_count = len(list_files(raw_dir / "articles"))
    paper_count = len(list_files(raw_dir / "papers"))
    repo_count = len(list_files(raw_dir / "repos"))
    dataset_count = len(list_files(raw_dir / "datasets"))
    image_count = len(list_files(raw_dir / "images"))
    code_count = len(list_files(raw_dir / "code"))

    lines: list[str] = [
        "# AI Infra 知识库索引",
        "",
        "该页面由 `scripts/update_index.py` 自动生成。",
        "",
        "## 入口",
        "",
        "- [[../00 Home|Vault Home]]",
        "- [[../AGENTS|AGENTS]]",
        "- [[log|操作日志]]",
        "- [[../inbox/README|Inbox]]",
        "- [[../output/README|Output]]",
        "",
        "## 资源统计",
        "",
        f"- 原始文章：{article_count}",
        f"- 原始论文：{paper_count}",
        f"- 原始仓库：{repo_count}",
        f"- 原始数据集：{dataset_count}",
        f"- 原始图片：{image_count}",
        f"- 原始代码：{code_count}",
        f"- 来源文件：{len([p for p in source_files if not p.name.startswith('_')])}",
        f"- 实体文件：{len([p for p in entity_files if not p.name.startswith('_')])}",
        f"- 概念文件：{len([p for p in concept_files if not p.name.startswith('_')])}",
        f"- 报告文件：{len(report_files)}",
        f"- 幻灯片文件：{len(slide_files)}",
        "",
    ]

    lines.extend(
        section(
            "实体页面",
            [rel_link(p) for p in entity_files if not p.name.startswith("_")],
        )
    )
    lines.extend(
        section(
            "概念页面",
            [rel_link(p) for p in concept_files if not p.name.startswith("_")],
        )
    )
    lines.extend(
        section(
            "来源摘要",
            [rel_link(p) for p in source_files if not p.name.startswith("_")],
        )
    )
    lines.extend(section("最近日志", recent_logs))
    lines.extend(section("报告", [rel_link(p) for p in report_files]))
    lines.extend(section("幻灯片", [rel_link(p) for p in slide_files]))

    target = wiki_dir / "index.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated {target}")


if __name__ == "__main__":
    main()
