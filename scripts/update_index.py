#!/usr/bin/env python3
"""重建 `wiki/index.md` 与 `wiki/maps/<主题>.md`。

索引按 frontmatter 的 `topic` / `entity_type` / `source_kind` 分组，所以页面必须先有
frontmatter（`python3 scripts/kb_meta.py check` 会校验）。

主题地图页顶部的手写导读区会被保留：脚本只重写 `AUTO_BEGIN` 标记之后的内容。
"""

from __future__ import annotations

from pathlib import Path

from kb_meta import (
    ENTITY_TYPES,
    SOURCE_KINDS,
    TOPICS,
    parse_frontmatter,
    wiki_pages,
)
from kb_utils import ROOT, visible_files

AUTO_BEGIN = "<!-- BEGIN AUTO：以下由 scripts/update_index.py 生成，改动会被覆盖 -->"
AUTO_HINT = "<!-- 手写导读区：这个主题该按什么顺序读、哪几页是主干。重新生成时会保留。 -->"


def list_files(base: Path, exts: tuple[str, ...] | None = None) -> list[Path]:
    return visible_files(base, exts)


def section(title: str, items: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    lines.extend(items or ["- 暂无"])
    lines.append("")
    return lines


def prefer_html_artifacts(files: list[Path]) -> list[Path]:
    """按目录和 stem 去重报告；同名 HTML 优先于旧 Markdown 副本。"""
    selected: dict[tuple[Path, str], Path] = {}
    for path in sorted(files):
        key = (path.parent, path.stem)
        current = selected.get(key)
        if current is None or path.suffix.lower() == ".html":
            selected[key] = path
    return sorted(selected.values())


def rel_link(path: Path, prefix: str = "../") -> str:
    relative = path.relative_to(ROOT).as_posix()
    if path.suffix.lower() == ".html":
        target = f"{prefix}{relative}".replace(" ", "%20").replace("(", "%28").replace(")", "%29")
        return f"- [{path.stem}]({target})"
    no_suffix = relative[:-3] if relative.endswith(".md") else relative
    return f"- [[{prefix}{no_suffix}|{path.stem}]]"


def recent_log_items(log_path: Path, limit: int = 8) -> list[str]:
    if not log_path.exists():
        return []
    entries = [
        f"- {line.removeprefix('## ').strip()}"
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.startswith("## [")
    ]
    return entries[:limit]


def load_pages() -> list[tuple[Path, dict[str, str]]]:
    pages = []
    for page in wiki_pages():
        fields, _ = parse_frontmatter(page.read_text(encoding="utf-8"))
        pages.append((page, fields))
    return pages


def grouped(pages, page_type: str, key: str, vocabulary: list[str]) -> list[str]:
    """按 frontmatter 某个字段分组，输出带三级标题的列表。"""
    lines: list[str] = []
    for value in vocabulary:
        hits = [p for p, f in pages if f.get("type") == page_type and f.get(key) == value]
        if not hits:
            continue
        lines.append(f"### {value}（{len(hits)}）")
        lines.append("")
        lines.extend(rel_link(p) for p in sorted(hits))
        lines.append("")
    return lines


def write_maps(pages) -> list[Path]:
    maps_dir = ROOT / "wiki" / "maps"
    maps_dir.mkdir(exist_ok=True)
    written = []

    for topic in TOPICS:
        target = maps_dir / f"{topic}.md"
        auto: list[str] = [AUTO_BEGIN, ""]
        for page_type, label in (("concept", "概念"), ("entity", "实体"), ("source", "来源")):
            hits = sorted(
                p for p, f in pages if f.get("type") == page_type and f.get("topic") == topic
            )
            if not hits:
                continue
            auto.append(f"## {label}（{len(hits)}）")
            auto.append("")
            auto.extend(f"- [[../{p.parent.name}/{p.stem}|{p.stem}]]" for p in hits)
            auto.append("")

        if target.exists():
            existing = target.read_text(encoding="utf-8")
            head = existing.split(AUTO_BEGIN, 1)[0].rstrip() + "\n\n"
        else:
            head = (
                f"---\ntype: map\ntopic: {topic}\n---\n\n"
                f"# {topic}\n\n## 导读\n\n{AUTO_HINT}\n\n"
            )

        target.write_text(head + "\n".join(auto).rstrip() + "\n", encoding="utf-8")
        written.append(target)
    return written


def main() -> None:
    raw_dir = ROOT / "raw"
    wiki_dir = ROOT / "wiki"
    output_dir = ROOT / "output"

    pages = load_pages()
    concept_count = sum(1 for _, f in pages if f.get("type") == "concept")
    entity_count = sum(1 for _, f in pages if f.get("type") == "entity")
    source_count = sum(1 for _, f in pages if f.get("type") == "source")

    report_files = prefer_html_artifacts(
        list_files(output_dir / "reports", (".md", ".html"))
    )
    export_files = list_files(output_dir / "exports", (".html",))
    slide_files = list_files(output_dir / "slides", (".md",))
    interview_files = [
        p for p in list_files(output_dir / "interview", (".md",)) if p.parent.name == "interview"
    ]
    card_files = [
        p for p in list_files(output_dir / "cards", (".md",)) if p.parent.name == "cards"
    ]
    maps = write_maps(pages)

    lines: list[str] = [
        "# AI Infra 知识库索引",
        "",
        "该页面由 `scripts/update_index.py` 自动生成。分组依据是各页面 frontmatter 的 "
        "`topic` / `entity_type` / `source_kind`。",
        "",
        "## 入口",
        "",
        "- [[../00 Home|Vault Home]]",
        "- [[../AGENTS|AGENTS]]",
        "- [[log|操作日志]]",
        "- [[../inbox/README|Inbox]]",
        "- [[../output/README|Output]]",
        "",
        "## 主题地图",
        "",
    ]
    lines.extend(f"- [[maps/{p.stem}|{p.stem}]]" for p in maps)
    lines.extend(
        [
            "",
            "## 资源统计",
            "",
            f"- 原始文章：{len(list_files(raw_dir / 'articles'))}",
            f"- 原始论文：{len(list_files(raw_dir / 'papers'))}",
            f"- 原始仓库：{len(list_files(raw_dir / 'repos'))}",
            f"- 原始数据集：{len(list_files(raw_dir / 'datasets'))}",
            f"- 原始图片：{len(list_files(raw_dir / 'images'))}",
            f"- 原始代码：{len(list_files(raw_dir / 'code'))}",
            f"- 来源文件：{source_count}",
            f"- 实体文件：{entity_count}",
            f"- 概念文件：{concept_count}",
            f"- 报告文件：{len(report_files)}",
            f"- HTML 导出：{len(export_files)}",
            f"- 面试文件：{len(interview_files)}",
            f"- 卡片文件：{len(card_files)}",
            f"- 幻灯片文件：{len(slide_files)}",
            "",
        ]
    )

    lines.extend(section("概念页面（按主题）", grouped(pages, "concept", "topic", TOPICS)))
    lines.extend(section("实体页面（按类型）", grouped(pages, "entity", "entity_type", ENTITY_TYPES)))
    lines.extend(section("来源摘要（按类型）", grouped(pages, "source", "source_kind", SOURCE_KINDS)))
    lines.extend(section("最近日志", recent_log_items(wiki_dir / "log.md")))
    lines.extend(section("报告", [rel_link(p) for p in report_files]))
    lines.extend(section("HTML 导出", [rel_link(p) for p in export_files]))
    lines.extend(section("面试备考", [rel_link(p) for p in interview_files]))
    lines.extend(section("复习卡片", [rel_link(p) for p in card_files]))
    lines.extend(section("幻灯片", [rel_link(p) for p in slide_files]))

    target = wiki_dir / "index.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated {target}")
    print(f"Updated {len(maps)} maps under wiki/maps/")


if __name__ == "__main__":
    main()
