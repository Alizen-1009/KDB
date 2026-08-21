#!/usr/bin/env python3
"""把 `output/cards/*.md` 里的问答卡片转换成 Anki 可导入的 TSV。

卡片源文件由 `kb-cards` skill 编写，是 vault 内可直接阅读、可回链的 Markdown；
本脚本只做机械转换，不生成内容。

源文件格式：

    ---
    type: cards
    deck: AI Infra::Attention
    tags: attention flashattention
    source: [[FlashAttention]]
    ---

    ### FlashAttention 为什么能降低显存占用？

    - 不物化 N×N 的注意力矩阵，改为分块计算并用 online softmax 递推。

每个 `### ` 标题是卡片正面，标题到下一个 `### ` 之间的内容是背面。

用法：

    python3 scripts/export_cards.py                      # 转换 output/cards/ 下全部卡片
    python3 scripts/export_cards.py output/cards/attention.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from kb_utils import OUTPUT, ROOT, visible_files

CARDS = OUTPUT / "cards"
ANKI = CARDS / "anki"

FRONT_RE = re.compile(r"^###\s+(.*?)\s*$")
WIKILINK_ALIAS_RE = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    body = text[end + 4 :].lstrip("\n")
    return meta, body


def parse_cards(body: str) -> list[tuple[str, str]]:
    cards: list[tuple[str, str]] = []
    front: str | None = None
    back: list[str] = []

    for line in body.splitlines():
        match = FRONT_RE.match(line)
        if match:
            if front is not None:
                cards.append((front, "\n".join(back).strip()))
            front = match.group(1)
            back = []
        elif front is not None:
            back.append(line)

    if front is not None:
        cards.append((front, "\n".join(back).strip()))
    return cards


def to_anki_field(text: str) -> str:
    text = WIKILINK_ALIAS_RE.sub(r"\2", text)
    text = WIKILINK_RE.sub(lambda m: m.group(1).split("/")[-1], text)
    text = text.replace("\t", "    ").strip()
    text = text.replace("\n", "<br>")
    return re.sub(r"(<br>){3,}", "<br><br>", text)


def convert(path: Path) -> tuple[Path, int, list[str]]:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    cards = parse_cards(body)
    warnings: list[str] = []

    seen: set[str] = set()
    rows: list[str] = []
    for front, back in cards:
        if not back:
            warnings.append(f"{path.name}: 卡片《{front}》没有背面内容，已跳过")
            continue
        if front in seen:
            warnings.append(f"{path.name}: 卡片正面重复《{front}》，已跳过")
            continue
        seen.add(front)
        tags = " ".join(meta.get("tags", "").split())
        rows.append(f"{to_anki_field(front)}\t{to_anki_field(back)}\t{tags}")

    header = [
        "#separator:tab",
        "#html:true",
        "#notetype:Basic",
        f"#deck:{meta.get('deck', 'AI Infra')}",
        "#tags column:3",
    ]

    ANKI.mkdir(parents=True, exist_ok=True)
    target = ANKI / f"{path.stem}.txt"
    target.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
    return target, len(rows), warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert markdown Q/A cards to Anki TSV.")
    parser.add_argument("sources", nargs="*", help="卡片文件路径，默认转换 output/cards/ 下全部")
    args = parser.parse_args()

    if args.sources:
        paths = [
            (ROOT / s).resolve() if not Path(s).is_absolute() else Path(s).resolve()
            for s in args.sources
        ]
    else:
        paths = [p for p in visible_files(CARDS, (".md",)) if p.parent == CARDS]

    if not paths:
        raise SystemExit(f"No card files found under {CARDS.relative_to(ROOT)}/")

    total = 0
    all_warnings: list[str] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Card file not found: {path}")
        target, count, warnings = convert(path)
        all_warnings.extend(warnings)
        total += count
        print(f"{path.relative_to(ROOT)} -> {target.relative_to(ROOT)} ({count} cards)")

    for warning in all_warnings:
        print(f"warning: {warning}")
    print(f"Total: {total} cards")


if __name__ == "__main__":
    main()
