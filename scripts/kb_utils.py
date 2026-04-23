#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
WIKI = ROOT / "wiki"
OUTPUT = ROOT / "output"
INBOX = ROOT / "inbox"

WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def now_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def slugify(value: str) -> str:
    value = value.strip().replace("/", "-").replace(" ", "-")
    value = re.sub(r"[^\w\-\u4e00-\u9fff]+", "-", value, flags=re.UNICODE)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or now_stamp()


def visible_files(base: Path, exts: tuple[str, ...] | None = None) -> list[Path]:
    if not base.exists():
        return []
    files = [p for p in base.rglob("*") if p.is_file() and not p.name.startswith(".")]
    if exts:
        files = [p for p in files if p.suffix.lower() in exts]
    return sorted(files)


def append_log_entry(action: str, title: str, bullets: list[str]) -> None:
    log_path = WIKI / "log.md"
    if not log_path.exists():
        log_path.write_text("# 知识库操作日志\n", encoding="utf-8")

    lines = ["", f"## [{now_date()}] {action} | {title}", ""]
    for bullet in bullets:
        lines.append(f"- {bullet}")
    lines.append("")

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def wikilinks_in(text: str) -> list[str]:
    return WIKILINK_RE.findall(text)
