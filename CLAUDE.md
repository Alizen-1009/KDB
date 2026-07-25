# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

An **Obsidian vault / AI-infra knowledge base**, not an application. There is no build, no test suite, no dependencies — the deliverable is Markdown. The Python in `scripts/` is a handful of stdlib-only helpers (`python3`, no venv/requirements); **they never generate knowledge content — Claude does that.**

`AGENTS.md` is the authoritative schema for this vault. Read it, plus `wiki/index.md` and `wiki/log.md`, before touching `wiki/`. Content is written in Chinese; keep new pages in Chinese.

## The three core actions are skills, not scripts

`/kb-ingest`, `/kb-query`, `/kb-export` in `.claude/skills/` hold the workflows — they need judgment (what to read, which pages to update, whether a conflict exists). **Use them instead of improvising** when asked to 摄入/收录 material, produce a comparison report, or generate review cards. The split is deliberate: skills decide, scripts do the deterministic parts.

`scripts/` (all paths resolve against the **repo root**, not the shell cwd):

```bash
python3 scripts/kb_meta.py check|sync          # validate frontmatter / refresh derived sources+updated
python3 scripts/update_index.py                # regenerate wiki/index.md + wiki/maps/*.md auto sections
python3 scripts/kb_log.py <action> "<title>" -b "…" -b "…"   # append to wiki/log.md (format-controlled)
python3 scripts/export_cards.py [output/cards/x.md]          # Markdown Q/A cards -> Anki TSV
python3 scripts/lint.py                        # broken links, wrong link paths, frontmatter, orphans
```

After any `wiki/` edit, run `kb_meta.py sync && update_index.py && lint.py` — lint is expected to be all-green.

`scripts/ingest.py` and `scripts/query.py` were deleted (2026-07): they only wrote a prompt template to `inbox/` for Claude to read back — `inbox/` never accumulated a single one in git history. Don't reintroduce that indirection.

## Architecture: raw → wiki → output, with a human gate

**Three content layers** (`AGENTS.md` defines them in full):

- `raw/` — immutable source material (`articles/ papers/ repos/ datasets/ images/ code/`). Never edit or "fix" it; record contradictions in `wiki/`, not here.
- `wiki/` — the compiled knowledge layer Claude maintains: `sources/` (one page per raw item), `entities/` (projects, orgs, people, hardware), `concepts/` (mechanisms, algorithms, terms), `maps/` (one page per topic), plus generated `index.md` and append-only `log.md`.
- `output/` — research artifacts: `reports/` (technical), `interview/` (interview prep, split out 2026-07-25 — 11 of the original 14 reports), `slides/` (Marp), `visuals/`, `cards/` (review cards + `cards/anki/`), `code/`. High-value conclusions get backfilled into `wiki/`.

**The two-phase gate is the core operating rule.** On ingest: read the full source, report `2-3` core summaries + `1-3` notable claims to the user, **wait for confirmation**, and only then write `wiki/sources/`, update related entity/concept pages, flag conflicts explicitly, update cross-references, `wiki/index.md`, and append to `wiki/log.md`. Finish by reporting what was created vs. updated and any unresolved points.

**Query flow:** start from `wiki/index.md` and the relevant wiki pages; only fall back to `raw/` when the wiki is insufficient. Separate 已知事实 / 实现差异 / 性能权衡 / 待核实 in the output.

## Conventions that are easy to get wrong

- **Every page in `wiki/{concepts,entities,sources}` carries frontmatter** (`type`, `topic`, plus `entity_type`/`source_kind`; `sources` and `updated` are derived). Topic vocabulary is fixed — it lives in `scripts/kb_meta.py` and is mirrored for humans in `AGENTS.md`. The index and topic maps group by these fields, so a page without valid frontmatter falls out of navigation entirely.
- **`wiki/index.md` is generated** by `update_index.py` — never hand-edit it. Same for everything below the `BEGIN AUTO` marker in `wiki/maps/<topic>.md`; **the 导读 section above that marker is hand-written and preserved** across regeneration — that's where reading order lives. `wiki/log.md` is append-only via `scripts/kb_log.py` — don't hand-edit it either, the entry format is script-controlled.
- **Wikilink targets are filename stems.** Cross-directory links inside `wiki/` use relative form (`[[../entities/vLLM]]`, `[[../sources/xxx]]`); same-directory links are bare (`[[MLA]]`). Renaming a page silently breaks every inbound link — grep before renaming.
- **Count the `../` carefully.** `wiki/sources/` → raw is `../../raw/...`; `output/reports|interview/` → wiki is `../../wiki/...`. 39 links had one `../` too few (fixed 2026-07-25); `lint.py` now catches this, since the name resolves even when the path doesn't.
- **Source page naming differs by kind**: articles and code mirror the raw file's stem (`raw/articles/PageAttention代码走读.md` → `wiki/sources/PageAttention代码走读.md`); papers use the paper title, not the arXiv id (`raw/papers/2603.15031v1.pdf` → `wiki/sources/Attention Residuals.md`). Source pages must contain an `原始文件：` field — `lint.py` checks for it.
- **`_`-prefixed files are infrastructure**: `wiki/*/_TEMPLATE.md` are the canonical page templates (excluded from the index and lint). Duplicate copies under `Templates/` were deleted 2026-07-25; only `Templates/Query Report Template.md` remains there, since `output/` has no `_TEMPLATE`.
- **Images** live under `raw/images/<主题>/` and are embedded as `![[raw/images/...]]`. Clipped articles keep remote image URLs (zhihu CDN) rather than local copies.
- **Web-clipped articles live in `raw/articles/`** with YAML frontmatter (`source`, `author`, `tags: clippings`). A separate root-level `Clippings/` staging dir existed until 2026-07-25 and was folded into `raw/articles/`; if it reappears (the Obsidian Web Clipper default target), fold it back rather than maintaining two intake paths.
- **`lint.py` output is expected to be clean** (as of 2026-07-25 it reports zero broken links). It scans every `.md` in the repo but ignores `[[...]]` inside YAML frontmatter, code fences, and inline code — so clipper `author:` fields and doc examples don't register. If it reports a broken link, it's real; fix it rather than dismissing it as noise.
- **Report → wiki links use `[[../../wiki/concepts/X|X]]`** (`output/reports/` and `output/interview/` are both two levels deep; a few legacy `../wiki/...` links are broken).

## Writing standards (from AGENTS.md)

- Structured Markdown over prose; prefer bullets and the template section headings.
- Distinguish 原文说法 / 实现细节 / 实验观察 / LLM 归纳. Mark uncertainty as 待核实.
- Never invent benchmark numbers, throughput figures, hardware configs, or version details.
- **Edit minimally**: read the target page first, add or amend the specific section, do not rewrite whole pages. One new source may legitimately touch several pages — keep each diff small.
