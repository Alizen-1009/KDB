---
title: "llm-wiki"
source: "https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f"
author:
published:
created: 2026-04-13
description: "llm-wiki. GitHub Gist: instantly share code, notes, and snippets."
tags:
  - "clippings"
---
## LLM Wiki

A pattern for building personal knowledge bases using LLMs.  
利用 LLM 构建个人知识库的模式。

This is an idea file, it is designed to be copy pasted to your own LLM Agent (e.g. OpenAI Codex, Claude Code, OpenCode / Pi, or etc.). Its goal is to communicate the high level idea, but your agent will build out the specifics in collaboration with you.  
这是一个概念文件，旨在复制粘贴到您自己的 LLM 代理（例如 OpenAI Codex、Claude Code、OpenCode/Pi 等）中。它的目标是传达高层次的概念，但您的代理将与您协作构建具体细节。

## The core idea 核心思想

Most people's experience with LLMs and documents looks like RAG: you upload a collection of files, the LLM retrieves relevant chunks at query time, and generates an answer. This works, but the LLM is rediscovering knowledge from scratch on every question. There's no accumulation. Ask a subtle question that requires synthesizing five documents, and the LLM has to find and piece together the relevant fragments every time. Nothing is built up. NotebookLM, ChatGPT file uploads, and most RAG systems work this way.  
大多数人使用学习型学习模型（LLM）和文档的体验都类似于随机生成（RAG）：你上传一系列文件，LLM 在查询时检索相关片段，并生成答案。这种方式虽然可行，但 LLM 每次都要从头开始重新发现知识，没有积累。如果你提出一个需要综合五个文档的复杂问题，LLM 每次都必须找到并拼凑相关的片段，没有任何知识积累。NotebookLM、ChatGPT 文件上传以及大多数 RAG 系统都是这样工作的。

The idea here is different. Instead of just retrieving from raw documents at query time, the LLM **incrementally builds and maintains a persistent wiki** — a structured, interlinked collection of markdown files that sits between you and the raw sources. When you add a new source, the LLM doesn't just index it for later retrieval. It reads it, extracts the key information, and integrates it into the existing wiki — updating entity pages, revising topic summaries, noting where new data contradicts old claims, strengthening or challenging the evolving synthesis. The knowledge is compiled once and then *kept current*, not re-derived on every query.  
这里的理念有所不同。LLM 并非仅仅在查询时从原始文档中检索信息， **而是逐步构建并维护一个持久化的维基** ——一个结构化的、相互链接的 Markdown 文件集合，它位于用户和原始数据源之间。当您添加新的数据源时，LLM 不仅会对其进行索引以便后续检索，还会读取它，提取关键信息，并将其整合到现有的维基中——更新实体页面，修订主题摘要，指出新数据与旧说法相矛盾之处，从而强化或挑战不断演进的综合分析。知识只需编译一次即可 *保持最新* ，无需在每次查询时重新推导。

This is the key difference: **the wiki is a persistent, compounding artifact.** The cross-references are already there. The contradictions have already been flagged. The synthesis already reflects everything you've read. The wiki keeps getting richer with every source you add and every question you ask.  
这就是关键区别： **维基百科是一个持续更新、不断完善的资源。** 交叉引用已经存在，矛盾之处也已被标记，综合信息已经反映了你阅读过的所有内容。你添加的每一个来源、提出的每一个问题都会让维基百科的内容更加丰富。

You never (or rarely) write the wiki yourself — the LLM writes and maintains all of it. You're in charge of sourcing, exploration, and asking the right questions. The LLM does all the grunt work — the summarizing, cross-referencing, filing, and bookkeeping that makes a knowledge base actually useful over time. In practice, I have the LLM agent open on one side and Obsidian open on the other. The LLM makes edits based on our conversation, and I browse the results in real time — following links, checking the graph view, reading the updated pages. Obsidian is the IDE; the LLM is the programmer; the wiki is the codebase.  
你几乎从不（或者很少）亲自编写维基——所有内容都由 LLM（知识库管理员）编写和维护。你负责寻找资料、探索和提出正确的问题。LLM 负责所有繁琐的工作——总结、交叉引用、归档和记账，这些工作使知识库能够长期发挥作用。实际上，我一边打开 LLM 代理，一边打开 Obsidian。LLM 根据我们的对话进行编辑，我实时浏览结果——点击链接、查看图表视图、阅读更新后的页面。Obsidian 是集成开发环境（IDE）；LLM 是程序员；维基就是代码库。

This can apply to a lot of different contexts. A few examples:  
这可以应用于许多不同的情境。以下是一些例子：

- **Personal**: tracking your own goals, health, psychology, self-improvement — filing journal entries, articles, podcast notes, and building up a structured picture of yourself over time.  
	**个人** ：跟踪自己的目标、健康、心理、自我提升——记录日记、文章、播客笔记，并随着时间的推移构建一个结构化的自我形象。
- **Research**: going deep on a topic over weeks or months — reading papers, articles, reports, and incrementally building a comprehensive wiki with an evolving thesis.  
	**研究** ：对某个主题进行数周或数月的深入研究——阅读论文、文章、报告，并逐步构建一个包含不断发展的论点的综合维基。
- **Reading a book**: filing each chapter as you go, building out pages for characters, themes, plot threads, and how they connect. By the end you have a rich companion wiki. Think of fan wikis like [Tolkien Gateway](https://tolkiengateway.net/wiki/Main_Page) — thousands of interlinked pages covering characters, places, events, languages, built by a community of volunteers over years. You could build something like that personally as you read, with the LLM doing all the cross-referencing and maintenance.  
	**阅读一本书** ：边读边整理章节，创建页面记录人物、主题、情节线索以及它们之间的联系。最终，你将拥有一个内容丰富的配套维基。想想像 [托尔金门户网站（Tolkien Gateway）这样的粉丝维基——成千上万个相互关联的页面，涵盖人物、地点、事件、语言等等，由志愿者社区历经数年构建而成。你可以在阅读的同时，亲手构建类似的内容，而 LLM（图书馆学硕士](https://tolkiengateway.net/wiki/Main_Page) ）则负责所有的交叉引用和维护工作。
- **Business/team**: an internal wiki maintained by LLMs, fed by Slack threads, meeting transcripts, project documents, customer calls. Possibly with humans in the loop reviewing updates. The wiki stays current because the LLM does the maintenance that no one on the team wants to do.  
	**业务/团队** ：一个由 LLM 维护的内部 wiki，内容来源于 Slack 讨论串、会议记录、项目文档和客户电话。可能也会有人工参与审核更新。wiki 之所以能保持最新，是因为 LLM 负责团队中其他人都不愿意做的维护工作。
- **Competitive analysis, due diligence, trip planning, course notes, hobby deep-dives** — anything where you're accumulating knowledge over time and want it organized rather than scattered.  
	**竞争分析、尽职调查、旅行计划、课程笔记、兴趣爱好深度研究** ——任何需要你随着时间的推移积累知识，并且希望将其整理成册而不是散乱无章的事物。

## Architecture 建筑学

There are three layers:  
它分为三层：

**Raw sources** — your curated collection of source documents. Articles, papers, images, data files. These are immutable — the LLM reads from them but never modifies them. This is your source of truth.  
**原始资源** ——您精心收集的源文档。包括文章、论文、图像和数据文件。这些资源是不可更改的——LLM 系统会读取它们，但绝不会对其进行修改。这就是您的真实来源。

**The wiki** — a directory of LLM-generated markdown files. Summaries, entity pages, concept pages, comparisons, an overview, a synthesis. The LLM owns this layer entirely. It creates pages, updates them when new sources arrive, maintains cross-references, and keeps everything consistent. You read it; the LLM writes it.  
**维基** ——一个由 LLM 生成的 Markdown 文件目录。内容包括摘要、实体页面、概念页面、对比、概述和综合。LLM 完全掌控这一层。它创建页面，在新资源到来时更新页面，维护交叉引用，并确保所有内容的一致性。您阅读它，LLM 编写它。

**The schema** — a document (e.g. CLAUDE.md for Claude Code or AGENTS.md for Codex) that tells the LLM how the wiki is structured, what the conventions are, and what workflows to follow when ingesting sources, answering questions, or maintaining the wiki. This is the key configuration file — it's what makes the LLM a disciplined wiki maintainer rather than a generic chatbot. You and the LLM co-evolve this over time as you figure out what works for your domain.  
**模式** 文件（例如 Claude Code 的 CLAUDE.md 或 Codex 的 AGENTS.md）告诉 LLM wiki 的结构、约定俗成的规则以及在导入资源、回答问题或维护 wiki 时应遵循的工作流程。这是关键的配置文件——它使 LLM 成为一个规范的 wiki 维护者，而不是一个普通的聊天机器人。随着时间的推移，您和 LLM 会共同完善这个模式文件，并不断探索适合您领域的有效方法。

## Operations 运营

**Ingest.** You drop a new source into the raw collection and tell the LLM to process it. An example flow: the LLM reads the source, discusses key takeaways with you, writes a summary page in the wiki, updates the index, updates relevant entity and concept pages across the wiki, and appends an entry to the log. A single source might touch 10-15 wiki pages. Personally I prefer to ingest sources one at a time and stay involved — I read the summaries, check the updates, and guide the LLM on what to emphasize. But you could also batch-ingest many sources at once with less supervision. It's up to you to develop the workflow that fits your style and document it in the schema for future sessions.  
**导入。** 您将新源添加到原始集合中，并指示 LLM 进行处理。流程示例：LLM 读取源，与您讨论关键要点，在 wiki 中编写摘要页面，更新索引，更新 wiki 中相关的实体和概念页面，并将条目添加到日志中。单个源可能会影响 10-15 个 wiki 页面。我个人更喜欢一次导入一个源并全程参与——我会阅读摘要，检查更新，并指导 LLM 重点关注哪些内容。当然，您也可以一次性批量导入多个源，这样可以减少监督。您可以根据自己的习惯开发工作流程，并将其记录在模式中，以便将来使用。

**Query.** You ask questions against the wiki. The LLM searches for relevant pages, reads them, and synthesizes an answer with citations. Answers can take different forms depending on the question — a markdown page, a comparison table, a slide deck (Marp), a chart (matplotlib), a canvas. The important insight: **good answers can be filed back into the wiki as new pages.** A comparison you asked for, an analysis, a connection you discovered — these are valuable and shouldn't disappear into chat history. This way your explorations compound in the knowledge base just like ingested sources do.  
**查询。** 您在维基中提出问题。LLM 会搜索相关页面，读取它们，并综合整理出包含引用的答案。答案可以根据问题的不同而呈现不同的形式——Markdown 页面、对比表格、幻灯片（Marp）、图表（matplotlib）或画布。重要的一点是： **优秀的答案可以作为新页面归档到维基中。** 您提出的对比、分析、发现的关联——这些都很有价值，不应该消失在聊天记录中。这样，您的探索就像已导入的资源一样，不断积累在知识库中。

**Lint.** Periodically, ask the LLM to health-check the wiki. Look for: contradictions between pages, stale claims that newer sources have superseded, orphan pages with no inbound links, important concepts mentioned but lacking their own page, missing cross-references, data gaps that could be filled with a web search. The LLM is good at suggesting new questions to investigate and new sources to look for. This keeps the wiki healthy as it grows.  
定期 **检查。** 请 LLM 定期对 wiki 进行健康检查。检查内容包括：页面之间的矛盾、已被新资料取代的过时说法、没有外部链接的孤立页面、提及但缺少独立页面的重要概念、缺失的交叉引用以及可以通过网络搜索填补的数据空白。LLM 擅长提出新的研究问题和新的参考资料。这有助于 wiki 在发展过程中保持健康。

## Indexing and logging 索引和日志记录

Two special files help the LLM (and you) navigate the wiki as it grows. They serve different purposes:  
随着维基百科的不断扩充，有两个特殊文件可以帮助 LLM（以及您）更好地浏览维基百科。它们各有不同的用途：

**index.md** is content-oriented. It's a catalog of everything in the wiki — each page listed with a link, a one-line summary, and optionally metadata like date or source count. Organized by category (entities, concepts, sources, etc.). The LLM updates it on every ingest. When answering a query, the LLM reads the index first to find relevant pages, then drills into them. This works surprisingly well at moderate scale (~100 sources, ~hundreds of pages) and avoids the need for embedding-based RAG infrastructure.  
**index.md** 是一个面向内容的索引。它是一个维基百科所有内容的目录——每个页面都包含一个链接、一行摘要，以及可选的元数据，例如日期或来源数量。索引按类别（实体、概念、来源等）组织。LLM 会在每次数据摄取时更新它。当响应查询时，LLM 首先读取索引以查找相关页面，然后深入查看这些页面。这种方法在中等规模（约 100 个来源，约数百个页面）下效果出奇地好，并且避免了使用基于嵌入的 RAG 架构。

**log.md** is chronological. It's an append-only record of what happened and when — ingests, queries, lint passes. A useful tip: if each entry starts with a consistent prefix (e.g. `## [2026-04-02] ingest | Article Title`), the log becomes parseable with simple unix tools — `grep "^## \[" log.md | tail -5` gives you the last 5 entries. The log gives you a timeline of the wiki's evolution and helps the LLM understand what's been done recently.  
**log.md** 是按时间顺序排列的日志。它记录了所有事件及其发生的时间，包括内容导入、查询和代码检查等。一个实用技巧：如果每个条目都以一致的前缀开头（例如 `## [2026-04-02] ingest | Article Title` ），则可以使用简单的 Unix 工具解析该日志—— `grep "^## \[" log.md | tail -5` 即可获取最近的 5 个条目。该日志提供了 wiki 的发展历程，并有助于 LLM 了解最近的工作内容。

## Optional: CLI tools 可选：命令行工具

At some point you may want to build small tools that help the LLM operate on the wiki more efficiently. A search engine over the wiki pages is the most obvious one — at small scale the index file is enough, but as the wiki grows you want proper search. [qmd](https://github.com/tobi/qmd) is a good option: it's a local search engine for markdown files with hybrid BM25/vector search and LLM re-ranking, all on-device. It has both a CLI (so the LLM can shell out to it) and an MCP server (so the LLM can use it as a native tool). You could also build something simpler yourself — the LLM can help you vibe-code a naive search script as the need arises.  
在某些时候，您可能需要构建一些小型工具来帮助 LLM 更高效地操作 wiki。最显而易见的就是 wiki 页面搜索引擎——在小规模时，索引文件就足够了，但随着 wiki 的增长，您需要更完善的搜索功能 [。qmd](https://github.com/tobi/qmd) 是一个不错的选择：它是一个本地 Markdown 文件搜索引擎，支持混合 BM25/vector 搜索和 LLM 重新排名，所有操作都在设备端完成。它既有命令行界面 (CLI)（LLM 可以通过 shell 调用它），也有 MCP 服务器（LLM 可以将其作为原生工具使用）。您也可以自己构建一个更简单的工具——LLM 可以在需要时帮助您编写一个简单的搜索脚本。

## Tips and tricks 技巧和窍门

- **Obsidian Web Clipper** is a browser extension that converts web articles to markdown. Very useful for quickly getting sources into your raw collection.
- **Download images locally.** In Obsidian Settings → Files and links, set "Attachment folder path" to a fixed directory (e.g. `raw/assets/`). Then in Settings → Hotkeys, search for "Download" to find "Download attachments for current file" and bind it to a hotkey (e.g. Ctrl+Shift+D). After clipping an article, hit the hotkey and all images get downloaded to local disk. This is optional but useful — it lets the LLM view and reference images directly instead of relying on URLs that may break. Note that LLMs can't natively read markdown with inline images in one pass — the workaround is to have the LLM read the text first, then view some or all of the referenced images separately to gain additional context. It's a bit clunky but works well enough.
- **Obsidian's graph view** is the best way to see the shape of your wiki — what's connected to what, which pages are hubs, which are orphans.
- **Marp** is a markdown-based slide deck format. Obsidian has a plugin for it. Useful for generating presentations directly from wiki content.
- **Dataview** is an Obsidian plugin that runs queries over page frontmatter. If your LLM adds YAML frontmatter to wiki pages (tags, dates, source counts), Dataview can generate dynamic tables and lists.
- The wiki is just a git repo of markdown files. You get version history, branching, and collaboration for free.

## Why this works

The tedious part of maintaining a knowledge base is not the reading or the thinking — it's the bookkeeping. Updating cross-references, keeping summaries current, noting when new data contradicts old claims, maintaining consistency across dozens of pages. Humans abandon wikis because the maintenance burden grows faster than the value. LLMs don't get bored, don't forget to update a cross-reference, and can touch 15 files in one pass. The wiki stays maintained because the cost of maintenance is near zero.

The human's job is to curate sources, direct the analysis, ask good questions, and think about what it all means. The LLM's job is everything else.

The idea is related in spirit to Vannevar Bush's Memex (1945) — a personal, curated knowledge store with associative trails between documents. Bush's vision was closer to this than to what the web became: private, actively curated, with the connections between documents as valuable as the documents themselves. The part he couldn't solve was who does the maintenance. The LLM handles that.

## Note

This document is intentionally abstract. It describes the idea, not a specific implementation. The exact directory structure, the schema conventions, the page formats, the tooling — all of that will depend on your domain, your preferences, and your LLM of choice. Everything mentioned above is optional and modular — pick what's useful, ignore what isn't. For example: your sources might be text-only, so you don't need image handling at all. Your wiki might be small enough that the index file is all you need, no search engine required. You might not care about slide decks and just want markdown pages. You might want a completely different set of output formats. The right way to use this is to share it with your LLM agent and work together to instantiate a version that fits your needs. The document's only job is to communicate the pattern. Your LLM can figure out the rest.