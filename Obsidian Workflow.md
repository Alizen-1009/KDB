# Obsidian Workflow

这个 Vault 按 Obsidian 作为前端、AI 作为维护者的方式来使用。

## 推荐工作流

1. 把网页、论文、repo、benchmark、截图放进 `raw/`
2. 在 Obsidian 中打开对应文件，确认命名和内容没问题
3. 运行 `python3 scripts/ingest.py raw/...`
4. 先阅读 AI 给出的 `2-3` 条摘要和重点论断
5. 确认重点后，再让 AI 写入 `wiki/sources/`、`wiki/entities/`、`wiki/concepts/`
6. 用 `python3 scripts/query.py "你的问题"` 发起研究输出
7. 在 Obsidian 中审阅 `output/` 下的结果，并决定是否回填到 `wiki/`

## Obsidian 侧最佳实践

- 优先使用 `[[双向链接]]`，减少裸路径
- 图片优先放在 `raw/images/`，在笔记里用 `![[raw/images/文件名.png]]`
- 不要手工大改 `wiki/` 的结构，除非你明确想重构
- 如果某个页面只是临时草稿，先放到 `inbox/`

## 和 AI 协作的方式

- 让 AI 先总结，再动手改页面
- 让 AI 尽量做局部更新，而不是重写整页
- 让 AI 把新结论写进来源页、实体页、概念页，而不是只停留在聊天里

## 和 Claudian / 类似插件的关系

如果你在 Obsidian 里使用 Claudian 这类插件：

- 可以直接在 Vault 里选中文件后让 AI 总结
- 可以让它按当前 Vault 模板写新笔记
- 可以把查询结果直接输出为新的 Markdown 笔记

关键不是插件本身，而是这个 Vault 已经有固定的目录、模板和 schema，AI 更容易稳定输出。
