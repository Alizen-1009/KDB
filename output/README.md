# Output

这里存放由 LLM 生成的成果文件。

## 子目录说明

- `reports/`：架构分析、技术综述、benchmark 对比、研究备忘；长篇人读报告默认是自包含 HTML，短线性或 Obsidian-native 备忘可用 Markdown
- `exports/`：从 wiki 或既有 output 编译出的可分享、自包含 HTML；由 `kb-export` 生成
- `interview/`：面试备考——题目拆解、专题稿、复盘；`interview/code/` 是配套的手撕代码
- `slides/`：Marp 幻灯片
- `visuals/`：图表、流程图、可视化图片
- `cards/`：复习卡片（Markdown 问答，可读可回链）；由 `kb-cards` 生成，`cards/anki/` 是转换出的 Anki 导入文件
- `code/`：实验脚本与代码产物

`reports/` 和 `interview/` 分开的原因：技术报告的组织方式是「机制 → 权衡 → 工程含义」，面试备考是「题目 → 答案 → 关联概念页」，两者的复习动线不同。

高价值输出建议在整理后回填到 `wiki/`，避免成果只停留在一次性文件里。HTML 报告与 wiki 之间使用普通相对链接；不要为同一报告同时维护 HTML 和 Markdown 两份可编辑源。
