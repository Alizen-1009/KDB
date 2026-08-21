---
name: kb-export
description: 把 wiki 页面、研究结论或既有 output 内容导出为可直接在浏览器打开和分享的自包含 HTML，写入 output/exports/。当用户说 export / 导出 / 导出 HTML / 做成网页 / 生成可分享版本时使用。Anki 复习卡使用 kb-cards，不走本 skill。
---

# Export：把知识导出成自包含 HTML

## 目标

把知识库中的已有知识编译成一个面向人的浏览器 artifact，而不是复制 Markdown 文本。默认产物是：

```text
output/exports/<描述性中文标题>.html
```

导出是 wiki/output 的派生物，不是新的事实来源。不要为同一导出同时维护 HTML 和 Markdown 两份可编辑源。

## 第 1 步：确定读者任务与选料范围

1. 明确读者要理解、比较、评审、分享还是操作什么；缺少会改变页面结构的关键信息时，只问一个聚焦问题。
2. 先读 `wiki/index.md` 和相关 `wiki/maps/<topic>.md`，再读需要导出的概念页、实体页、来源页或既有 `output/` 内容。
3. 默认只编译知识库中已经存在的结论，不重新研究。若发现关键结论缺来源、互相冲突或标有“待核实”，在 HTML 中保留这些边界，不静默补成确定事实。

完成标准：每个 substantive section 都能指向已读取的 wiki/output 材料，页面只有一个主要读者目标。

## 第 2 步：应用 HTML format gate

读取并遵循全局 `html-artifacts` skill。它是信息架构、自包含实现、可访问性、交互闭环、信任边界和浏览器验证的单一来源，本 skill 不复制其规则。

本仓库补充约束：

- 正文默认中文，术语首次出现保留英文原词。
- HTML 内引用 wiki/output 使用普通相对链接，例如 `../../wiki/concepts/MoE.md`。
- 重要事实区分来源事实、实现差异、性能权衡和待核实项。
- 需要图解时优先使用 inline SVG 或语义化 HTML/CSS；不依赖远程字体、脚本或 CDN。
- 若用户明确要求 `.md`、`.pdf`、`.pptx`、Anki 或其它格式，使用对应 workflow，不强包成 HTML。

## 第 3 步：生成一个 portable artifact

文件名使用描述性中文标题，不加时间戳。HTML 至少应包含：

- 准确的 `<title>`、`lang="zh-CN"`、charset 和 viewport；
- 清晰的结论摘要与内容导航；
- 可追溯的来源/知识库路径；
- 响应式布局、打印样式和可见键盘焦点；
- 仅在有助于比较、筛选或理解时加入 JavaScript 交互。

关键结论和警告必须无需点击即可看到。导出已有报告时，重新组织成适合浏览器阅读的信息结构，不把 Markdown 包进 `<pre>`。

## 第 4 步：验证

按 `html-artifacts` 的检查清单在真实浏览器中验证：

1. 桌面与窄屏布局；
2. 每个交互、键盘操作和导出动作；
3. console 无 error/warning；
4. heading hierarchy、focus、contrast、reduced motion；
5. 表格、代码、图和 print 输出可读；
6. 相对链接有效，事实边界与来源没有丢失。

浏览器工具不可用时，执行 HTML parser、链接、JavaScript 语法和结构审计，并在最终汇报中说明限制。

## 第 5 步：索引与记录

```bash
python3 scripts/update_index.py
python3 scripts/lint.py
python3 scripts/kb_log.py export "<导出标题>" \
  -b "选料页面：\`wiki/concepts/xxx.md\`、\`wiki/entities/yyy.md\`" \
  -b "创建 HTML：\`output/exports/xxx.html\`" \
  -b "验证：桌面/窄屏、交互、console、链接与打印" \
  -b "待核实：<保留在导出中的不确定项；没有则写无>"
```

导出本身不自动回填 wiki；如果在导出过程中形成了新的稳定结论，切换到 `kb-distill` 处理，不要让派生 artifact 反过来成为事实来源。

## 最终汇报

先给出 `.html` 路径，再用一句话说明读者用途、选料范围和任何验证限制。不要在聊天中重复粘贴整份导出内容。
