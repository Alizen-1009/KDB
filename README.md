# AI Infra LLM Wiki

这是一个面向 AI infra 程序员的个人知识库，采用 `raw -> wiki -> schema` 的工作流。

- `raw/`：原始资料层，只读不写
- `wiki/`：由 LLM 持续编译和维护的知识层
- `AGENTS.md`：schema 规则层，约束 LLM 如何 ingest / query / export
- `output/`：研究输出、对比分析、幻灯片、可视化、复习卡片
- `inbox/`：待处理资料与待回答问题
- `.pi/skills/`：三个核心动作的执行流程（ingest / query / export）
- `scripts/`：确定性的辅助脚本（重建索引、写日志、卡片转换、健康检查）

## 核心理念

这不是“每次问问题时临时去检索原文”的 RAG 项目，而是先把知识编译成可增长的 Markdown wiki。新增论文、博文、repo、数据集、图片或代码后，LLM 会更新来源摘要页、实体页、概念页、索引和日志，让知识真正累积下来。

## 三层结构

### Raw

保存未经编译的原始资料：

- `raw/articles/`
- `raw/papers/`
- `raw/repos/`
- `raw/datasets/`
- `raw/images/`
- `raw/code/`

### Wiki

保存由 LLM 维护的知识页面：

- `wiki/sources/`
- `wiki/entities/`
- `wiki/concepts/`
- `wiki/index.md`
- `wiki/log.md`

### Obsidian Frontend

Obsidian 是这个知识库的阅读与导航前端。建议把以下页面固定放到书签或首页：

- `00 Home.md`
- `Obsidian Workflow.md`
- `wiki/index.md`
- `wiki/log.md`

### Schema

`AGENTS.md` 规定：

- 页面结构与命名方式
- ingest / query / export 的工作流约束
- 如何处理冲突、缺口、版本差异和回填

## 你要的 Ingest 风格

这套仓库按“先讨论、再落盘”的方式设计：

1. 你在 Pi 中执行 `/skill:kb-ingest raw/...`
2. LLM 先完整阅读原始资料
3. 先和你确认 `2-3` 条摘要以及值得关注的论断
4. 得到确认后，再创建 `wiki/sources/` 页面
5. 再精确更新相关 `wiki/entities/`、`wiki/concepts/`
6. 标注冲突，更新交叉引用、`wiki/index.md` 和 `wiki/log.md`

这个“先汇报、等确认”的人工闸门是整套流程的核心，写在 skill 里强制执行。

## 三个核心动作

三个动作都是 Pi skill（`.pi/skills/`），因为它们需要判断：读什么、更新哪些页面、有没有冲突。确定性的部分才是脚本，由 skill 调用。

### Ingest

```
/skill:kb-ingest raw/papers/flashattention-3.pdf
/skill:kb-ingest                      # 列出 raw/ 里还没进 wiki 的资料
```

### Query

```
/skill:kb-query 对比 vLLM、SGLang 和 TensorRT-LLM 的推理架构权衡
```

产出 `output/reports/` 下的技术报告（面试备考类放 `output/interview/`，幻灯片放 `output/slides/`），高价值结论回填 `wiki/`。

### Export

```
/skill:kb-export attention
```

从概念页和面试整理生成 `output/cards/<主题>.md` 问答卡片，再转成 Anki 可导入的 TSV：

```bash
python3 scripts/export_cards.py output/cards/attention.md
```

### 健康检查

```bash
python3 scripts/lint.py
```

## 推荐工作流

1. 用网页剪藏、手工收集或 git clone 把资料放入 `raw/`
2. 用 `/skill:kb-ingest` 发起一轮摄入
3. 先读 LLM 给出的摘要和重点判断
4. 你确认关注点后，再让 LLM 更新 wiki 页面
5. 通过 `/skill:kb-query` 发起研究问题
6. 把高价值输出回填到 `wiki/`
7. 需要背的主题用 `/skill:kb-export` 导出复习卡片
8. 定期运行 `python3 scripts/lint.py`

## Obsidian 优化

这个 Vault 已经针对 Obsidian 做了三类优化：

- 主页导航：`00 Home.md`
- 协作说明：`Obsidian Workflow.md`
- 模板目录：`Templates/`

后续如果你在 Obsidian 里配合其他 AI 插件使用，AI 更容易直接按当前 Vault 结构产出笔记，而不是随意散落文件。
