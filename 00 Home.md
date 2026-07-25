# AI Infra Vault Home

这是这个 Obsidian Vault 的主入口。

## 现在从这里开始

- 总说明：[[README]]
- AI 维护规则：[[AGENTS]]
- 知识库索引：[[wiki/index]]
- 操作日志：[[wiki/log]]

## 主题地图

每张地图顶部是手写导读（这个主题按什么顺序读），下半是自动维护的清单。

- [[wiki/maps/注意力机制|注意力机制]] ｜ [[wiki/maps/KV Cache|KV Cache]] ｜ [[wiki/maps/推理服务|推理服务]]
- [[wiki/maps/并行与分布式|并行与分布式]] ｜ [[wiki/maps/GPU 编程|GPU 编程]] ｜ [[wiki/maps/性能分析|性能分析]]
- [[wiki/maps/模型架构|模型架构]] ｜ [[wiki/maps/投机解码|投机解码]] ｜ [[wiki/maps/训练与 Scaling|训练与 Scaling]] ｜ [[wiki/maps/位置编码|位置编码]]

## 日常工作流

- 摄入新资料：[[Obsidian Workflow]]
- 待处理任务：[[inbox/README]]
- 研究输出：[[output/README]]
- 原始资料区：[[raw/README]]

## 三个核心动作

在 Pi 里执行，流程写在 `.pi/skills/`：

- `/skill:kb-ingest raw/...`：把原始资料编译进 wiki
- `/skill:kb-query <问题>`：产出研究报告，回填 wiki
- `/skill:kb-export <主题>`：导出复习卡片到 `output/cards/`

## 页面模板

- 来源摘要模板：[[wiki/sources/_TEMPLATE|来源页模板]]
- 实体页模板：[[wiki/entities/_TEMPLATE|实体页模板]]
- 概念页模板：[[wiki/concepts/_TEMPLATE|概念页模板]]
- 查询输出模板：[[Templates/Query Report Template]]

## 使用建议

- 你主要在 Obsidian 里阅读、浏览、回顾和提问题
- 我主要负责写入、维护、补链、更新索引和整理结构
- 高价值问答结果尽量回填到 `wiki/`
