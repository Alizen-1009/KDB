---
type: source
source_kind: 文章
topic: 投机解码
updated: 2026-08-26
---

# DSpark：结合半自回归生成与置信度调度的投机解码技术

## 来源信息

- 标题：DSpark：结合半自回归生成与置信度调度的投机解码技术
- 作者：军舰
- 日期：2026-06-28
- 类型：文章 / 论文解读
- 原始文件：[[../../raw/articles/DSpark：结合半自回归生成与置信度调度的投机解码技术|DSpark：结合半自回归生成与置信度调度的投机解码技术]]
- 原始链接：[wangjunjian.com](https://wangjunjian.com/posts/2026-06-28-link-dspark/)
- 论文链接：[DeepSpec / DSpark paper](https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf)

## 2-3 条核心摘要

- 文章把 [[../concepts/DSpark|DSpark]] 的问题背景概括为两点：完全并行 drafter 容易出现后缀接受率衰减；固定长度 verification 在高并发时会让 target 把算力浪费在低存活概率后缀上。
- DSpark 在 [[../concepts/DFlash|DFlash]] 式并行 backbone 后增加低成本 Markov / RNN 顺序头，让候选根据本轮已采样前缀进行修正；再通过 conditional confidence、Sequential Temperature Scaling（STS）和 Hardware-Aware Prefix Scheduler 动态选择验证长度。
- 文章转述论文的离线 acceptance 结果与 DeepSeek-V4 预览版线上流量结果，强调 DSpark 同时优化候选质量和 serving batch 级验证预算；所有数字均应保留为来源声称，并绑定模型、流量和 SLA 口径。

## 值得关注的论断

- 来源称，在 Qwen3 4B/8B/14B 与 Gemma4-12B、GSM8K/HumanEval/MT-Bench 等设置上，DSpark 的宏观平均接受长度相对 Eagle3 提升 `26.7%–30.9%`，相对 DFlash 提升 `16.3%–18.4%`。
- 来源称，轻量顺序模块只增加 `0.2%–1.3%` drafting 延迟，却显著改善接受长度；该结果说明少量顺序依赖可能比完全并行且缺少采样前缀条件化更有效。
- 来源称，DSpark 已部署于 DeepSeek-V4 Flash / Pro 预览版线上 serving：相同系统吞吐下，单用户生成速度分别提升 `60%–85%` 与 `57%–78%`，并在严格低延迟 SLA 下扩展 throughput-latency Pareto frontier。该生产结论缺少本文中的完整硬件、流量与 baseline 配置，不能直接外推。

## 关键概念

- [[../concepts/DSpark|DSpark]]
- [[../concepts/DFlash|DFlash]]
- [[../concepts/并行投机解码|并行投机解码]]
- [[../concepts/Speculative Decoding|Speculative Decoding]]
- [[../concepts/Continuous Batching|Continuous Batching]]
- [[../concepts/Benchmarking|Benchmarking]]

## 相关实体

- [[../entities/DeepSeek-AI|DeepSeek-AI]]
- [[../entities/DeepSeek V4|DeepSeek V4]]

## 与现有 wiki 的关系

- 更新 [[../concepts/DSpark|DSpark]]：补充离线接受长度、顺序模块额外延迟与生产 serving 来源声称。
- 更新 [[../concepts/DFlash|DFlash]]：澄清文章所称“各 token 独立预测”是简化表述；DFlash block 内可双向 Attention，但不能根据本轮实际采样前缀重新条件化。
- 更新 [[../concepts/Speculative Decoding|Speculative Decoding]]：补充 confidence-scheduled verification 在生产并发下联合优化接受收益与 target verify shape 的案例。
- 更新 [[../entities/DeepSeek V4|DeepSeek V4]] 与 [[../entities/DeepSeek-AI|DeepSeek-AI]]：记录 DSpark 联合研究及 V4 预览版部署线索。
- 与现有 wiki 无机制冲突；主要需要收窄二手文章对 DFlash “独立预测”的表述。

## 待确认

- 离线 benchmark 的逐模型、逐任务接受长度、draft block size、采样设置与硬件环境需回原论文表格核对。
- `0.2%–1.3%` 额外延迟的分母和测量口径，以及 V4-Flash / V4-Pro 线上提升的 baseline、并发、硬件和流量分布，需用官方生产报告或论文附录核对。
- DeepSpec 开源仓库是否完整包含论文级 STS、Hardware-Aware Prefix Scheduler 与线上异步调度路径，应绑定具体 commit 验证。
