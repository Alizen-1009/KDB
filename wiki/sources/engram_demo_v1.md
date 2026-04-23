# engram_demo_v1

## 来源信息

- 标题：engram_demo_v1
- 作者：DeepSeek-AI（官方 demo 实现）
- 日期：2026-01（与论文配套发布）
- 类型：代码 / demo implementation
- 原始文件：`raw/code/engram_demo_v1.py`

## 2-3 条核心摘要

- 这份 demo 明确展示了论文里 `Engram` 的完整数据流：`tokenizer compression -> n-gram hashing -> multi-head embedding lookup -> context-aware gating -> short convolution -> residual injection`。
- 代码很重要的一点是把“确定性地址”落实成了可执行实现：`CompressedTokenizer` 先做词表压缩，`NgramHashMapping` 再按层构造固定 hash 与质数表尺寸，这让训练分片和推理预取都具备可操作性。
- 这份实现刻意把 `Attention / MoE / Hyper-connection` 简化掉，只保留 Engram 主体，因此非常适合用来理解模块边界，但不能直接代表生产级训练或推理实现。

## 值得关注的论断

- 代码层面最能体现论文特色的不是“大 embedding 表”，而是“静态检索 + 动态门控”的组合：检索键只依赖输入 token，真正的上下文选择发生在 gate 上。
- `CompressedTokenizer` 说明论文里的 tokenizer compression 不是口头概念，而是通过规范化规则把多个原始 token ID 压到同一个 canonical ID 上，提升 N-gram 记忆密度。
- demo 使用 depthwise `ShortConv` 扩展门控后记忆的感受野，也印证了论文把 Engram 视为“查表后再做轻量局部建模”的模块，而不是单纯 embedding 拼接。

## 关键概念

- [[Conditional Memory]]
- [[Sparsity Allocation]]

## 相关实体

- [[../entities/Engram]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`Conditional Memory`
- 会更新哪些实体页：`Engram`
- 是否存在冲突：与现有 wiki 无直接冲突，但会把原先偏论文描述的页面推进到代码级实现视角

## 待确认

- 这份代码是 demo，不包含真实多机训练、host-memory prefetch、custom kernel 和完整 hyper-connection 细节；生产级实现仍需结合官方仓库和后续工程材料
