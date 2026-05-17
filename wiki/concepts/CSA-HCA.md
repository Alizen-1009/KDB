# CSA/HCA

## 定义

`CSA/HCA` 是本文所称 DeepSeek V4 中带 KV 压缩的 attention 结构；在当前知识库语境中，它主要用于讨论压缩 KV、`MQA/KV 共享` 与 [[RoPE]] 的兼容问题。

## 它解决什么问题

- 多个 token 的 KV 状态被压缩为更少的 KV 状态，以降低 cache 与 attention 访问成本
- `MQA` 风格的 KV 共享可以进一步减少缓存，但会让 K/V 共用表示下的位置编码处理变复杂
- 压缩 attention 需要在保留相对位置信息的同时，避免 V 路径被 RoPE 的绝对相位污染

## 核心机制

- 文章以 HCA 为例，指出 RoPE 相关位置包括窗口通道 `SWA` 的 KV、`C128A` 压缩器输出的压缩 KV、上采样后的 Q，以及 attention 输出 O
- 对压缩 KV，推荐在压缩后施加 RoPE：每个压缩块选择一个标定位置，再按该位置计算旋转角
- 文中示例为 `C128A` 每 128 个 KV 状态压缩成 1 个 KV 状态，HCA 对第 `t` 个压缩块采用起始位置 `128 * t`
- 对共享 KV 中被旋转的 V 路径，HCA 通过对输出 O 做逆旋转，减少绝对位置项残留，使输出更接近相对位置形式
- `P` 不直接旋转，因为 attention probability 是标量权重矩阵，不具备 RoPE 所需的 `head_dim` 二维旋转平面

## 关键权衡

- 压缩后旋转需要人为选择块位置标尺，例如起始点、终点或中点；规则必须在训练与推理中保持一致
- 起始位置 `128 * t` 简单且确定，但压缩块内部 token 的细粒度位置会被折叠，真实效果需要结合模型训练和实现验证
- 输出逆旋转保留了 KV 共享的存储优势，但增加了位置编码路径的实现复杂度

## 相关实体

- [[../entities/DeepSeek-AI]]
- [[../entities/DeepSeek V4]]

## 相关来源

- [[../sources/DeepSeekV4中RoPE设计解析]]

## 相关概念

- [[RoPE]]
- [[MLA]]
- [[KV Cache]]

## 研究备注

- 当前页面基于单篇解析文章整理，`CSA/HCA` 与 `C128A` 的正式定义、代码接口和 tensor shape 仍需按公开源码或论文进一步核实。
- 后续可补一张从 `Q/K/V -> 压缩 KV -> RoPE -> attention -> 输出逆旋转` 的流程图，帮助区分 QK score 位置项和 PV/O 路径位置项。
