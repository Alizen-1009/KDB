---
type: concept
topic: 位置编码
sources: 6
updated: 2026-05-17
---

# RoPE

## 定义

`RoPE`（Rotary Position Embedding）是一种把位置信息注入到 attention 的 `Q/K` 向量中的方法。它通过对向量的偶数维-奇数维成对施加位置相关旋转，使 attention score 自然携带相对位置信息。

## 它解决什么问题

- 避免只依赖绝对位置向量相加时，对相对位置信息建模不够直接的问题
- 在不引入额外大规模位置参数表的前提下，把位置信息写进 `Q/K` 的几何关系
- 为长上下文模型提供一种在工程上高效、在机制上更适合外推的位置编码路径

## 核心机制

- 对 `Q/K` 的维度按两两一组配对，每组视为一个二维平面
- 对于位置 `m`，在每个二维平面上施加角度与 `m * theta_i` 相关的旋转
- 不同维度组使用不同频率 `theta_i`，低频负责更长尺度，高频负责更细局部变化
- 由于 `Q_m` 与 `K_n` 各自旋转后做内积，attention score 中会自然出现与 `m - n` 相关的项
- 工程实现通常不会显式构造旋转矩阵，而是通过 `cos/sin` 广播或复数乘法高效完成

## 实现细节

- 多维 RoPE 可理解为块对角旋转矩阵：每个 `2D` 子空间独立旋转，整体仍是正交变换
- 数学说明常把相邻维度配成 `(x_0,x_1)、(x_2,x_3)`；常见实现也会把前半维和后半维配成 `(x_i, x_{i+d/2})`
- `rotate_half(x)` 的作用是构造与 `sin` 相乘的交换/取负分量，使 `(x * cos) + (rotate_half(x) * sin)` 等价于二维旋转
- `cos/sin` 通常按最大位置和旋转维度预计算或缓存，前向时按 `position_ids` lookup/broadcast 到 `Q/K`
- 具体维度配对方式会影响代码张量布局，但只要配对和 `cos/sin` 拼接一致，数学上仍对应一组二维平面旋转

## Qwen3.8-Flash-Next 的位置编码边界

- Qwen3.8 的 GDN hybrid 在周期性 full-attention 层保留 RoPE。NoPE 变体在预训练阶段与 RoPE 近似，但 post-training 后 endless generation 比例明显更高，因此论文没有采用 NoPE；这是一项晚阶段生成质量观察，不代表 RoPE 单独决定终止行为。
- [[Qwen Sparse Attention|QSA]] 的 indexer head dimension 为 `128`，其中 `64` 维使用 partial RoPE。key 先按 `r=4` 平均池化，再把压缩块赋为 block starting position；query 保留 token position。先 pool 后旋转可避免平均不同 rotary phases。

## DeepSeek Sparse Attention 的 Partial RoPE

[[../entities/DeepSeek-V3.2-Exp]] 的 [[DeepSeek Sparse Attention|lightning indexer]] 对 indexer query/key 部分维度应用 partial RoPE；核心 MLA attention 继续使用 decoupled RoPE 路径。原始技术报告没有披露 indexer head 数、维度或 rotary dimension，不能套用 QSA 的 `4 heads / 128 dim / 64 rotary dim` 配置。

## GLM-5 系列的位置配置演进

- [[../entities/GLM-5 系列|GLM-5 / 5.1]] 的公开 Base 配置上限约 `200K`（GLM-5 为 `max_position_embeddings=202752`），MLA Q/K head 拆为 `192` 维 NoPE 与 `64` 维 RoPE。
- GLM-5.2 将 `max_position_embeddings` 提升到 `1,048,576`，并设置 `rope_theta=8,000,000`；[[IndexShare]] 降低 DSA Indexer 在超长上下文下的重复成本，但它不是位置编码机制。
- GLM-5.3 官方说明沿用 GLM-5.2 Base；由于没有独立公开文本 checkpoint config，不能写成直接核对了 5.3 的 RoPE 字段。
- [[../entities/GLM-5.3-Flash]] 同样配置 `1M` context，但其 DSA/MLA 设置 `qk_nope_head_dim=256`、`qk_rope_head_dim=0`、`mla_use_nope=true`，即该路径不保留 RoPE 子空间。不能仅由“1M context”反推其采用与 5.2 相同的 RoPE 方案。

## 关键权衡

- 相比加性绝对位置编码，RoPE 更直接服务于相对位置建模，也更适合很多 decoder-only 模型
- 它没有额外的大位置参数表，但长上下文下仍然会面临频率设计和外推稳定性问题
- 实际长上下文能力不仅取决于“有没有 RoPE”，还取决于 `rope_theta`、scaling 方式，以及是否采用 `partial_rotary_factor`
- 视觉和视频场景可扩展为二维或三维 RoPE，但需要处理轴向位置、模态区分、样本边界和频率分配
- 在压缩 attention 或 `MQA/KV 共享` 中，RoPE 不能无脑作用到共享 KV 表示；否则可能污染 V 路径，或在压缩前聚合时混合多个 token 的位置相位

## 相关实体

- [[../entities/DeepSeek-AI]]
- [[../entities/DeepSeek V4]]
- [[../entities/Gemma 4]]
- [[../entities/Qwen VL]]
- [[../entities/Qwen3.8-Flash-Next]]
- [[../entities/DeepSeek-V3.2-Exp]]
- [[../entities/GLM-5 系列]]
- [[../entities/GLM-5.3-Flash]]

## 相关来源

- [[../sources/十分钟读懂旋转编码（RoPE）]]
- [[../sources/彻底搞懂RoPE计算原理：从1D到3D]]
- [[../sources/DeepSeekV4中RoPE设计解析]]
- [[../sources/On the Design of Qwen3.8-Next Architecture：Evaluation, Efficiency, and Training Stability]]
- [[../sources/DeepSeek-V3.2-Exp：Boosting Long-Context Efficiency with DeepSeek Sparse Attention]]
- [[../sources/glm-5-architecture-evolution]]

## 相关概念

- [[CSA-HCA|CSA/HCA]]
- [[Dual RoPE]]
- [[M-RoPE]]
- [[混合注意力]]
- [[Qwen Sparse Attention]]
- [[DeepSeek Sparse Attention]]
- [[IndexShare]]

## 研究备注

- 当前页面覆盖基础 RoPE 机制、常见实现配对和多维扩展入口；后续可继续拆出 `RoPE scaling`、`YaRN`、`NTK-aware RoPE`、`LongRoPE` 等长上下文变体
- 对“远程衰减”和“长度外推”的表述应保持谨慎：它们是 RoPE 的重要直觉和经验现象，但真实长上下文效果还依赖训练长度、频率缩放和 attention 实现
- DeepSeek V4 解析补充了一个压缩 attention 语境：压缩 KV 更倾向于压缩后按块标定位置再旋转；若 V 路径被旋转，则可能需要输出逆旋转来消除绝对位置项
