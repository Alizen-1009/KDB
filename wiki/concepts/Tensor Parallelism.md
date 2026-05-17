# Tensor Parallelism

## 定义

在单层内部按张量维度切分权重和计算，让多个 GPU 协同完成同一层前向与反向计算的并行方式。

## 它解决什么问题

- 降低超大模型在单卡上放不下或算不动的问题
- 在多 GPU 环境中压缩单请求时延
- 在训练场景中沿层宽方向拆分大矩阵计算，提升模型可扩展性

## 核心机制

- 把单层权重与矩阵乘法切分到多个 GPU
- 每个 GPU 计算部分结果
- 通过 all-reduce 或类似同步机制在层间汇总
- 通常优先用于节点内高速互联环境，因为每层都会发生较重通信
- 在 Lecture 8 的最小实现里，前向先做局部 matmul，再通过 `all_gather` 拼回完整激活

## Forward 通信量粗估

- 在常见 Megatron-style TP 中，`QKV / MLP up` 这类 column-parallel linear 通常不立刻通信；`attention output projection / MLP down` 这类 row-parallel linear 之后需要把各 rank 的 partial output 求和。
- 因此一个 Transformer layer 的推理 forward 通常有约 `2` 次 activation all-reduce：一次在 attention 输出后，一次在 MLP 输出后。
- 设本步 token 数为 `M`，hidden size 为 `H`，dtype 字节数为 `b`，TP size 为 `P`，单次 all-reduce 的逻辑张量大小为 `N = M * H * b` bytes。
- Ring all-reduce 下，每个 rank 的单次发送通信量可粗略估为 `2 * (P - 1) / P * N`；如果把 send 与 receive 都计入，则再乘 `2`。
- 所以每层 forward 的每 rank 发送通信量约为 `4 * (P - 1) / P * M * H * b` bytes；整模型再乘层数 `L`。
- Decode 场景里 `M` 通常是当前 step 的 active sequence 数；Prefill 场景里 `M` 是本轮 prompt tokens 数，因此 prefill 单次通信量更大，但 decode 的通信更高频、更容易受 all-reduce latency 影响。

## 关键权衡

- 能有效降低单请求时延
- 高度依赖节点内高速互联，跨节点通信成本会迅速上升
- 相比流水线并行没有 bubble，但通信更频繁、对带宽要求也更高
- 对 [[MLA]] / DeepSeek 类模型的 attention，普通 TP 可能导致 latent `KV Cache` 在多卡间重复保存；工程上会用 [[DP Attention]] 这类策略重新组织 attention 侧并行。

## 相关实体

- [[../entities/TensorRT-LLM]]
- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/斯坦福CS336 Lecture 7 - Parallelism basics]]
- [[../sources/斯坦福CS336 Lecture 8 - Distributed communication and training code]]

## 相关概念

- [[PD分离]]
- [[KV Cache]]
- [[Torch Distributed]]
- [[Sequence Parallelism]]
- [[流水线并行]]
- [[DP Attention]]

## 研究备注

- 后续可补 `Expert Parallelism` 和训练/推理下 Tensor Parallel 的不同瓶颈
- 面试里需要说明“通信量”和“通信时延”不同：decode 的每次 all-reduce payload 可能不大，但每层两次同步、层数多、step 高频，因此跨节点 TP 往往被 latency 和同步拖住。
