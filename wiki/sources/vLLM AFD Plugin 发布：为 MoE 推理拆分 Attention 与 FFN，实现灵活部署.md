---
type: source
source_kind: 文章
topic: 推理服务
updated: 2026-07-25
---

# vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署

## 来源信息

- 标题：vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署
- 作者：AFD Plugin Team
- 日期：2026-07-24
- 类型：文章
- 原始文件：[[../../raw/articles/vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署.md]]
- 原始链接：https://mp.weixin.qq.com/s/UpbZf12Ap-mNJfKe2GsLLw

## 2-3 条核心摘要

- vLLM AFD Plugin 将 MoE Transformer 每个切分层中的 Attention 与 FFN/MoE 放到两类独立服务：Attention 侧保留请求调度、KV Cache、批处理和采样，FFN 侧以连接器驱动的后台循环接收中间激活与执行元数据、计算专家输出并返回。
- AFD 不是把所有 Attention 层和所有 FFN 层重排为两个连续模型阶段，而是在层内保持 `Attention[l] -> FFN[l] -> Attention[l+1]` 的数据依赖。同步路径需要每个切分层进行 `A -> F` hidden states 发送和 `F -> A` FFN 输出返回；异步路径通过不同 ubatch 之间的流水重叠通信与计算。
- 插件以小型连接器契约适配 NVIDIA GPU 与昇腾 NPU，覆盖同步 decode、异步 prefill、decode-only graph 和特定 dual-batch 路径，并允许 Attention 与 FFN 使用不同 rank 拓扑、独立扩缩容。

## 值得关注的论断

- AFD 具有两级算子流水的执行特征，但不同于按连续层切分的传统 Pipeline Parallelism：它按算子角色切分，激活在 A/F 服务之间逐层往返；传统 PP 按模型深度切分，激活通常沿 stage 单向流动。
- AFD 可以与其他并行维度组合：FFN 侧可采用 [[Expert Parallelism]] 进行 token dispatch 与专家计算，两侧可各自采用适配的 TP/DP 拓扑，外层还可与 [[PD分离]] 组合。是否支持任意组合仍取决于具体连接器、recipe 和版本验证范围。
- 受控 decode 实验中，48A16F 的归一化吞吐低于 EP64，而 64A16F 在 16K、32K 输入下分别高出 11.3% 和 9.0%，说明分离本身不保证提速，Attention/FFN rank 配比是关键变量。
- 异步 prefill 实验在裁剪到 10 层的 DeepSeek V3.2 W8A8、两个昇腾 910C 节点和强制专家均衡条件下，将 12 req/s 时的 P50 TTFT 从 15.1 秒降至 8.0 秒；该结果不能外推为完整模型或一般生产负载的通用结论。

## 关键概念

- [[../concepts/Attention-FFN 分离]]
- [[../concepts/MoE]]
- [[../concepts/Expert Parallelism]]
- [[../concepts/Tensor Parallelism]]
- [[../concepts/流水线并行]]
- [[../concepts/PD分离]]

## 相关实体

- [[../entities/vLLM AFD Plugin]]
- [[../entities/vLLM]]
- [[../entities/NCCL]]

## 与现有 wiki 的关系

- 新增 [[../concepts/Attention-FFN 分离]]，用于解释逐层 hidden-state 往返、与传统 PP 的区别，以及和 EP、TP、DP、PD 的组合关系。
- 补充 [[../concepts/MoE]]、[[../concepts/Expert Parallelism]] 与 [[../concepts/PD分离]] 的 serving 拓扑视角，并更新 [[../entities/vLLM]]。
- 未发现与现有 wiki 的直接冲突；AFD 与 PD 分离是不同切分轴，可以组合而非相互替代。

## 待确认

- 当前文章是项目发布说明，且明确将项目标为实验阶段；任意 A/F rank 比、跨后端拓扑、完整模型精度与生产稳定性仍需根据仓库版本和可复现实验核实。
- 当前两类角色都加载完整权重，尚未实现理想化的按角色只驻留 Attention 或 FFN 权重；后续版本是否改变这一点需要继续跟踪。
- 文章给出的 benchmark 使用模拟逻辑规模、强制均衡路由或裁剪模型，应保留实验边界，不作为一般化容量规划数字。
