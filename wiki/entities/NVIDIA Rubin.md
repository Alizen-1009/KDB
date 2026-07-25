---
type: entity
entity_type: 硬件
topic: GPU 编程
sources: 1
updated: 2026-07-25
---

# NVIDIA Rubin

## 一句话说明

`NVIDIA Rubin` 是 NVIDIA 在 Blackwell 之后面向 AI 训练与推理的 GPU 架构；当前知识页依据官方架构博客、PTX 9.4 Developer Preview 及二手分析，尚缺实机验证。

## 类型

- GPU 架构 / 硬件平台

## 核心信息

- 来源转述 Rubin 双计算 Die 共 224 个 SM、896 个 Tensor Core，并把 PCIe/NVLink/NVLink-C2C 等 I/O 移到独立 I/O Die，通过 NV-HBI 合封。
- 来源表格列出的整卡峰值包括 dense NVFP4 35 PFLOPS、dense FP8 17.5 PFLOPS、HBM 带宽 22 TB/s；这些是理论/官方峰值口径，不代表具体 kernel 或模型吞吐。
- 延续 `tcgen05` Tensor Core family，同时提高部分低精度路径的 K 维处理能力，并继续增强 SFU，以缓解 Softmax、SwiGLU 和 epilogue 供给不足。
- PTX 9.4 预览包含 sparse score compression、TMA runtime override、cache `applypriority`、counted fabric 和 LUT decompression 等机制。

## MoE 优化链路

1. Router 选择当前 expert。
2. 一份共享 TensorMap 描述 experts 的公共 shape/layout，TMA runtime override 用 expert id 对应的 base address/dimensions/strides 发起搬运，减少 per-expert descriptor 和 patch/fence 开销。
3. Expert 权重被标成倾向 `evict_last`，在处理多个 token/tile 时提高 L2 复用。
4. Tensor Core 执行 grouped GEMM；增强 SFU 处理 SwiGLU 等 epilogue，目标是减少 GEMM 后的非矩阵瓶颈。
5. Expert 完成 last-use 后，通过 `applypriority` 把相同 tensor footprint 恢复到 `evict_normal`，把有效 L2 容量让给下一个热点 expert。
6. 跨 GPU dispatch/combine 可利用设备发起的 counted put/reduction，让接收端按已访问字节数判断数据 ready，减少独立 barrier、ack 和 atomic flag 协调。

这条链路优化动态 expert 权重寻址、搬运、缓存生命周期、激活 epilogue 和通信同步，不改变 Router/Top-k 或 expert FFN 的数学定义，也不保证各环节能在任意 workload 上同时转化为端到端收益。

## 相关概念

- [[GPU执行模型]]
- [[Programmatic Dependent Launch]]
- [[CUDA内存层次]]
- [[MoE]]
- [[Expert Parallelism]]
- [[通信-计算重叠]]
- [[算子融合]]
- [[混合精度训练与推理]]

## 相关来源

- [[../sources/Nvidia Rubin架构分析预览]]

## 冲突与备注

- 页面中的具体数字来自预览期资料和二手分析，必须绑定来源日期；后续正式产品规格、SKU 与 PTX 文档可能变化。
- 约 25% 频率提升、L2/NV-HBI 有效带宽和 Thread Block dependency 的具体实现是作者推测，不是已确认硬件事实。
- Counted fabric 指令最低 target 与 Rubin 官方功能叙事存在重叠但不等价；需用实际 CUDA/NVLink 支持矩阵核实 Blackwell 与 Rubin 的可用差异。
- Attention 2:4 score compression 可能改变注意力概率分布，应同时评估精度、staging 成本与性能，不能只按非零数量推导 2× 加速。
