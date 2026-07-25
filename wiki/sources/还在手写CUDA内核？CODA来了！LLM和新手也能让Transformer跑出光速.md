---
type: source
source_kind: 文章
topic: GPU 编程
updated: 2026-06-21
---

# 还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速

## 来源信息

- 标题：还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速
- 作者：机器之心
- 日期：2026-05-24 编辑；原始资料创建于 2026-06-08
- 类型：文章
- 原始文件：[[../../raw/articles/还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速|还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速]]
- 外部线索：文章指向论文 `CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs` 与代码仓库 `HanGuo97/coda-kernels`

## 2-3 条核心摘要

- 文章把 [[CODA]] 概括为一套面向 Transformer 训练的 `GEMM + epilogue` 编程抽象：固定高性能 GEMM mainloop，把 RMSNorm、SwiGLU、RoPE、残差、交叉熵等 memory-bound 小操作通过代数重写塞进 GEMM epilogue，减少中间张量在 HBM 中的写回和再次读取。
- CODA 不是简单把相邻 PyTorch op 拼成一个更大的 kernel，而是利用 Transformer 中部分操作的代数性质重排计算。例如在 `GEMM-RMSNorm-GEMM` 中，RMSNorm 的行缩放因子可以延后到后续 GEMM 的 epilogue 处理，前一个 GEMM 只需要计算 partial RMS，再由轻量规约合并。
- 文章强调 CODA 的抽象可被人工程序员和 LLM 共同使用：给定 epilogue 原语说明、示例和实现日志，LLM 生成的内核在多数 benchmark 中接近人工实现。

## 值得关注的论断

- CODA 应放在 [[算子融合]] 的大类下理解，但更准确的表述是“围绕 GEMM epilogue 的代数重写 + 融合抽象”，而不是普通的 op-level fusion。
- 文章转述论文称，CODA 在部分 backward kernel 上相对基线可达 `1.6x-1.8x` 加速，完整 Transformer 层前向端到端加速约 `5%-20%`；这些数字需要回到原论文、代码和具体硬件配置核实后再作为稳定 benchmark 引用。
- CODA 与 [[FlashAttention]] 的共同点是都围绕 GPU 内存层级重新组织数据流：FlashAttention 主要让 attention 中间状态留在片上，CODA 则尝试让归一化、激活、残差等非 attention 小操作进入 GEMM epilogue。

## 关键概念

- [[CODA]]
- [[算子融合]]
- [[CUDA Kernel]]
- [[RMSNorm]]
- [[Triton]]
- [[Torch Compile]]
- [[FlashAttention]]

## 相关实体

- Tri Dao
- Han Guo
- NVIDIA CUTLASS / [[../concepts/CuTe DSL|CuTe DSL]]
- Claude Code
- Liger Kernel
- FlashInfer

## 与现有 wiki 的关系

- 更新概念页：[[CODA]]、[[算子融合]]、[[CUDA Kernel]]、[[RMSNorm]]、[[Triton]]、[[Torch Compile]]
- 是否存在冲突：未发现直接冲突。该来源补充了“融合边界不是简单越大越好”的一个正例：CODA 通过代数重写找到适合进入 GEMM epilogue 的局部计算，而不是把所有阶段硬塞进同一个超大 kernel。

## 待确认

- 论文编号、论文内容、代码仓库实现细节和 benchmark 数字需按原论文与 `coda-kernels` 仓库核实。
- 文章称当前 CODA 主要覆盖单 GPU、标准 Transformer 训练中除 attention 和 embedding 外的大量操作；多 GPU、非标准架构和推理 serving 场景的适用性仍需进一步验证。
