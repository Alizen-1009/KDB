# HazyResearch

## 一句话说明

HazyResearch 是 Stanford 相关的机器学习系统研究团队 / 实验室品牌，常发布面向高性能模型训练与推理的系统论文、博客和代码。

## 类型

- 组织 / 研究团队

## 核心信息

- `Look Ma, No Bubbles!` 来源来自 HazyResearch 博客，聚焦 Llama-1B 低延迟 decode 的 megakernel 设计。
- 该来源将低延迟 batch size 1 推理视为 memory-bound workload，并通过单 kernel 化、on-GPU interpreter、shared memory paging 和显式同步追求更高 HBM 带宽利用率。
- 文章指向开源项目 `HazyResearch/Megakernels`。

## 相关概念

- [[Megakernel]]
- [[CUDA Kernel]]
- [[算子融合]]
- [[Roofline 模型]]

## 相关来源

- [[../sources/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B]]

## 冲突与备注

- 当前仅根据博客来源记录；具体作者、代码实现和 benchmark 需在 ingest repo 或论文后进一步补全。
