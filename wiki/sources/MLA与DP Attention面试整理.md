# MLA与DP Attention面试整理

## 来源信息

- 标题：MLA 与 DP Attention 面试整理
- 作者：对话整理
- 日期：2026-05-11
- 类型：对话整理 / 面试专题笔记
- 原始文件：[[../../raw/articles/MLA与DP Attention面试整理|MLA与DP Attention面试整理]]
- 主要参考：DeepSeek-V2 paper、DeepSeek-V3 inference code、SGLang DP/DPA guide

## 2-3 条核心摘要

- `MHA` decode attention 的核心瓶颈通常来自每步读取完整历史 `K/V cache`，`FP16/BF16` 下粗略约 `1 FLOP/byte`，因此常见表现是 memory-bound。
- `MLA` 通过低秩 KV 联合压缩缓存 latent KV，并用 decoupled RoPE 保留位置编码；推理时通过矩阵吸收把 `W_UK` 移到 query 侧、把 `W_UV` 移到 output 侧，避免显式展开完整历史 K/V。
- `DP Attention` 是多 GPU serving 并行策略，主要解决 DeepSeek/MLA 类模型在普通 TP attention 下 latent KV cache 被重复保存、显存浪费和 batch size 受限的问题。

## 值得关注的论断

- `MLA` 的性能收益不能简单理解为“少算”，更准确是“显著少读 HBM，并用额外或重排后的计算换更高算术强度”。
- `DP Attention` 与 `MLA` 的关系是层级互补：前者是系统并行策略，后者是模型结构；一个避免多卡复制浪费，一个降低单 token cache 成本。

## 关键概念

- [[MLA]]
- [[DP Attention]]
- [[KV Cache]]
- [[Roofline 模型]]
- [[Tensor Parallelism]]

## 相关实体

- [[../entities/DeepSeek-AI]]
- [[../entities/SGLang]]

## 与现有 wiki 的关系

- 创建概念页：`MLA`
- 创建来源页：`MLA与DP Attention面试整理`
- 更新概念页：`KV Cache`、`Roofline 模型`、`DP Attention`
- 更新实体页：`DeepSeek-AI`、`SGLang`
- 是否存在冲突：与现有 wiki 无直接冲突；本次主要把先前零散的 `MHA/GQA/MLA` 算术强度、MLA 计算流程和 DPA serving 策略合并成可复用面试稿。

## 待确认

- 具体模型的 `MLA` latency、吞吐与 compute-bound 边界需要结合模型配置、dtype、硬件 ridge point 和实际 attention backend profiling。
- `DP Attention` 的最佳组合方式与收益依赖 SGLang/vLLM/TensorRT-LLM 等框架版本、并行拓扑和 MoE expert parallel 配置。
