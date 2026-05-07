# 多卡GPU监控与SM执行模型面试整理

## 来源信息

- 标题：多卡 GPU 监控与 SM 执行模型面试整理
- 作者：LLM 归纳整理
- 日期：2026-05-07
- 类型：面试问答整理 / query report
- 原始文件：[[../../output/reports/多卡GPU监控与SM执行模型面试整理|多卡GPU监控与SM执行模型面试整理]]

## 2-3 条核心摘要

- 多卡 GPU 监控要分成业务层、作业层、设备层和通信层，不能只看 `nvidia-smi` 或单一 GPU util。
- `nvidia-smi` 的显存数是真实的驱动/进程视角容量信号，但不等于框架内部活跃 tensor 占用；分析 OOM 或显存泄漏时要结合 allocator、KV cache 和 workspace 视角。
- SM/warp 执行模型可以直接解释 LLM 推理性能：prefill 更容易打满 Tensor Core，decode 更容易受小 batch、KV cache 读写和 HBM 带宽限制。

## 值得关注的论断

- 判断 LLM 推理是否退化，需要把 `Tensor Active / FP32 pipe active / BF16 pipe active / SM Active / SM Issue / DRAM Bandwidth` 与 prefill、decode 阶段对应起来，而不是只看平均 GPU 利用率。
- `SM Active`、`SM Issue` 和 `Occupancy` 不等价：前者看 SM 是否有 warp，第二个看 scheduler 是否发射指令，第三个看可驻留 warp 数是否足够隐藏延迟。
- 低精度推理是否真的更快，取决于 dtype、shape、layout、backend 和 kernel 是否命中 Tensor Core 路径；否则可能显存省了，但吞吐没有明显提升。

## 关键概念

- [[../concepts/Profiling]]
- [[../concepts/GPU执行模型]]
- [[../concepts/Occupancy]]
- [[../concepts/Warp Divergence]]
- [[../concepts/混合精度训练与推理]]
- [[../concepts/KV Cache]]
- [[../concepts/FlashAttention]]
- [[../concepts/Continuous Batching]]

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/vLLM]]
- [[../entities/TensorRT-LLM]]

## 与现有 wiki 的关系

- 更新概念页：`Profiling`、`GPU执行模型`、`Occupancy`、`Warp Divergence`
- 可继续关联概念页：`混合精度训练与推理`、`KV Cache`、`FlashAttention`、`Continuous Batching`
- 是否存在冲突：未发现直接冲突；本整理主要把既有 CUDA/GPU profiling 知识映射到 LLM 推理面试问答场景

## 待确认

- 若后续需要绑定具体硬件指标阈值，应按 A100/H100/H200/B200 等具体架构分别记录，不宜泛化为统一数值。
