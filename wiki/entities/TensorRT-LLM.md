# TensorRT-LLM

## 一句话说明

NVIDIA 面向大模型推理部署的高性能推理框架，强调对硬件资源和推理路径的细粒度优化。

## 类型

- 项目 / 推理框架

## 核心信息

- 文中以 `kv_cache_config.free_gpu_memory_fraction` 为例说明 KV Cache 的显存预算配置。
- 它常被用来代表“硬件感知较强”的推理系统实现路线。
- 在多 GPU 和高性能部署语境中，TensorRT-LLM 经常作为工程参考实现出现。
- 从面试口径看，可以把它概括为“围绕 NVIDIA 硬件做图优化、kernel 优化、量化和多卡执行路径优化的推理框架”，重点不是单个功能，而是整条硬件感知推理栈。
- 新来源以 `TensorRT-LLM` 作为性能标杆对照 `SGLang V2`，认为后者在部分 H100 Llama3 serving benchmark 中已接近甚至超过它；该结论来自文章转述和外部博客图表，仍需按版本、硬件、模型和参数复核。

## 相关概念

- [[KV Cache]]
- [[Tensor Parallelism]]
- [[PD分离]]
- [[LLM Programs]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/SGLang：LLM推理引擎发展新方向]]

## 冲突与备注

- 本文只引用了一个配置示例，不足以代表 TensorRT-LLM 的整体设计权衡
- 现有 wiki 对它的细节仍偏少；后续若要系统比较 `vLLM / TGI / TensorRT-LLM / SGLang`，需要补更完整的官方文档或实践来源
