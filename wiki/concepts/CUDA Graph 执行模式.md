---
type: concept
topic: GPU 编程
updated: 2026-07-27
sources: 1
---

# CUDA Graph 执行模式

## 定义

在 vLLM V1 中，`cudagraph_mode` 决定模型执行的哪一部分、哪些 batch 类型使用 CUDA Graph capture/replay。CUDA Graph 通过重放预先捕获的 GPU 工作序列减少 CPU 逐 kernel 提交开销，但要求被捕获路径的地址、形状和控制流满足静态化约束。

> 版本边界：以下模式语义按 vLLM 官方源码 `vllm/config/compilation.py` 与 `vllm/v1/cudagraph_dispatcher.py`（commit `439f336`）整理；该配置仍被官方标为可能变化。

## PIECEWISE

- 先由 vLLM compile 在指定 splitting ops 处分割模型图，再只 capture CUDA Graph 兼容的子图。
- 默认 splitting ops 主要是 attention ops；这些不兼容或需要保持动态性的算子留在 CUDA Graph 外执行。
- 可用于 prefill、纯 decode 和 prefill-decode mixed batch，但单次模型 forward 不是一张完整 CUDA Graph：各个可捕获 partition 分别 replay，中间穿插图外算子。
- 依赖 `VLLM_COMPILE`、非空 `splitting_ops` 和 piecewise compilation；它不是简单地“把完整图切成几张”而不经过编译分区。

作用与权衡：

- 相比完全 eager，减少多个 graph-safe 片段中的 kernel launch 开销。
- 相比 full graph，保留 attention 等动态路径的灵活性，通常兼容更多 backend 和 mixed workload。
- 代价是图覆盖率较低，仍存在多个 partition 的 replay、图外算子提交及边界开销，纯 decode 的极致低延迟通常不如可用的 full graph。

## FULL_DECODE_ONLY

源码把它表示为 `(decode=FULL, mixed=NONE)`：

- 对满足条件的纯 decode batch，capture/replay 包含 attention 在内的完整模型 forward CUDA Graph。
- prefill 与 prefill-decode mixed batch 不使用 CUDA Graph，而不是自动退化为 piecewise graph。
- 当前 dispatcher 更精确地要求 **uniform decode**：batch 中请求具有统一的 query length；普通自回归 decode 通常为每请求 1 token，开启 speculative decoding 时则对应统一的 `1 + num_speculative_tokens`。
- 只有命中已 capture 的 size/LoRA 等 graph key 时才 replay；超过最大 capture size、形态不匹配或该路径不支持 full graph 时会回退 `NONE`。

作用与权衡：

- 完整封装 decode forward，最大化减少 decode 阶段的 CPU launch 开销，适合 decode 占主导、TPOT/低延迟敏感的实例。
- 不为 prefill/mixed batch capture graph，可降低相对 `FULL_AND_PIECEWISE` 的 capture、预热与显存成本。
- 对 full graph 兼容性要求更严格；prefill 或 mixed batch 的图加速收益为零。

## 核心区别

| 维度 | `PIECEWISE` | `FULL_DECODE_ONLY` |
|---|---|---|
| 捕获粒度 | 模型中的 graph-safe 子图 | 纯 decode 的完整模型 forward |
| Attention 等 splitting op | 通常留在 CUDA Graph 外 | 满足 backend 条件时包含在 full graph 内 |
| 纯 decode | piecewise graph | full graph |
| 纯 prefill | piecewise graph | 无 CUDA Graph |
| prefill-decode mixed batch | piecewise graph | 无 CUDA Graph |
| 灵活性/兼容性 | 较高 | 较低，依赖 full graph 支持 |
| decode launch 开销 | 中等 | 通常更低 |
| capture/显存重点 | 多个子图与 capture size | 只为 decode full graph 形态付费 |

## 选择建议

- **混合 serving、chunked prefill、动态 attention backend**：优先 `PIECEWISE`，用部分 graph coverage 换兼容性。
- **P/D 分离中的 decode worker、几乎全是 decode、TPOT 敏感**：优先验证 `FULL_DECODE_ONLY`。
- **同一实例既重视 decode，又希望 prefill/mixed batch 也获得 graph 加速**：若模型和 backend 支持，通常优先默认的 `FULL_AND_PIECEWISE`，即 decode 用 full、其余用 piecewise。
- **排查问题**：若怀疑 full graph/backend 兼容性，可先降到 `PIECEWISE`；若要完全关闭 CUDA Graph，应使用 `NONE`。`NONE` 仍不等价于 `--enforce-eager`，后者还会关闭 compile 集成。

## 相关概念

- [[Torch Compile]]
- [[Continuous Batching]]
- [[Chunked Prefill]]
- [[PD分离]]

## 相关实体

- [[../entities/vLLM]]

## 相关来源

- [[../sources/vLLM Large Scale Serving DeepSeek @ 2.2k toksH200 with Wide-EP]]：记录 `FULL_AND_PIECEWISE` 在 Wide-EP 配置中的使用；模式精确定义以官方源码为准。

## 官方实现核对

- [`vllm/config/compilation.py`](https://github.com/vllm-project/vllm/blob/439f336212227833e126526d3c5f3ef3968dfbf5/vllm/config/compilation.py)：枚举、模式定义、适用说明与 compilation 依赖。
- [`vllm/v1/cudagraph_dispatcher.py`](https://github.com/vllm-project/vllm/blob/439f336212227833e126526d3c5f3ef3968dfbf5/vllm/v1/cudagraph_dispatcher.py)：capture key 初始化、uniform decode 判定入口与 `FULL -> PIECEWISE -> NONE` runtime dispatch。

## 待核实

- 具体模型、attention backend、并行方式、LoRA 与 speculative decoding 组合是否支持 full graph，应绑定部署所用 vLLM 版本实测。
- 两种模式的 TPOT、吞吐、capture 时间和额外显存没有通用固定数字，应在相同模型、硬件、batch 分布和 capture sizes 下 benchmark。
