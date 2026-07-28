# vLLM CUDA Graph Piecewise 与 Full Decode Only

## 背景

用户问题中的 `picewise` 应为 `PIECEWISE`。二者是 vLLM V1 的 `cudagraph_mode`，控制哪些 batch、以多大粒度使用 CUDA Graph。详见 [[../../wiki/concepts/CUDA Graph 执行模式|CUDA Graph 执行模式]]。

版本口径：本文交叉核对 vLLM 官方源码 commit `439f336`；该配置被官方标为可能变化。

## 核心观点

- `PIECEWISE`：**各种 batch 都可尝试使用，但只 capture 模型里的安全子图**；attention 等 splitting ops 通常留在图外。
- `FULL_DECODE_ONLY`：**只对满足条件的纯 decode batch capture 整个模型 forward**；prefill 和 mixed batch 完全不用 CUDA Graph。
- 前者优先灵活性和覆盖 workload 类型，后者优先纯 decode 的 graph 覆盖率与低 launch 开销。
- 如果既要纯 decode 的 full graph，又要 prefill/mixed 的 piecewise graph，通常应看默认的 `FULL_AND_PIECEWISE`，而不是在这两者中二选一。

## 机制拆解

### 已知事实

vLLM 源码枚举的组合语义为：

```text
PIECEWISE          = PIECEWISE
FULL_DECODE_ONLY   = (decode=FULL, mixed=NONE)
FULL_AND_PIECEWISE = (decode=FULL, mixed=PIECEWISE)
```

`PIECEWISE` 依赖 vLLM compile 进行图分区。默认 splitting ops 主要是 attention ops：分区后，兼容 CUDA Graph 的片段被分别 capture，不兼容片段在图外运行。

`FULL_DECODE_ONLY` 在纯 decode 上使用顶层 full graph。当前 runtime 的精确条件是 uniform decode；普通自回归 decode 通常是每请求 1 token，投机解码时可以是统一的 `1 + num_speculative_tokens`。prefill 或 mixed batch 走 `NONE`。

### 实现差异

| 维度 | `PIECEWISE` | `FULL_DECODE_ONLY` |
|---|---|---|
| capture 对象 | 多个 graph-safe partition | decode 的完整 forward |
| 纯 decode | 部分图 | 完整图 |
| prefill | 部分图 | 不用图 |
| prefill + decode mixed batch | 部分图 | 不用图 |
| attention | 通常在图外 | full graph 支持时在图内 |
| compile 依赖 | 必须 piecewise compile | full graph 可独立于 compile 使用 |
| backend 兼容性 | 通常更宽 | 要求更严格 |

### 性能权衡

- `PIECEWISE` 仍能减少许多 kernel launch，但每层/每个 partition 之间仍有 replay 和图外算子边界；纯 decode 的 launch 降幅通常不如 full graph。
- `FULL_DECODE_ONLY` 对 decode 的 CPU 提交开销更友好，但它没有优化 prefill/mixed batch 的 graph launch。
- CUDA Graph 要为捕获尺寸和其他 key 准备静态 buffer。`FULL_DECODE_ONLY` 避免为 prefill/mixed 路径建立 piecewise graphs，因此相对 `FULL_AND_PIECEWISE` 可节省部分预热、capture 与显存成本。
- 两种模式都可能因 batch 超过最大 capture size、没有匹配 graph key、LoRA 形态或 backend 限制而回退到无 CUDA Graph 路径。

## 对比分析：怎么选

### 适合 `PIECEWISE`

- 单实例同时承载 prefill、decode 和 mixed batch。
- continuous batching / chunked prefill 使 batch 形态较动态。
- attention backend 或模型路径无法安全进入 full graph。
- 希望比 eager 更快，但更看重兼容性、功能覆盖和调试稳定性。

### 适合 `FULL_DECODE_ONLY`

- P/D 分离中的 decode worker。
- workload 几乎全是 uniform decode，关注 TPOT 或低 batch decode latency。
- full graph 已确认兼容，而 prefill 性能不重要。
- 不想承担 `FULL_AND_PIECEWISE` 中 prefill/mixed piecewise graph 的额外 capture 和显存成本。

### 更常见的折中

如果一个服务既有明显的 decode 热路径，也会执行 prefill/mixed batch，而且 backend 支持，`FULL_AND_PIECEWISE` 通常更合理：

- decode → `FULL`
- prefill / mixed → `PIECEWISE`

这也是当前源码注释所称“对多数模型性能最好”的 V1 默认模式。

## 工程含义

1. 不要把 `FULL_DECODE_ONLY` 理解成“只编译 decode”：它说的是 CUDA Graph capture 范围，不等于 `torch.compile` 范围。
2. 不要把 `PIECEWISE` 理解成“任何算子都在若干 CUDA Graph 里”：splitting ops 明确留在 graph 外。
3. 对 decode-only 实例，先验证 attention backend、并行策略、LoRA/spec decode 是否支持 full graph，再比较 TPOT、吞吐、capture 时间与 graph memory。
4. `cudagraph_mode=NONE` 只关 CUDA Graph；若需完全禁用 vLLM compile 集成，应区分 `--enforce-eager`。

## 待核实

- 部署版本中具体 attention backend 是否支持 full graph。
- 实际流量是否经常形成 uniform decode；若经常 mixed/chunked prefill，`FULL_DECODE_ONLY` 的命中率会下降。
- 不提供通用性能百分比：收益取决于模型大小、batch、GPU、capture sizes、backend 和 CPU launch 是否为瓶颈。

## 核对来源

- [vLLM `compilation.py` @ `439f336`](https://github.com/vllm-project/vllm/blob/439f336212227833e126526d3c5f3ef3968dfbf5/vllm/config/compilation.py)
- [vLLM `cudagraph_dispatcher.py` @ `439f336`](https://github.com/vllm-project/vllm/blob/439f336212227833e126526d3c5f3ef3968dfbf5/vllm/v1/cudagraph_dispatcher.py)
- [[../../wiki/entities/vLLM|vLLM]]
- [[../../wiki/concepts/Torch Compile|Torch Compile]]
