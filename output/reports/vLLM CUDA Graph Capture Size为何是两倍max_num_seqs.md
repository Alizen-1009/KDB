# vLLM CUDA Graph Capture Size 为何是两倍 max_num_seqs

## 结论

当前检查的vLLM源码不是：

```python
max(512, 2 * max_num_seqs)
```

而是：

```python
decode_query_len = 1 + num_speculative_tokens
max_cudagraph_capture_size = min(
    max_num_seqs * decode_query_len * 2,
    512,
)
max_cudagraph_capture_size = min(
    max_cudagraph_capture_size,
    max_num_batched_tokens,
)
```

其中CUDA Graph `size`表示一次Model Forward的**扁平化Token Batch大小**，不是单条Sequence的Context Length，也不等于并发Sequence数量。

---

## 1. 四个容易混淆的配置

```text
max_model_len
= 单个请求允许的最大上下文长度

max_num_seqs
= Scheduler同时容纳/调度的最大请求数

max_num_batched_tokens
= 一次调度Step最多处理多少新Tokens

max_cudagraph_capture_size
= CUDA Graph覆盖的最大num_tokens Shape
```

CUDA Graph Dispatcher接收的是：

```python
dispatch(num_tokens=...)
```

不是：

```python
dispatch(num_requests=...)
```

---

## 2. 为什么num_tokens可以大于num_seqs

一次Forward的Token数是：

$$
N_{tokens}=\sum_{i=1}^{N_{seqs}}q_i
$$

其中`q_i`是请求`i`在当前Step处理的新Token数。

### 普通Decode

每请求1个新Token：

```text
q_i = 1
num_tokens = num_seqs
```

### Speculative Decode

若每请求验证`k`个Draft Tokens并包含一个额外位置：

```text
uniform_decode_query_len = 1 + k
num_tokens = num_seqs × (1 + k)
```

这就是当前公式显式乘：

```python
decode_query_len = 1 + num_speculative_tokens
```

的原因。

### Prefill或Mixed Batch

一个请求在同一Step可贡献多个Prompt Tokens：

```text
请求A：1个Decode Token
请求B：1个Decode Token
请求C：64个Chunked Prefill Tokens

num_seqs = 3
num_tokens = 66
```

因此CUDA Graph Token Shape天然可能大于`max_num_seqs`。

---

## 3. 额外乘2是什么

在已经计入`decode_query_len`之后再乘2，主要是一个默认Capture Headroom启发式，而不是数学正确性条件：

```text
典型纯Decode Token数
≈ max_num_seqs × decode_query_len

默认Mixed/Piecewise Capture上限
≈ 上述值的2倍
```

它让Piecewise或Mixed路径覆盖一些Token数高于纯Decode上限的Batch，例如包含短Prefill Chunk、非均匀Query Length或Padding后的Shape，同时又不直接按很大的`max_num_batched_tokens`捕获大量Graph。

官方源码注释说明该默认值的目标是折中：

- `max_num_seqs`较小时避免捕获过大Graph导致紧张场景OOM；
- 将上限截断在512，避免捕获很多大Graph显著增加启动时间和Graph显存，而性能收益有限。

源码没有把“2”定义为严格协议不变量；它是经验默认值，用户可通过CompilationConfig显式覆盖。

---

## 4. 为什么FULL Decode实际不一定使用2倍范围

Dispatcher初始化纯FULL Decode Graph Key时还会过滤：

```python
max_num_tokens = (
    uniform_decode_query_len * max_num_seqs
)

capture_sizes_for_decode = [
    x for x in cudagraph_capture_sizes
    if x <= max_num_tokens
]
```

因此普通Decode：

```text
uniform_decode_query_len = 1
FULL Decode最大有效Token数 = max_num_seqs
```

即使全局Candidate Capture List生成到`2×max_num_seqs`，单独的Uniform FULL Decode路径也不会创建超出实际最大Decode Token数的Full Graph Key。

额外的2倍空间主要对：

- PIECEWISE；
- Mixed Batch；
- 非Uniform Token Batch；
- 其他按`num_tokens`调度的Graph路径；

更有意义。

---

## 5. Capture Size还包含Padding Shape

vLLM不会为每个整数Token数都捕获一张Graph。默认候选大致为：

```text
1, 2, 4
8, 16, 24, ... 248
256, 272, 288, ... max
```

Runtime会向上选择最近的Captured Size，并Padding：

```text
实际num_tokens = 137
最近capture size = 144
使用144-token Graph
其中7个位置是Padding
```

因此Graph Shape大于当前实际Token数是正常的。它换取更少的Graph数量、更低Capture时间和更少Graph内存。

---

## 6. 数值例子

### 普通Decode

```text
max_num_seqs = 256
num_speculative_tokens = 0
decode_query_len = 1
```

默认全局上限：

```text
min(256 × 1 × 2, 512) = 512
```

但Uniform FULL Decode有效上限：

```text
256 × 1 = 256 tokens
```

所以：

```text
FULL Decode Graph：最多256 Token Shape
PIECEWISE/Mixed Candidate：最多512 Token Shape
```

### Speculative Decode

```text
max_num_seqs = 32
num_speculative_tokens = 7
decode_query_len = 8
```

全局默认上限：

```text
min(32 × 8 × 2, 512) = 512
```

Uniform Spec Decode实际最大Token数：

```text
32 × 8 = 256
```

同样保留了到512的Mixed/Piecewise Headroom。

### 上限被Token Budget截断

```text
max_num_seqs = 256
默认Capture上限 = 512
max_num_batched_tokens = 384
```

最终：

```text
max_cudagraph_capture_size = min(512,384) = 384
```

---

## 7. 为什么不直接捕获到max_num_batched_tokens

`max_num_batched_tokens`可能是8192甚至更大。若为从小到大的大量Token Shape捕获Graph：

- 初始化时间显著增长；
- 每个Graph需要静态Input/Output/Workspace地址与Graph资源；
- Graph显存增加；
- 大Prefill/Mixed Shape重复出现率可能低；
- 大Batch中CPU Launch Overhead相对GEMM计算占比更小，CUDA Graph边际收益下降。

因此默认策略倾向：

```text
重点捕获高频、Launch-sensitive的小/中Decode Batch
大Shape必要时回退非Graph路径
```

512是默认经验Cap，不是CUDA硬件限制；特定Model Config可以覆盖到1024，用户也可自定义Capture Sizes。

---

## 8. 最简答案

```text
max_num_seqs：请求数上限
CUDA Graph size：本次Forward的Token数Shape
```

一个请求可能在一次Forward贡献多个Tokens，所以：

```text
num_tokens > num_seqs
```

当前默认公式中的：

```text
× decode_query_len
```

用于覆盖Speculative Decode每请求多个Token；额外：

```text
× 2
```

是给Mixed/Piecewise Token Batch保留经验Headroom；`min(...,512)`则限制Capture启动和显存成本。纯Uniform FULL Decode路径仍会把Graph Key过滤到：

```text
max_num_seqs × uniform_decode_query_len
```

## 官方源码

- `vllm/config/vllm.py::_set_cudagraph_sizes`
- `vllm/config/compilation.py::CompilationConfig.max_cudagraph_capture_size`
- `vllm/v1/cudagraph_dispatcher.py::CUDAGraphDispatcher`
- 检查版本：vLLM commit `1ad5182ba95a6f1de23b537d57b860082912b28e`

## 版本边界

如果具体分支确实写的是：

```python
max(512, 2 * max_num_seqs)
```

则与当前主线逻辑不同，需要提供commit或文件路径再解释。`max`会把512变成最低Capture上限，而当前主线的`min`是把512作为默认最高Cap，两者语义相反。
