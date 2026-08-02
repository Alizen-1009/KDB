# Kimi K3 的 KDA 部署与 Prefix Cache

> 本文聚焦 Kimi K3 Technical Report §5.1 与 §5.4：KDA 在 Prefill、Decode、P/D 分离、Prefix Cache 和投机解码中的部署方式。模型计算公式见 [[KDA伪代码与输入输出]]。

相关页面：[[../../wiki/concepts/KDA|KDA]]、[[../../wiki/entities/Kimi K3|Kimi K3]]、[[../../wiki/concepts/Chunked Gated Delta Rule|Chunked Gated Delta Rule]]。

---

## 1. 核心结论

Kimi K3 是混合注意力模型，每个主要block包含：

```text
3层KDA + 1层Gated MLA
```

因此一个请求同时拥有两类完全不同的历史状态：

```text
KDA：固定大小的递归状态
     Matrix State + ShortConv State

MLA：随token数增长的KV Cache
     按token/page保存
```

Prefix命中只有在同一个token边界同时具备：

```text
MLA KV前缀
+ 所有KDA cache groups的状态checkpoint
```

才可以从该边界继续执行。

---

## 2. 一个请求在运行时保存什么

### 2.1 KDA Matrix State

每个KDA层、每个head维护：

$$
S_t^h\in\mathbb{R}^{d_k\times d_v}
$$

它压缩了该层在前缀 `[0,t)` 上的历史。其大小与上下文长度无关：

$$
O(Hd_kd_v)
$$

但该矩阵通常并不小，所以不能在每个token位置保存一份快照。

### 2.2 KDA Conv State

KDA的Q/K/V经过ShortConv。Decode继续执行时还需要最近 `conv_width-1` 个卷积输入：

```text
Conv State
```

Kimi K3报告在Prefix Cache章节主要用“KDA recurrent state/checkpoint”描述缓存对象，没有单独展开Conv State的打包格式；但从计算语义和当前vLLM实现看，准确恢复KDA层需要同时恢复：

```text
Conv State + Matrix State
```

### 2.3 MLA KV Cache

MLA层仍保留每个历史token的latent KV表示：

```text
MLA cache size ∝ sequence length
```

因此KDA减少了69个KDA层的序列增长状态，但24个Gated MLA层仍需要Paged KV Cache。

---

## 3. 请求状态生命周期

```text
请求开始
  │
  ├─ Prefix lookup
  │    ├─ 找MLA最长匹配前缀
  │    └─ 检查相同边界是否有全部KDA checkpoints
  │
  ├─ 命中边界B
  │    ├─ MLA pages引用/部分page做COW
  │    ├─ KDA checkpoint复制到请求私有running state
  │    └─ 从token B继续Prefill
  │
  ├─ Prefill新token
  │    ├─ FlashKDA更新Matrix State
  │    ├─ ShortConv更新Conv State
  │    ├─ MLA写入新KV
  │    └─ 在合适hash边界保存KDA checkpoint
  │
  ├─ Decode
  │    ├─ KDA状态每token原地更新
  │    └─ MLA KV每token追加
  │
  └─ 请求空闲/驱逐
       └─ KDA状态与对应MLA cache一起offload/evict
```

---

## 4. Prefill 如何部署

### 4.1 FlashKDA Chunkwise Kernel

KDA状态跨token递推，逐tokenPrefill会形成很长串行链。K3使用FlashKDA：

```text
块内：token-parallel矩阵计算
块间：head-parallel状态传播
```

CUTLASS kernel将块内计算与跨块状态传播重叠，服务于：

- 训练；
- 推理Prefill。

报告称其作为Flash Linear Attention的backend自动dispatch。

### 4.2 为什么纯TP不能解决超长Prefill

TP只把heads分到不同GPU：

```text
TP前：每张卡处理更多heads
TP后：每张卡处理更少heads
```

但每个head内部的token recurrence没有变短。TP较大时，每张卡只有少量heads，超长序列Prefill的串行传播阶段可能无法填满SM。

### 4.3 单卡内部Context Parallelism

K3在单个rank内部把序列切成多个segments，分给不同SM：

```text
每个segment并行计算：
1. 对incoming state的累计变换M
2. 从零状态产生的local state

随后合并，恢复每个segment的精确initial state
```

这是SM-level CP，不发生跨GPU通信。

### 4.4 跨卡KDA Context Parallelism（KCP）

普通线性注意力若状态是简单加法，可以对各rank局部状态求前缀和；KDA不行，因为每个token还会用：

$$
M_t=\left(I-\beta_tk_tk_t^\top\right)\operatorname{Diag}(\alpha_t)
$$

变换incoming state。

每个sequence segment必须表示成仿射变换：

$$
S_{\text{out}}=M_{\text{segment}}S_{\text{in}}+\tilde S_{\text{segment}}
$$

各rank局部计算：

```text
M_segment：该段对incoming state的累计作用
S_tilde：该段从零状态生成的state
```

这些segment变换可以结合，因此通过prefix scan恢复各rank的incoming state。报告中的实现以一次all-gather交换这两个固定大小fragment。

与Softmax Context Parallel交换随序列增长的KV blocks不同，KCP通信payload与序列长度无关，但累计transition矩阵本身仍有成本。

---

## 5. Decode 如何部署

Decode时 `T≈1`，不再缺token并行度，主要瓶颈变成：

```text
读取大Matrix State
→ 更新
→ 写回大Matrix State
```

因此K3使用融合Decode kernel，将以下步骤尽量放在同一个kernel中：

```text
ShortConv
→ Q/K normalization
→ gate计算
→ KDA recurrence
→ output normalization
```

Running State按请求占用私有slot，每个token原地更新。

当前vLLM K3源码也采用：

```text
kv_cache = (conv_state, recurrent_state)
```

并通过request对应的`state_indices`从state pool定位状态slot。Prefill取initial state、输出final state；Decode直接通过slot index原地更新。

---

## 6. 为什么普通Prefix Cache不够

Softmax/MLA Prefix Cache保存每个token的KV。只要prefix token hash一致，对应KV blocks就可以共享。

KDA不同：

```text
每个请求每个KDA层只有一个当前状态
而不是每个token一条state记录
```

若只缓存当前最终状态：

```text
prefix [0,10000) -> S_10000
```

就不能恢复：

```text
S_512, S_1024, ..., S_9728
```

除非这些边界曾单独保存checkpoint。

因此，MLA在边界 `B` 命中还不够；KDA必须也有严格对应 `[0,B)` 的状态快照。

---

## 7. Unified Cache Pool

K3将KDA状态与MLA KV pages放进同一个paged block pool：

```text
统一page byte size
统一allocation
统一reference counting
统一eviction
统一transfer管理
```

但page的类型和内容仍不同，不是把KDA状态当作普通token KV。

KDA page内部：

```text
head 0的完整byte stream
head 1的完整byte stream
...
```

每个head的字节连续、自包含，作为跨节点传输的最小单位。

### P/D采用不同TP时

如果Prefill节点和Decode节点TP degree不同：

```text
Prefill：按prefill TP进行head sharding
Decode：按decode TP重新分配heads
```

K3在传输路径中完成re-layout，避免接收后再做GPU-side reshuffle。

注意：这是K3报告中的生产系统设计，不等于任意vLLM部署都已经提供相同的数据面实现。

---

## 8. 物理Page、Hash Block与KDA Checkpoint解耦

### 8.1 如果三者使用同一粒度

KDA checkpoint很大，无法每几个token保存一次。若为了KDA把物理block设成：

```text
1024～6144 tokens
```

并让prefix hash也只能在物理block末尾生成，则：

- 短于一个block的请求完全无法复用；
- Chunked Prefill未跨过完整block前没有cacheable prefix；
- 即使两请求共享几千token，也可能因未对齐物理block而miss。

### 8.2 K3的解法

将三个粒度分开：

```text
Physical cache page：粗粒度，例如6144 tokens
Prefix hash block：  细粒度，例如512 tokens
KDA checkpoint：     只保存在部分hash endpoints
```

示例：

```text
一个6144-token物理page
= 12个512-token hash blocks
```

MLA可以在page内部的每个完整512-token端点注册prefix hash；KDA只在其中一部分端点保留checkpoint。

---

## 9. Checkpoint 如何保存

Prefill每次forward结束后：

```text
1. 找到本次已经处理的最后一个hash-aligned位置
2. 将该位置的KDA state持久化
3. 新请求继续使用独立running state
```

由于checkpoint很大：

- 请求继续前进后，已经被更晚checkpoint覆盖价值的中间checkpoint会回收；
- conversation-turn边界通常保留，用于未来跨请求复用；
- 不在每512 tokens都永久保存一份。

Cached checkpoint是只读snapshot：dui y

```text
命中：snapshot -> copy -> private running state
继续：只修改private state
新checkpoint：写到fresh slot
```

不能让新请求直接在共享checkpoint上原地Decode，否则会污染其他请求的prefix。

---

## 10. Prefix Lookup 两阶段

### Stage 1：MLA候选边界

1. 先匹配完整physical pages；
2. 到第一个不完整/不匹配page时，再匹配page内部的fine hash endpoints；
3. 使用chained hash，端点hash证明从token 0到该边界的完整prefix一致。

得到最长MLA候选边界 `B_mla`。

### Stage 2：KDA checkpoint校验

在MLA候选范围中向前寻找：

```text
所有KDA cache groups都存在checkpoint
```

的最长边界。

最终命中：

$$
B=\max\{b:\operatorname{MLAHit}(b)\land\operatorname{AllKDACheckpoints}(b)\}
$$

它：

- 必须是hash block倍数；
- 不需要是physical page倍数。

### 报告示例

```text
请求匹配前2800 tokens
physical page = 6144 tokens
hash block = 512 tokens

最终B = 2560 = 5 × 512
```

系统执行：

```text
复用5个MLA hash blocks
恢复B=2560对应KDA checkpoint
对部分MLA page执行copy-on-write
从token 2560继续Prefill
不重算[0,2560)
```

---

## 11. 并发一致性

混合cache并发共享存在三个关键约束。

### 11.1 先Pin所有命中对象，再Allocation

所有cache groups共享free list。如果先为某个group分配private page，分配过程可能驱逐另一个group刚命中的page。

因此必须：

```text
先跨所有groups pin hit blocks
再执行任何private allocation
```

### 11.2 当前调度step新分配的page暂不可命中

GPU上的copy在forward前才执行。若一个page刚复用旧slot但copy尚未落地，lookup可能读到上一owner的字节。

因此当step新分配/注册的page，要等copy完成后才能对其他请求可见。

### 11.3 KDA groups必须全有或全无

只有每个KDA cache group都有同一边界checkpoint，才能恢复请求。

若一个group的checkpoint被evict：

```text
原子地使其所有siblings失效
```

不能出现“部分KDA层命中、部分KDA层缺失”的可见状态。

---

## 12. Prefix Cache 与投机解码回滚不是一件事

### Prefix Cache

目的：跨请求/跨轮次复用长前缀。

```text
稀疏KDA state checkpoints
+ MLA KV pages
```

生命周期较长，可能offload到CPU或跨节点传输。

### Speculative Decode Rollback

MTP一次draft多个tokens，KDA running state已经原地前进。如果只接受前几个draft token，需要回到accepted boundary。

最直接方案是每个draft位置保存完整state snapshot，但state流量过大。

K3采用ReplaySSM思路：

```text
不保存每个draft位置的大state
只缓存较小的projected inputs
验证后，在片上重放accepted tokens
重建正确state
写回verified token与bonus token后的状态
```

重放、bonus token与下一draft window放在一个fused recurrent loop中，覆盖ShortConv、norm、gating、KDA recurrence与output norm。

这些projection caches只存在于Decode stage：

- 不改变Prefix Cache payload；
- 不改变P/D disaggregation payload。

---

## 13. Offload 与Fleet调度

在百万token agentic workload中，空闲但可能复用的prefix会挤占GPU显存。

K3采用write-back：

```text
活跃Decode blocks：留在GPU
空闲可复用prefix：只有被GPU驱逐时才写回CPU DRAM
下次复用前prefetch回GPU
```

KDA states和对应MLA KV blocks一起offload/prefetch，使二者生命周期对齐。

Fleet级还使用cache-aware affinity：同一session尽量路由到持有其prefix cache的cluster，因为长前缀miss的重算代价远高于短增量Prefill。

---

## 14. KDA State为什么必须稀疏保存

状态字节数近似：

$$
\text{MatrixBytes/request/rank}
=L_{KDA}\times\frac{H}{TP}\times d_k\times d_v\times\text{bytes(dtype)}
$$

以当前K3/vLLM公开形状作说明性估算：

```text
KDA layers = 69
H = 96
TP = 8
local heads = 12
d_k = d_v = 128
Matrix State dtype = FP32
```

单rank、单层：

$$
12\times128\times128\times4
=786{,}432\ \text{bytes}
\approx0.75\ \text{MiB}
$$

69层：

$$
\approx51.75\ \text{MiB/request/rank}
$$

再加Conv State约为MiB量级。跨TP8求和时，一个请求的完整KDA state约为数百MiB量级，但分片驻留在不同ranks。

这只是按公开结构和当前vLLM state layout计算的说明性估算，不是报告给出的benchmark。它解释了为什么：

```text
不能每512 tokens永久保存一套完整KDA state
而要只保留稀疏checkpoint，尤其是conversation-turn边界
```

---

## 15. 当前vLLM源码能确认什么

在当前检查版本中，vLLM Kimi K3实现明确显示：

```text
KDA cache = (conv_state, recurrent_state)
```

状态形状：

```text
conv_state：[(3×projection_size)/TP, conv_width-1]
recurrent_state：[H/TP, head_dim, head_dim]
```

默认KDA state dtype逻辑为：

```text
Conv State：由model/cache dtype决定
Recurrent State：FP32
```

执行路径：

- Prefill：FlashKDA或Triton chunk KDA，接收initial state并返回final state；
- 普通Decode：按state index执行fused recurrent update；
- Spec Decode：维护spec state indices与accepted token元数据。

FlashKDA当前快速路径要求包括：

```text
CUDA SM90/SM10x/SM12x
BF16
head_dim=128
bounded KDA gate
```

这些是vLLM特定版本的实现约束，不是KDA数学定义。

K3技术报告描述的统一cache pool、细粒度hash、全cache-group原子一致性和跨TP transfer re-layout是Moonshot生产系统设计；不能仅凭模型kernel源码断言某个上游vLLM版本已经逐项实现了同一套Prefix Cache系统。

---

## 16. 最重要的心智模型

```text
MLA KV Cache：
“保存前缀中每个token的可寻址记忆”

KDA Running State：
“保存整个前缀压缩后的当前结果”

KDA Checkpoint：
“在少数可复用边界冻结一份Running State”
```

因此混合模型的Prefix Cache不是只找最长KV hash，而是：

```text
最长可复用前缀
= MLA内容匹配
∩ 同边界KDA状态存在
∩ 所有KDA groups一致
```

## 官方来源

- [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)，§5.1、§5.3、§5.4
- [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)
- [FlashKDA](https://github.com/MoonshotAI/FlashKDA)
- vLLM Kimi K3 KDA implementation，检查版本：`vllm/models/kimi_k3/nvidia/kda.py`

## 待核实边界

- 报告没有公开生产cache manager全部代码，KDA checkpoint是否把Conv State打包在同一page、独立page还是可重建，需以生产实现为准。
- Prefix cache checkpoint保留策略是conversation-aware且受容量影响，不应理解为固定每512 tokens永久保存。
- 具体physical page大小、hash block大小、state dtype、TP/DP布局和P/D transfer协议均可随backend版本与部署调整；6144/512是报告示例。
