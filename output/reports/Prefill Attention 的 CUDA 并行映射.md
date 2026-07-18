# Prefill Attention 的 CUDA 并行映射

## 问题

Prefill 阶段的 attention 如何在 CUDA 中并行？调度到一个 SM 上的 thread block 具体在做什么？

## 一句话答案

在典型 FlashAttention-2 风格的 fused prefill kernel 中，**一个 CUDA thread block 通常承包一个 `batch × Q-head × Q-row tile` 的最终输出**：它把这块 `Q` 留在片上，沿序列方向一块块流式读取 `K/V`，在多个 warp 中完成 `QK^T → online softmax → PV`，最后只把完成的输出 tile 写回 HBM。

## 先纠正一个 CUDA 执行模型误区

- 不是“每个 SM 固定运行一个 block”。
- grid 中通常有远多于 SM 数量的 blocks，硬件把它们分波次动态调度到空闲 SM。
- 一个 block 从开始到结束只驻留在一个 SM 上，不会跨 SM 迁移。
- 一个 SM 常能同时驻留多个 block；能驻留多少受 threads、registers 和 shared memory 限制。
- SM 真正调度执行的单位是 warp。多个驻留 block 的 warps 可以交错执行，用来隐藏访存和指令延迟。

## 从张量到 CUDA grid

忽略 QKV projection，从已经生成的张量开始：

```text
Q: [B, Hq,  Nq, D]
K: [B, Hkv, Nk, D]
V: [B, Hkv, Nk, D]
O: [B, Hq,  Nq, D]
```

### 变长 batch 中的 `cu_seqlens`

FlashAttention varlen 等接口常把多个 request 的 token 拼成一个连续 buffer，再用累计边界 `cu_seqlens` 分隔它们。标准形式以 `0` 开头：

```text
request 长度: [2, 3, 1]
cu_seqlens:    [0, 2, 5, 6]
```

因此 `num_requests = len(cu_seqlens) - 1`，第 `i` 个 request 的长度是 `cu_seqlens[i+1] - cu_seqlens[i]`，最后一个值是 packed buffer 中的 token 总数。按这个约定，单个长度为 `2` 的 request 应写成 `[0, 2]`；字面上的 `[2]` 缺少起始边界，对标准 API 来说是不完整的。如果某个框架在日志或 JSON 中专门省略起始 `0`，那么 `[2]` 才可解读为“1 个 request，长度 2”；需以实际接口定义为准。

把 query 序列按 `Br` 行分块。典型 forward launch 可抽象为：

```text
grid ≈ [ceil(Nq / Br), B, Hq]
```

因此 block `(q_tile=i, batch=b, q_head=h)` 拥有的输出是：

```text
O[b, h, i*Br : (i+1)*Br, :]
```

在 GQA/MQA 中，多个 Q head 会映射到同一个 KV head，但 block 的输出所有权仍可按 Q head 理解。

## 一个 block 的完整工作

### 1. 确定自己的输出范围

block 用 `blockIdx` 确定 `batch / head / Q tile`，计算边界。对 causal attention，它还会算出这块 query 最远能看到哪个 KV tile；未来 tile 不需读取或计算。

### 2. 载入 Q tile，初始化行状态

block 的 threads 协作把 `Q_i: [Br, D]` 从 HBM 搬入 shared memory / registers，并为每个 query 行维护：

```text
m     = -inf       # 目前见过的最大 logit
l     = 0          # softmax 分母累加值
O_acc = 0          # 未归一化的输出分子
```

`m/l` 的逻辑形状是 `[Br]`，`O_acc` 的逻辑形状是 `[Br, D]`；它们实际分散在各线程的寄存器片段中。

### 3. 循环处理 K/V tiles

对每个 `K_j/V_j: [Bc, D]`，block 内部做：

1. 多个 warp 协作把 `K_j/V_j` 从 HBM 搬到 shared memory，高性能实现会预取下一 tile。
2. 用 Tensor Core / MMA 协作计算 `S_j = Q_i K_j^T`，局部 score tile 的逻辑形状为 `[Br, Bc]`。
3. 乘上 `1/sqrt(D)`，施加 padding / causal / local mask。
4. 对 `S_j` 的每一行并行求 `max` 和 `sum(exp())`；这是 warp/block 级 reduction，不是一个 thread 串行扫完一行。
5. 使用 online softmax 更新：

```text
m_new = max(m, rowmax(S_j))
alpha = exp(m - m_new)
P_j   = exp(S_j - m_new)
l_new = alpha * l + rowsum(P_j)
O_acc = alpha * O_acc + P_j @ V_j
m, l  = m_new, l_new
```

6. 当前 `K_j/V_j` 使用完后丢弃，继续下一 tile。完整 `Nq × Nk` score / probability matrix 从未写回 HBM。

### 4. 归一化并写回

当该 Q tile 已经看完所有允许的 KV tiles：

```text
O_tile = O_acc / l
```

block 把 `O_tile` 以合并访存写回 HBM，并在训练或特定 backward 需求下保存 log-sum-exp 等小量统计量。

## 三层并行性

| 层级 | 并行的工作 | 典型单位 |
| --- | --- | --- |
| block 之间 | 不同 batch、Q head、Q row tile | 一块最终 `O` tile |
| block 内 warp 之间 | Q/K/V 搬运、`QK^T`/`PV` MMA、行归约 | 一组 MMA / row fragments |
| warp 内 thread 之间 | 向量化 load/store、MMA fragments、shuffle/reduce | 若干元素或 fragment |

KV tile 循环在常规路径上存在 `m/l/O_acc` 数据依赖，所以同一 Q tile 的多个 KV tiles 通常由同一 block 顺序流过。这不等于 GPU “串行”：tile 内有大量并行，不同 Q tiles / heads / batches 也在其他 blocks 中同时计算。

## 数字化例子

设 `B=1, Hq=32, Nq=Nk=4096, D=128, Br=Bc=64`：

- Q tiles 数：`4096 / 64 = 64`
- grid 大小：`64 × 1 × 32 = 2048 blocks`
- 每个 block 产出：某个 head 的 `64 × 128` 输出 tile
- 非 causal：每个 block 流式扫过 64 个 KV tiles
- causal：第 `i` 个 Q tile 只需看 `0..i` 的 KV tiles，对角 tile 再做元素级 mask

如果 GPU 有 108 个 SM，2048 个 blocks 会分多波调度。“平均每个 SM 前后处理约 19 个 blocks”不代表同时驻留 19 个；同时驻留数必须由具体 kernel 的资源用量决定。

## 两个容易混淆的特例

### 未融合的 naive attention

如果 backend 实际调用的是“`QK^T` GEMM kernel → softmax kernel → `PV` GEMM kernel”，那么三个 kernel 里 block 的任务完全不同：GEMM block 负责一个矩阵 tile，softmax block 负责若干 score rows。上文“一个 block 完成 `QK → softmax → PV`”专指 fused FlashAttention-style kernel。

### Split-KV / sequence parallel

当 batch、heads 或 Q tiles 太少，grid 并行度不够时，某些 backend 会把同一 Q tile 的 KV 范围拆给多个 blocks。它们分别产生局部 `m/l/O`，再用 log-sum-exp / online-softmax 规则正确合并。它能换取更多 block 并行，但会增加 workspace、额外写回和 combine kernel 开销。

## 读 kernel 时应该找哪些线索

1. grid 三个维度分别映射什么？
2. `kBlockM / kBlockN / head_dim` 是多少？
3. Q、K、V、score、output accumulator 分别放在 HBM / shared memory / registers 哪一层？
4. KV loop 的边界怎么处理 causal / local / varlen？
5. 多少 warps，warp 之间如何分 Q rows 和 MMA fragments？
6. 是否有 split-KV？如果有，局部结果如何 combine？

## 自测

如果一个 block 已经负责 64 行 Q，为什么不默认再让另一个 block 帮它算后半段 KV？

答案：可以，但两个 block 不能用普通加法直接合并归一化输出；它们需要写出局部 `m/l/O`，再用 online-softmax 的指数补偿做第二阶段合并。常规 prefill grid 并行度已足够时，这个额外开销通常没有必要。

## 来源与可追溯性

- [[../../wiki/concepts/FlashAttention]]
- [[../../wiki/concepts/GPU执行模型]]
- [[../../wiki/concepts/Online Softmax]]
- [[../../wiki/concepts/Tiling]]
- [[../../wiki/sources/Flash Attention 详细解释推演与Pytorch代码实现]]
- [[../../wiki/sources/斯坦福CS336 Lecture 5 - GPUs]]
- 实现交叉检查：[Dao-AILab/flash-attention `flash_fwd_launch_template.h`](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_launch_template.h) 中的典型 forward grid 为 `num_m_block × batch × heads`；具体代码可随版本变化。

## 待核实与实现差异

- 本报告不把 `Br/Bc`、warp 数、驻留 block 数写成固定值；它们依赖 GPU 架构、dtype、head dimension、causal/varlen 路径和 backend autotuning。
- FlashAttention-3、Hopper warp-specialized kernels、cuDNN SDPA、Triton kernels 在 pipeline 和 warp 角色上可能不同，但“输出 tile 所有权 + KV 流式经过 + online softmax”仍是理解大多数 fused prefill attention 的有效起点。
