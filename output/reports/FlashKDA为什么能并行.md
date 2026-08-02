# FlashKDA 为什么能并行

> KDA逐token公式是递归的，但状态更新对旧状态是**仿射变换**。因此可以把一段token压缩成可组合的segment transition，并把chunk内部的token依赖改写为lower-triangular GEMM。FlashKDA利用这一结构在训练和Prefill中并行，不代表Decode `T=1`也有序列并行度。

相关页面：[[../../wiki/concepts/KDA|KDA]]、[[../../wiki/concepts/Chunked Gated Delta Rule|Chunked Gated Delta Rule]]、[[../../wiki/concepts/线性注意力递归状态|线性注意力递归状态]]。

---

## 1. FLA 与 FlashKDA 不是同一个名字

### FLA

`FLA`通常指 [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)：

- 提供GDN、DeltaNet、KDA等线性注意力算子；
- 包含naive recurrent correctness reference；
- 包含Triton chunk/recurrent kernels；
- 为不同模型和backend提供统一接口。

### FlashKDA

`FlashKDA`是Moonshot针对KDA开发的CUTLASS chunkwise kernel：

- 面向训练和Inference Prefill；
- 将块内token计算与块间state传播重叠；
- 利用K3 lower-bounded gate使所有secondary tiles都能走dense Tensor Core GEMM；
- 可作为FLA的KDA backend被自动dispatch。

因此：

```text
FLA：线性注意力算子库/框架
FlashKDA：其中可被调用的KDA专用高性能backend
```

---

## 2. 看起来为什么不能并行

KDA逐token递推：

$$
\bar S_t=\operatorname{Diag}(\alpha_t)S_{t-1}
$$

$$
\hat v_t=k_t^\top\bar S_t
$$

$$
S_t=\bar S_t+\beta_tk_t(v_t-\hat v_t)^\top
$$

$$
o_t=S_t^\top q_t
$$

直接执行是：

```text
S_0 -> token 1 -> S_1 -> token 2 -> S_2 -> ...
```

`S_t`依赖`S_{t-1}`，所以朴素实现无法同时计算所有tokens。

---

## 3. 第一把钥匙：更新对旧状态是仿射的

把KDA更新展开：

$$
S_t
=\left(I-\beta_tk_tk_t^\top\right)
\operatorname{Diag}(\alpha_t)S_{t-1}
+\beta_tk_tv_t^\top
$$

定义：

$$
M_t=
\left(I-\beta_tk_tk_t^\top\right)
\operatorname{Diag}(\alpha_t)
$$

$$
B_t=\beta_tk_tv_t^\top
$$

则：

$$
\boxed{S_t=M_tS_{t-1}+B_t}
$$

虽然`S_t`递归，但`M_t`和`B_t`只由当前token的`k/v/alpha/beta`决定，不依赖`S_{t-1}`的具体数值。

因此每个token都可以先独立生成一个状态变换：

```text
F_t(S) = M_t S + B_t
```

---

## 4. 第二把钥匙：仿射变换可以结合

连续两个token：

$$
S_1=M_1S_0+B_1
$$

$$
S_2=M_2S_1+B_2
$$

代入：

$$
S_2=M_2M_1S_0+M_2B_1+B_2
$$

所以两个token可以合成一个大变换：

$$
M_{2:1}=M_2M_1
$$

$$
B_{2:1}=M_2B_1+B_2
$$

定义组合操作：

$$
(M_2,B_2)\circ(M_1,B_1)
=
(M_2M_1,M_2B_1+B_2)
$$

该操作满足结合律：

$$
F_3\circ(F_2\circ F_1)
=(F_3\circ F_2)\circ F_1
$$

因此可以使用parallel prefix scan，像并行前缀和一样组合token segments。

这也是KDA Context Parallel可以先在不同SM/GPU上独立计算segment transition，再恢复每段exact initial state的数学基础。

---

## 5. 为什么不能直接构造所有 M_t

理论上做scan需要：

$$
M_t\in\mathbb{R}^{d_k\times d_k}
$$

但若`d_k=128`，每token显式存储并相乘这些矩阵会很贵。

KDA的特殊结构是：

$$
M_t=(I-\beta_tk_tk_t^\top)\operatorname{Diag}(\alpha_t)
$$

其中：

- `Diag(alpha)`是对角缩放；
- $k_tk_t^\top$是rank-1矩阵。

因此不需要把每个`M_t`完全物化，而可以使用UT/WY一类低秩变换，把一串rank-1 updates整理成矩阵乘法。

---

## 6. Chunkwise并行

将长度`T`切成多个chunk，每个chunk包含`C`个tokens：

```text
chunk 0 | chunk 1 | chunk 2 | ...
```

### Chunk之间

最基础的chunk算法仍然传递final state：

```text
S_in[0] -> chunk 0 -> S_out[0]
                       ↓
                    S_in[1] -> chunk 1 -> ...
```

因此跨chunk仍有递归依赖。

### Chunk内部

通过展开递推，将`C`个token之间的依赖写成lower-triangular矩阵：

```text
        key token
        0 1 2 3
query 0 x . . .
      1 x x . .
      2 x x x .
      3 x x x x
```

第`j`个token只能依赖`i<=j`的tokens，所以使用causal `Tril`。

原本的逐tokenloop变成：

```text
QK^T GEMM
→ causal lower triangle
→ triangular matrix × transformed V
```

GPU可以并行计算chunk中大量token pairs。

---

## 7. KDA官方Chunk公式

一个chunk内定义channel-wise cumulative retention：

$$
\gamma_{i\to j}
=\prod_{r=i}^{j}\alpha_r
$$

令：

$$
\Gamma^{1\to C}
\in\mathbb{R}^{C\times d_k}
$$

保存从chunk开头到各位置的累计retention。

UT transform生成：

```text
U: [C,d_v]
W: [C,d_k]
```

并构造pseudo-value：

$$
\widetilde V=U-WS_{in}
$$

chunk内causal关联矩阵：

$$
A=
\operatorname{Tril}
\left[
(Q\odot\Gamma)
(K/\Gamma)^\top
\right]
$$

输出：

$$
\boxed{
O=
(\Gamma\odot Q)S_{in}
+A\widetilde V
}
$$

形状：

```text
Gamma * Q: [C,K]
S_in:       [K,V]
inter:      [C,V]

A:          [C,C]
V_tilde:    [C,V]
intra:      [C,V]

O:          [C,V]
```

两个部分的含义：

```text
(Gamma * Q) @ S_in：
读取进入chunk之前的历史，inter-chunk contribution

A @ V_tilde：
处理当前chunk内部token之间的Delta updates，intra-chunk contribution
```

这些都是规则GEMM，不再逐token启动kernel。

`Tril`保留对角线，因为KDA是read-after-write：当前token先更新state，再用当前query读取。

---

## 8. 为什么 Gamma 会引发数值问题

Chunk公式包含：

$$
K/\Gamma
$$

而：

$$
\Gamma=\prod_t\alpha_t
$$

若某些`alpha`极小，`Gamma`会接近0：

$$
1/\Gamma\to\infty
$$

旧Kimi Linear采用无下界log-decay，必须：

- 在log space计算relative decay；
- 将chunk再切成16-token secondary tiles；
- 对角tile使用特殊position-pair kernel；
- 只有非对角tiles直接使用dense Tensor Core GEMM。

---

## 9. Kimi K3 Lower-Bounded Gate为什么帮助并行

K3使用：

$$
g_t=g_{min}\sigma(e^Az_t),\qquad g_{min}=-5
$$

$$
\alpha_t=e^{g_t}
$$

因此单步：

$$
g_t\in(-5,0)
$$

16-token tile内：

$$
\sum_{t=1}^{16}g_t\in(-80,0)
$$

$$
1/\Gamma<e^{80}
$$

仍在BF16动态范围内。

结果是：

```text
旧路径：
非对角tile -> Tensor Core GEMM
对角tile   -> position-pair特殊kernel

K3：
非对角tile -> Tensor Core GEMM
对角tile   -> Tensor Core GEMM
```

它不仅提高数值稳定性，还使所有causal tiles都使用统一、高吞吐的矩阵乘路径。

---

## 10. FlashKDA实际并行哪些维度

### Token-parallel

chunk内部的`C`个tokens通过triangular GEMM并行。

### Head-parallel

不同KDA heads有独立state，可以并行处理。TP也可将heads分到不同ranks。

### Batch/Sequence-parallel

不同请求、不同packed sequences可并行。

### Segment/Context-parallel

长序列可切成segments。每段先独立计算仿射transition，再通过prefix scan恢复incoming state。

### Pipeline/Overlap

K3报告称FlashKDA将：

```text
intra-chunk token-parallel computation
与
cross-chunk head-parallel state propagation
```

重叠调度，减少串行状态传播阶段SM空闲。

---

## 11. “并行”不代表没有串行部分

需要区分：

```text
朴素Recurrent：
T个token完全逐个执行

Chunkwise：
chunk内并行，chunk间传state

Context Parallel Scan：
segments先并行产生transition，再做scan/合并
```

所以FlashKDA不是把任意长度递推变成完全无依赖的单次GEMM，而是：

1. 将大部分工作变成Tensor Core友好的块矩阵计算；
2. 将无法消除的state依赖压缩为较少的chunk/segment transition；
3. 通过scan、head parallel与计算重叠降低串行部分的占比。

---

## 12. 为什么Decode不用同一套Chunk并行

Decode通常：

```text
T = 1
```

没有chunk内部token并行度，也没有`C×C`三角矩阵值得构造。

所以Decode使用fused recurrent kernel：

```text
ShortConv
→ Q/K Norm
→ Gate
→ 读取Matrix State
→ Delta update
→ 写回State
→ Output Norm
```

此时主要瓶颈是固定Matrix State的读写带宽，而不是长序列递归。

因此：

```text
FlashKDA chunk kernel：训练与Prefill
Fused recurrent KDA：Decode
```

---

## 13. 当前vLLM中的backend边界

当前检查版本的vLLM K3实现：

- Prefill可选择`flashkda`或`triton`；
- FlashKDA快速路径要求CUDA、BF16、`head_dim=128`和bounded gate；
- 支持的GPU代际包括SM90/SM10x/SM12x；
- 接收initial recurrent state并输出final state；
- 普通Decode走独立的fused recurrent KDA路径。

这些是版本相关实现约束，不是KDA数学本身的限制。

---

## 14. 最小心智模型

```text
逐tokenKDA：
S_t = M_t S_{t-1} + B_t

为什么可并行：
(M,B)仿射变换可以结合

chunk内怎么并行：
展开成causal lower-triangular GEMM

chunk间怎么办：
传递final state，或组合segment transition做prefix scan

FlashKDA多做了什么：
CUTLASS/Tensor Core实现
+ lower-bounded decay
+ 对角/非对角tile统一GEMM
+ token计算与state传播重叠
```

## 官方来源

- [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)，§2.1.1、§5.1
- [FlashKDA](https://github.com/MoonshotAI/FlashKDA)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
- 当前vLLM K3 KDA实现：`vllm/models/kimi_k3/nvidia/kda.py`

## 待核实

- FlashKDA公开仓库与vLLM集成的具体tile size、pipeline stages、workspace和支持shape会随版本变化。
- 报告说明了token-parallel stage与head-parallel recurrence的重叠，但完整CUTLASS schedule需结合对应commit源码与profile验证。
