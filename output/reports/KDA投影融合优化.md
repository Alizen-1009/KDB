# KDA 投影融合优化

> Kimi K3 的“投影融合”不是把整层 KDA 从 `x` 到输出塞进一个巨型 kernel。更准确地说，它包含三个层次：多个输入投影合成一次 GEMM、Q/K/V 使用 packed convolution layout、Decode 将 Conv + KDA recurrence + Output Gate + RMSNorm 融合。Prefill 与 Decode 的融合边界不同。

相关页面：[[../../wiki/concepts/KDA|KDA]]、[[../../wiki/entities/FlashKDA|FlashKDA]]、[[../../wiki/entities/vLLM|vLLM]]。

---

## 1. 技术报告与vLLM源码分别明确了什么

### Kimi K3 Technical Report

技术报告§5.4.2明确描述的Decode融合范围是：

```text
Short Convolution
Input Normalization
Gating
KDA Recurrence
Output Normalization
```

它还明确说投机验证只缓存比State小得多的`projected inputs`，再在片上Replay accepted tokens。这反而说明Input Projection通常先在融合Recurrent Loop外完成，Replay缓存的是Projection结果。

报告没有把`Q/K/V/G/F_A/Beta合成一个Merged GEMM`作为独立算法贡献详细展开，也没有明确声称从`hidden_states`开始的Input Linear已经与KDA Decode Core放进同一个Kernel。技术报告中明确的Projection Fusion主要出现在Stable LatentMoE，而不是KDA Decode段落。

### 开源vLLM

当前检查的vLLM源码明确实现了两层优化：

```text
1. Merged Input Projection：
   Q/K/V/G/F_A/Beta由一次Column-Parallel GEMM产生

2. Fused Decode Core：
   Packed QKV Conv + KDA + Output Gate + RMSNorm
```

两者之间仍有独立的Decay第二级Projection，最后仍有独立`o_proj`。因此“开源vLLM有KDA投影优化”是成立的，但应叫**Merged Projection + Fused Recurrent Core**，而不是“整层Projection与KDA全部融合成一个Kernel”。

vLLM官方Preview博客使用了“fused KDA projections and convolution”等概括性表述；具体到当前源码，最好以上述Kernel边界为准，并保留Release Branch版本差异。

## 2. 未优化时有哪些投影

KDA从hidden state生成：

```text
q = W_q x
k = W_k x
v = W_v x

g_out = W_g x       # full-rank output gate
f_a = W_a x          # decay低秩中间量
g_decay = W_b f_a    # channel-wise decay logits

beta = W_beta x      # Delta write gate
```

若每项单独执行，主输入`x`会被多个GEMM重复读取，并产生多个Kernel Launch和中间Tensor。

K3的两级Decay Projection不能完全化成一层，因为：

```text
x -> f_a -> g_decay
```

第二层依赖第一层结果。但第一层`f_a`可以与Q/K/V、Output Gate和Beta一起计算。

---

## 3. Merged Input Projection

当前vLLM K3源码使用一个Merged Column-Parallel Linear：

```python
projected_qkvgfab = in_proj_qkvgfab(hidden_states)
```

一次GEMM同时产生：

```text
Q
K
V
G：full-rank output gate
F_A：decay低秩中间量
Beta：Delta write gate
```

然后只做view/split：

```python
mixed_qkv, g_proj_states, f_a, beta = split(projected_qkvgfab)
```

再用较小的第二层投影：

```python
g_decay = f_b_proj(f_a)
```

因此结构从：

```text
多个独立的大输入GEMM
```

变为：

```text
1个Merged Input GEMM
+ 1个依赖f_a的小型Decay Projection
```

---

## 4. K3 TP8下的具体Shape

公开配置：

```text
hidden d = 7168
heads H = 96
head_dim D = 128
projection_size = H × D = 12288
TP = 8
local_heads = 12
local_projection_size = 1536
```

每个TP rank的Merged Projection输出：

```text
Q:       1536
K:       1536
V:       1536
G_out:   1536
F_A:      128   # 在TP ranks复制
Beta:      12   # 每个local head一个
----------------
总计:    6284
```

源码再补4列padding到：

```text
6288，16对齐
```

目的是命中对齐的BF16 GEMM路径。

`F_A [128]`在TP ranks间复制，是因为后续`f_b_proj`再按输出heads做Column Parallel，所有ranks需要同一份低秩输入；Q/K/V/G/Beta则按local heads切分。

---

## 5. 为什么一次大GEMM通常优于多个小GEMM

### 4.1 少读Hidden State

分开执行：

```text
Wq读取x
Wk再次读取x
Wv再次读取x
Wg再次读取x
Wa再次读取x
Wbeta再次读取x
```

Merged GEMM可以在同一计算中复用`x` tile。

### 4.2 减少Kernel Launch

KDA有69层。每层减少几个Projection Launch后，Prefill和TPOT的累计收益会明显放大。

### 4.3 增大GEMM的N维

多个窄输出Projection合并成一个更宽的输出矩阵，更容易获得好的Tensor Core tile利用率。

### 4.4 减少框架与调度开销

不再为每个分支独立创建Linear调用、调度节点和Kernel边界；Split通常只是view，不需要复制数据。

---

## 6. Packed Q/K/V Layout

Merged Projection的前三段不立即拆成三个独立Tensor，而是先保持：

```text
mixed_qkv: [tokens, 3 × local_projection_size]
```

K3为Q/K/V使用同一个packed Conv参数与Conv State：

```text
conv weight:
[Q conv | K conv | V conv]

conv state:
[Q state | K state | V state]
```

这样Decode可以用一次packed causal-conv update处理Q/K/V，而不是分别启动三个Conv更新。

源码注释的设计目标是：

```text
One packed parameter and cache
→ decode runs a single conv update
```

---

## 7. Decode真正融合了什么

支持Shape和GPU架构时，Decode调用大致为：

```python
fused_kda_decode(
    x=mixed_qkv,
    conv_weight=packed_conv_weight,
    conv_state=conv_state,
    raw_decay=g_decay,
    raw_beta=beta,
    recurrent_state=matrix_state,
    output_gate=g_out,
    norm_weight=rmsnorm_weight,
)
```

这个Kernel融合：

```text
Packed Q/K/V ShortConv Update
→ SiLU
→ Q/K Normalization
→ Decay/Beta Gate
→ KDA Matrix-State Update
→ Query Read
→ Full-rank Output Gate
→ RMSNorm
```

最终输出再进入单独的Row-Parallel `o_proj`。

所以当前源码下更准确的Kernel边界是：

```text
Merged Input Projection GEMM       # 单独
Decay f_b Projection              # 单独
Fused Conv + KDA + Gate + Norm    # 一个融合核心
Output Projection                 # 单独
```

**Input Projection本身并没有和整个Decode recurrence放进同一个kernel。**“Fused KDA Projection and Convolution”更准确地理解为：输入投影分支被Merged，Q/K/V以packed形式直接喂给单次Conv/KDA融合路径，减少中间拆分和多次Conv更新。

---

## 8. 为什么Decode特别需要这种融合

Decode通常每请求只有一个新Token：

```text
M维很小
Projection/Conv/KDA kernels都较短
```

如果分开执行：

```text
Q Conv kernel
K Conv kernel
V Conv kernel
Norm kernel
Gate kernel
State update kernel
Output kernel
```

Kernel Launch、State索引、HBM往返和同步开销会占很大比例。

K3有69层KDA，一层多几微小Kernel会在一个Token的完整Forward中重复69次，直接累积到TPOT。

融合后，Conv State与Matrix State可在同一个Kernel中读取、更新和写回，中间Q/K/V和Gate结果尽可能留在寄存器/片上存储中。

---

## 9. Prefill的融合边界不同

Prefill有大量tokens，核心使用FlashKDA或Triton Chunk KDA。

当前检查的vLLM源码中，Prefill流程是：

```text
Merged Input Projection
→ Split packed Q/K/V views
→ Q/K/V Causal Conv
→ Gather Initial Matrix States
→ FlashKDA/Triton Chunk Core
→ Fused Output Gate + RMSNorm
→ O Projection
```

Prefill的Q/K/V Conv仍可见三个逻辑调用，因为FlashKDA希望获得dense Q/K/V Tensor；实现特别避免额外的V copy。Initial Matrix States则根据请求的state slots一次Gather成batch initial states。

因此不能把博客中的“fuses input projections and causal convolution”无条件理解为：

```text
整个Prefill Projection + Conv已经是单个Monolithic Kernel
```

更稳妥的解释是：

- 多个Input Projections已经Merged；
- Conv参数、状态与数据布局被Packed/Fused优化；
- Initial States Gather合并处理；
- Release Branch具体是否进一步融合Projection与Conv，需绑定commit核对。

---

## 10. 权重为什么维护两种Conv Layout

Prefill/fallback路径与Fully Fused Decode对Conv权重Layout的需求不同。

当前源码：

- 保留Prefill和Fallback Kernel消费的原始Layout；
- 额外注册width-major的Decode Conv Weight副本；
- Load Weight时一次性填充两个Layout；
- 运行时不再为Decode做Transpose/重排。

这是典型的：

```text
增加少量静态权重副本
换取每Token不做动态Layout Conversion
```

同理，Decode RMSNorm Weight在加载时Upcast/复制到融合Kernel偏好的格式，避免每步转换。

---

## 11. 哪些中间Tensor仍然存在

Projection Fusion并不意味着所有中间结果消失：

```text
projected_qkvgfab仍要生成
mixed_qkv/g_out/f_a/beta是其views
f_b_proj仍要生成raw decay
core output仍要进入o_proj
```

真正减少的是：

- 多次读取输入`x`；
- 多个Projection GEMM Launch；
- Q/K/V独立Conv Launch；
- Conv、Recurrence、Gate、Norm之间的HBM中间量；
- Runtime Layout Conversion。

---

## 12. 与MLA Gate Projection并行的区别

两者不要混淆：

```text
KDA Projection Fusion：
多个输入Projection合并成一个GEMM；packed Q/K/V接入融合Conv/Recurrence

MLA Projection Parallelism：
Gate Projection与Attention主路径相互独立，Decode可放不同CUDA Streams并行
```

KDA的Q/K/V是递推核心输入，不能简单与KDA Core完全并行；只有与Core无依赖的Projection分支或后续阶段才能重叠。

---

## 13. 最小伪代码

```python
def kda_layer_decode(x, state):
    # 一个Merged GEMM，而不是6个独立Linear
    packed = merged_in_proj(x)

    mixed_qkv, output_gate, f_a, beta = split_views(packed)

    # decay的第二级低秩投影仍有数据依赖
    raw_decay = f_b_proj(f_a)

    # packed QKV conv + KDA recurrence + output gate + norm
    core_output, new_state = fused_kda_decode(
        mixed_qkv=mixed_qkv,
        raw_decay=raw_decay,
        raw_beta=beta,
        output_gate=output_gate,
        state=state,
    )

    # Row-parallel output projection
    y = o_proj(core_output)
    return y, new_state
```

---

## 14. 一句话总结

> KDA投影融合的核心不是消灭Projection，而是把共享同一输入`x`的Q/K/V、Full-rank Output Gate、Decay低秩入口和Beta合成一次宽GEMM；Q/K/V保持Packed Layout进入单次Conv更新；Decode再把ShortConv、Normalization、Gate、KDA State读写和Output Norm融合。这样减少重复读取`x`、Kernel Launch、中间HBM流量和动态Layout转换。Prefill因FlashKDA需要dense Q/K/V，融合边界与Decode不同，必须按具体commit区分。

## 来源

- [[../../wiki/sources/A Preview of Production-Scale Kimi K3 Support on vLLM|A Preview of Production-Scale Kimi K3 Support on vLLM]]
- 当前检查的vLLM K3实现：`vllm/models/kimi_k3/nvidia/kda.py`

## 待核实

- 官方博客所指Release Branch与当前检查commit之间的Projection+Conv融合边界。
- Fully Fused Decode支持的GPU、local head count、dtype和Conv State layout会随版本变化。
