# Kimi K3 权重构成与 TP8 切分

## 结论

Kimi K3 的**总参数存储几乎全在 MoE experts**，但单 token 的 active compute 没有那么极端：

- 总参数视角：MoE block 约占 `98.5%`，Attention、embedding、vision、dense首层和norm等合计约 `1.5%`；
- active-parameter视角：MoE路径粗估约占 `63%`，剩余 Attention/其它 dense 模块约占 `37%`。

TP8 对 K3 的主要作用是切分总权重和 active compute。MLA 的单 latent KV cache只影响 cache布局，不代表 Attention权重不能TP切分。

相关页面：[[../../wiki/entities/Kimi K3|Kimi K3]]、[[../../wiki/concepts/LatentMoE|LatentMoE]]、[[../../wiki/concepts/Tensor Parallelism|Tensor Parallelism]]、[[../../wiki/concepts/Expert Parallelism|Expert Parallelism]]。

## 官方规格

```text
层数：93
Dense FFN层：1
MoE层：约92
Hidden d：7168
Attention heads：96
KDA / Gated MLA：69 / 24
LatentMoE维度 ℓ：3584
Expert intermediate m：3072
Routed experts E：896
Top-k K：16
Shared experts：2
总参数：约2.78T
Active parameters：约104.2B/token
```

## 总参数账本

以下按 SiTU-GLU 的 gate/up/down 三矩阵做机制级估算，忽略bias、norm和少量实现细节。

### Routed expert

单个 latent routed expert：

```text
Gate: ℓ × m
Up:   ℓ × m
Down: m × ℓ

单expert ≈ 3ℓm
         = 3 × 3584 × 3072
         ≈ 33.03M parameters
```

每个 MoE layer 的896个 routed experts：

```text
33.03M × 896 ≈ 29.60B
```

约92个 MoE layers：

```text
29.60B × 92 ≈ 2.723T
```

仅 routed experts 就约占官方2.78T总参数的：

```text
2.723 / 2.78 ≈ 97.9%
```

### Shared experts

每层有2个 full-width shared experts。若每个使用 `d→m→d` 的 gated FFN，则合计：

```text
3 × d × (2m) × 92
≈ 12.2B
≈ 0.44% total
```

### Latent上下投影与Router

```text
W_down + W_up:
2 × d × ℓ × 92 ≈ 4.73B

Router:
d × E × 92 ≈ 0.59B
```

### 汇总

| 模块 | 粗估参数 | 占2.78T |
| --- | ---: | ---: |
| Routed experts | 2.723T | 97.9% |
| Shared experts | 12.2B | 0.44% |
| Latent上下投影 | 4.7B | 0.17% |
| Router | 0.6B | 0.02% |
| **MoE相关合计** | **约2.740T** | **约98.6%** |
| Attention + embedding + vision + dense首层 + norms等 | 约39.8B | 约1.4% |

最后一项是用官方总参数减去上述结构估算得到的剩余量，**不是纯Attention精确参数量**。它还包含160K vocabulary embedding/LM head、401M vision encoder、第一层dense FFN、norm和其它模块。没有最终checkpoint tensor manifest时，不应把39.8B全部写成Attention。

## Active parameters账本

每 token 只激活16/896 routed experts：

```text
16 × 33.03M × 92 ≈ 48.6B
```

再加每层都执行的：

```text
Shared experts          ≈ 12.2B
Latent down/up          ≈ 4.7B
Router                  ≈ 0.6B
```

MoE active path粗估：

```text
约66.1B / 104.2B ≈ 63.4%
```

其余 Attention/KDA/MLA、embedding、dense首层、vision等 active部分约：

```text
104.2B - 66.1B ≈ 38.1B
约36.6%
```

因此：

> 存储上几乎全是experts；每token计算上，Attention和其它dense模块仍占约三分之一以上，TP8对它们的计算分摊很有价值。

## TP8下Attention如何切

### KDA

96个heads按TP切分：

```text
96 / 8 = 12 heads/rank
```

每个rank持有本地12个KDA heads的投影分片和recurrent states。输入投影通常column-parallel，输出投影row-parallel并在边界做collective。

### Gated MLA

- Q侧仍有96 heads，因此q_b/head相关权重按TP8切为12 heads/rank；
- output projection按row parallel切分；
- q_a/kv_a低秩投影和部分latent projection在当前vLLM通用路径中是replicated；
- absorbed decode的latent KV cache接近单KV-head，因此当前纯TP8会在8 ranks复制24层Gated MLA cache；
- 当前vLLM K3 MLA不支持DCP/PCP，暂时无法沿context消除复制。

## TP8下MoE权重如何切

需要区分纯TP和Expert Parallel。

### 纯TP8

```text
每个rank都有全部896个逻辑experts
但每个expert矩阵只持有约1/8 tensor shard
```

直觉上：

```text
Gate/Up：沿intermediate输出维切
Down：沿intermediate输入维切
输出通过TP collective合并
```

routed expert总权重每rank粗略为：

```text
2.723T / 8 ≈ 340B parameters-equivalent
```

expert weight采用MXFP4时，主payload约170GB/rank，另有scale/metadata/alignment以及其它高精度权重和runtime buffer。

Shared experts也走类似column/row TP切分；router通常复制。当前vLLM通用K3路径中的latent down/up projection标为replicated，而官方生产kernel设计会进一步做跨rank分片和fused AllGather，二者需按具体backend区分。

### TP8 + EP8

```text
Attention仍按TP8切
896 routed experts按EP8放置
每rank约112个experts
Token通过All-to-All去专家所在rank
```

这时不是“每rank持有896个expert的1/8”，而是“每rank持有约112个完整/本地expert权重”，具体expert内部是否再TP取决于框架的EP/ETP配置。

两者每rank的expert权重总量同样约为全局1/8，但通信形态不同：

- 纯TP：矩阵partial结果做TP collective；
- EP：token dispatch/combine做All-to-All。

### DP + EP

Attention权重在DP replicas间复制，每个请求只落到一个Attention replica；experts跨EP group分布。它适合高并发，但K3 replicated dense/attention/shared权重和runtime开销大，因此vLLM recipe把DEP最低规模设为16 GPUs，而TP/TEP可从8 GPUs起步。

## TP8每rank的结构直觉

```text
KDA/Q heads：12/96
Hidden shard：7168/8 = 896
Latent shard（若采用分片优化）：3584/8 = 448
Expert intermediate shard（纯TP）：3072/8 = 384
Routed expert storage：全局约1/8
Gated MLA latent KV：当前复制完整cache
Router/部分low-rank projection：复制
```

所以TP8不是所有权重严格除以8；更准确是：

```text
大部分大矩阵/experts被切分
少量router、norm、low-rank projection和当前MLA cache复制
```

## 误差与边界

- `2.78T`和`104.2B`来自官方技术报告；上述模块拆分是根据公开维度推导，不是checkpoint逐tensor求和。
- 总参数剩余39.8B包含Attention之外的embedding、vision、dense首层等，不能当成纯Attention精确占比。
- pure TP、TP+EP、MegaMoE和专用latent-tail fusion的weight layout不同，应绑定vLLM commit与启动参数。
- MXFP4实际显存还包括scale、packing、alignment、metadata以及高精度非expert权重。

## 官方依据

- [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)
- [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)
- [vLLM Kimi K3 recipe](https://github.com/vllm-project/recipes/blob/72626067968e70856b79a2e4841edea5d6846012/models/moonshotai/Kimi-K3.yaml)
- [vLLM K3 model](https://github.com/vllm-project/vllm/blob/1ad5182ba95a6f1de23b537d57b860082912b28e/vllm/models/kimi_k3/nvidia/model.py)
- [vLLM K3 MLA](https://github.com/vllm-project/vllm/blob/1ad5182ba95a6f1de23b537d57b860082912b28e/vllm/models/kimi_k3/nvidia/mla.py)
