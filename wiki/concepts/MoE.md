---
type: concept
topic: 模型架构
sources: 6
updated: 2026-06-12
---

# MoE

## 定义

`MoE`（Mixture of Experts）把 Transformer 中常规 dense FFN 替换为多个 expert FFN，并由 router 为每个 token 选择少数 expert 执行，从而把总参数量和每 token 激活计算量部分解耦。

## 它解决什么问题

- 在不让每个 token 经过全部参数的前提下扩大模型容量
- 让不同 expert 学到不同 token / 语义 / 任务模式
- 在多卡系统中配合 [[Expert Parallelism]] 把 expert 权重分散到不同设备

## 算子计算流程

设输入 hidden states 为 `X: [B, S, D]`，展平后得到 `T = B * S` 个 token，`X_flat: [T, D]`。expert 数为 `E`，每个 token 选 `K` 个 expert。

1. `Router projection`
   - 对每个 token 做一次线性层：`logits = X_flat @ W_r`，shape 为 `[T, E]`。
   - `W_r` 通常很小，参数量约为 `D * E`，但它决定后续 token 分发路径。

2. `Softmax / score`
   - 对 expert 维度做 softmax：`p = softmax(logits)`。
   - 有些实现会在 router 前后加入 norm、scale、噪声、温度或 expert bias，用于稳定训练和负载均衡。

3. `Top-k routing`
   - 对每个 token 选出概率最高的 `K` 个 expert：`topk_idx: [T, K]`，`topk_weight: [T, K]`。
   - `top-1` 只走一个 expert；`top-2` / `top-k` 会让同一个 token 被复制到多个 expert，最后再加权合并。
   - 常见实现会把 `topk_weight` 重新归一化，使被选中的 `K` 个权重和为 `1`。

4. `Capacity / load balancing`
   - 训练时常设置每个 expert 的 capacity，避免热门 expert 接收无限多 token。
   - 超过 capacity 的 token 可能被 drop、fallback 到其它 expert，或通过更复杂的 balanced routing 重分配。
   - 负载均衡 loss / importance loss 会鼓励不同 expert 的 token 数和概率质量更均匀。

5. `Token dispatch`
   - 根据 `topk_idx` 把 token 按 expert 分桶：同一个 expert 的 token 被打包成局部 batch。
   - 单卡上这主要是 gather / scatter / sort / index select；多卡 `Expert Parallelism` 下，token 还要跨 GPU 发送到 expert 所在设备，通信模式常接近 `all-to-all`。
   - 这是 MoE 的关键系统成本之一，因为 token 路由不规则，batch size 可能很碎。

6. `Expert FFN`
   - 每个 expert 对自己的 token batch 执行一套 FFN，典型形式为 `W_down(activation(W_up x))`，也可能是 SwiGLU / gated MLP。
   - 如果每个 expert 的隐藏维度是 `Hff`，单 expert 参数粗略为 `2 * D * Hff`；总 expert 参数为 `E * 2 * D * Hff`，但每 token 只激活 `K` 份。

7. `Combine / gather back`
   - expert 输出按原 token 顺序 scatter 回 `Y: [T, D]`。
   - 对 `top-k` 路由，同一个 token 来自多个 expert 的输出按 gate 权重求和：

```text
Y[t] = sum_{i=1..K} topk_weight[t, i] * Expert_{topk_idx[t, i]}(X[t])
```

8. `Reshape and residual`
   - 把 `Y` reshape 回 `[B, S, D]`。
   - 在 Transformer block 中，它通常走 FFN 残差路径：`hidden = hidden + MoE(norm(hidden))`。

## Qwen3.5-MoE 形状口径

以 `Qwen3_5MoeForConditionalGeneration` 的 MoE block 为例，文本侧 `hidden_size` 是语言模型主干每个 token 的向量宽度；decoder layer 输入输出、attention 输出、MoE 输入输出、最终 `lm_head` 输入都围绕 `[B, S, hidden_size]`。

给定一个文本侧 `hidden_size=4096`、`num_experts=512`、`num_experts_per_tok=10`、`moe_intermediate_size=1024`、`shared_expert_intermediate_size=1024` 的 config，单层 MoE 的计算可以按下面理解：

- Router 输入：`X_flat: [T, 4096]`
- Router logits：`[T, 512]`
- Top-k 结果：`selected_experts: [T, 10]`、`routing_weights: [T, 10]`
- 每个 routed expert 是一个 SwiGLU FFN：`4096 -> 2 * 1024 -> 4096`
- shared expert 也是一个始终计算的 SwiGLU FFN：`4096 -> 1024 -> 4096`
- MoE block 输出：`Y: [B, S, 4096]`

因此这个 config 中每个 token 激活的是 `10` 个路由专家加 `1` 个 shared expert，而不是把 `512` 个专家全部计算一遍。`moe_intermediate_size` 只控制单个 routed expert 内部 FFN 的扩展宽度；它不是主干 hidden 维度，也不是所有专家拼起来后的维度。

视觉侧要单独看：`vision_config.hidden_size=1152` 是视觉 encoder 内部 patch token 的宽度；`vision_config.intermediate_size=4304` 是视觉 MLP 的中间宽度；`vision_config.out_hidden_size=4096` 才是视觉特征 merge 后对齐到语言模型 hidden space 的宽度。

### TP8 下的 MoE shape

下面只讨论 tensor parallel，不讨论 expert parallel。也就是说，每个 TP rank 负责同一批 expert 权重的一个张量切片，而不是只放一部分 expert。

设 `TP=8`，`T=B*S`。若不启用 sequence parallel，MoE 输入通常在每个 TP rank 上都是完整 token hidden：

```text
X_flat: [T, 4096]
```

router 通常复制在每个 TP rank：

```text
router_weight:  [512, 4096]
router_logits:  [T, 512]
topk_idx:       [T, 10]
topk_weight:    [T, 10]
```

routed expert 的完整权重为：

```text
gate_up_proj: [512, 2048, 4096]   # 512 experts, 2 * 1024 intermediate, 4096 input
down_proj:    [512, 4096, 1024]
```

TP8 后，每个 rank 上的专家权重切片为：

```text
gate_up_proj_local: [512, 256, 4096]   # 2 * (1024 / 8)
down_proj_local:    [512, 4096, 128]   # 1024 / 8
```

对某个 expert `e`，假设它在本 step 收到 `n_e` 个 token：

```text
X_e:              [n_e, 4096]
gate_up_local:    [n_e, 256]
gate_local:       [n_e, 128]
up_local:         [n_e, 128]
ffn_mid_local:    [n_e, 128]
down_partial:     [n_e, 4096]
```

`down_partial` 需要在 TP group 内 all-reduce，得到该 expert 对这些 token 的完整输出：

```text
expert_out_e: [n_e, 4096]
```

shared expert 的完整权重为：

```text
gate_proj: [1024, 4096]
up_proj:   [1024, 4096]
down_proj: [4096, 1024]
```

TP8 后每个 rank：

```text
gate_proj_local: [128, 4096]
up_proj_local:   [128, 4096]
down_proj_local: [4096, 128]
```

shared expert 的本地中间激活是 `[T, 128]`，down 后得到 `[T, 4096]` partial，再 all-reduce 成 `[T, 4096]`。最终：

```text
Y = routed_expert_out + shared_expert_out
Y: [T, 4096] -> [B, S, 4096]
```

如果启用 sequence parallel，上述权重切片不变，但 `T` 会变成本 rank 负责的 `T_local`；实现还需要在 layernorm、router、dispatch 或专家计算前后处理 token 维的 gather / scatter。

## 算子视角的性能形态

- `router` 是小矩阵乘 + top-k，通常不是 FLOPs 大头，但会引入动态控制流。
- `dispatch / gather` 是 gather-scatter 类算子，常受间接寻址、访存随机性、排序/分桶和跨卡通信影响。
- `expert FFN` 本质仍是 GEMM，但每个 expert 的局部 batch 可能很小；如果 batch 被切碎，Tensor Core 利用率会下降。
- 多卡 MoE 的瓶颈经常不是单个 expert GEMM，而是 `all-to-all`、负载不均和 dispatch buffer 峰值。
- 推理 decode 阶段每 step token 少，expert batch 更碎；prefill 阶段 token 多，更容易把 expert GEMM 做大，但 dispatch 通信量也更大。

## 单卡瓶颈

只考虑单卡时，MoE 没有跨 GPU `all-to-all`，但仍然不是一个普通 dense MLP。主要瓶颈会落在三类地方：

- `dispatch / permutation`：router 选完 expert 后，需要按 expert 把 token 重排成连续 buffer。这里是 gather / scatter / sort / prefix-sum / index copy 一类操作，容易受随机访存、额外读写和 launch 开销影响。
- `small / irregular GEMM`：每个 expert 收到的 token 数 `n_e` 不一样，真正要算的是一组 `X_e [n_e, D] @ W_e [D, H]`。如果 `n_e` 很小，单个 expert GEMM 很难喂满 Tensor Core，逐 expert 调 cuBLAS 还会有大量 kernel launch 开销。
- `load imbalance / tail effect`：热门 expert 的 `n_e` 大，冷门 expert 的 `n_e` 小。即使在单卡内，没有网络通信，kernel 内不同 group 的工作量也不均匀，容易出现尾部等待。

因此单卡 MoE 的瓶颈要分场景看：prefill token 多时，主要看 expert GEMM 能否被 grouped GEMM 做大做满；decode token 少时，top-k、dispatch/gather、权重读取和小 GEMM launch overhead 往往更显眼。

## Grouped Matmul 与 Batched Matmul

`grouped matmul` / `grouped GEMM` 和 `batched matmul` 都发生在 `Expert FFN` 阶段，而不是 router 阶段。router 阶段负责产生 `topk_idx` 和 `topk_weight`；dispatch 阶段负责把 token 按 expert 分桶；分桶之后才进入 expert 的矩阵乘。

对每个 expert `e`，第一层 expert FFN 形如：

```text
Y_e = X_e @ W_up_e
X_e:    [n_e, D]
W_up_e: [D, Hff]
Y_e:    [n_e, Hff]
```

第二层再做：

```text
O_e = act(Y_e) @ W_down_e
W_down_e: [Hff, D]
O_e:      [n_e, D]
```

`batched matmul` 更适合一批 shape 相同的矩阵乘，例如把 token buffer padding 成 `X: [E, C, D]`，权重是 `W: [E, D, Hff]`，然后做：

```text
out = bmm(X, W)  # [E, C, Hff]
```

这里 `C` 是每个 expert 的固定 capacity。优点是抽象简单，缺点是要为 padding token 做无效计算；当 expert token 数差异大时，浪费会很明显。

`grouped matmul` 更适合 MoE 的真实形态：每个 expert 的 `n_e` 不同，但 `D / Hff` 通常相同。它把多组不同 `M=n_e` 的 GEMM 描述给同一个 grouped GEMM kernel：

```text
for e in experts:
    C_e[n_e, Hff] = A_e[n_e, D] @ B_e[D, Hff]
```

实现上不是 Python 循环逐个调 matmul，而是在一个或少量 kernel 中调度多组 GEMM tile。这样可以减少 launch 开销，避免 padding 到统一 capacity，并让不同 expert 的 tile 在 GPU 上混合调度，提高 SM 和 Tensor Core 利用率。

所以高性能单卡 MoE 通常更倾向使用 `grouped matmul / grouped GEMM` 来算 expert FFN；`batched matmul` 主要出现在 padding 到固定 capacity 的简单实现、训练参考实现，或 expert token 数较均匀且 padding 浪费可接受的场景。

## Rubin 的 MoE 执行链优化

[[../entities/NVIDIA Rubin]] 的相关机制可以串成一条动态 expert 执行链：

1. Router 选中 expert 后，复用公共 TensorMap，并通过 TMA runtime override 覆盖 global base address/dimensions/strides，避免为每个同 shape/layout 的 expert 保存或频繁 patch 独立 descriptor。
2. Expert 权重加载后以 `evict_last` 倾向留在 L2，供多个 token/tile 重复使用；last-use 后用 `applypriority` 恢复 `evict_normal`，把 cache capacity 让给下一个热点 expert。
3. Tensor Core 执行 Grouped GEMM，增强 SFU 处理 SwiGLU activation/epilogue，降低“GEMM 已完成但 epilogue 未跟上”的空泡风险。
4. 跨 GPU dispatch/combine 可结合 counted put/reduction，让接收端按已访问字节数判断数据 ready，减少额外 barrier、ack 和 atomic flag。

这些机制优化权重寻址、搬运、缓存生命周期、epilogue 与同步，不减少 MoE 数学 FLOPs，也不保证理论子系统提升能完全转化为端到端收益。

## NCCL EP 中的 dispatch/combine

[[../entities/NCCL Extensions]] 的 `nccl_ep` 将 MoE token 的 dispatch/combine 下沉为带模型结构语义的通信组件。与通用 All-to-All 相比，它直接表达 top-k 路由、expert/rank 布局、接收布局及低延迟/高吞吐算法模式；但 Router 决策、Grouped GEMM 和完整 serving 生命周期仍由上层系统负责。

## AFD 中的 MoE 服务化

[[Attention-FFN 分离]] 可以把 MoE/FFN 路径从维护请求状态和 KV Cache 的 Attention worker 中拆出。每个切分层由 Attention 侧发送 hidden states、`layer_id` 和执行元数据，FFN 侧完成 Router、token dispatch、expert compute 与 combine 后返回 FFN output。FFN 侧仍可内部使用 [[Expert Parallelism]]；此时要区分 A/F 服务间的激活往返与 FFN ranks 内部的 EP All-to-All。

这种分离不改变 `Attention[l] -> FFN[l] -> Attention[l+1]` 的模型依赖，也不保证自动提速。收益来自 Attention 与专家容量的独立配比及不同 ubatch 的通信—计算重叠，代价是每个切分层的双向激活通信。

## 面试口径

一句话：`MoE` 算子不是“把很多 MLP 全算一遍”，而是 `router -> top-k -> token dispatch -> expert FFN -> weighted gather` 这一条稀疏执行链。

更系统地说：MoE 用 router 把每个 token 分给少数 expert，只计算被选中的 FFN，再按 gate 权重把结果合并回原 token 位置；它省的是 active FLOPs，增加的是路由、搬运、负载均衡和碎片化 batch 的系统复杂度。

## 关键权衡

- 优点：总参数可扩展，active params / FLOPs 控制在较低水平。
- 代价：路由不稳定、expert 负载不均、token dispatch/gather 复杂。
- 工程风险：跨卡 `all-to-all`、局部 batch 太小、capacity 溢出、dispatch buffer 和中间激活导致显存峰值上升。
- 训练风险：router 早期塌缩、少数 expert 变成热门 expert，低利用 expert 学不到东西。

## 相关概念

- [[CUDA Kernel]]
- [[Tensor Parallelism]]
- [[Expert Parallelism]]
- [[DP Attention]]
- [[Sparsity Allocation]]
- [[Warp Divergence]]
- [[Attention-FFN 分离]]
- [[通信-计算重叠]]
- [[CUDA内存层次]]
- [[算子融合]]

## 相关实体

- [[../entities/DeepSeek-AI]]
- [[../entities/Gemma 4]]
- [[../entities/vLLM AFD Plugin]]
- [[../entities/NCCL Extensions]]
- [[../entities/NVIDIA Rubin]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构]]
- [[../sources/Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models]]
- [[../sources/vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署]]
- [[../sources/NVIDIA 开源 NCCL Extensions：把 MoE 专家路由与跨 Mesh 权重重分片推进到 GPU 设备侧]]
- [[../sources/Nvidia Rubin架构分析预览]]

## 研究备注

- 后续可继续把 MoE 中的 token dispatch / gather、EPLB、all-to-all overlap 和 serving 并行拓扑拆开到更具体的实现页。
- 不同模型的 router 细节差异很大，例如是否有 shared expert、expert bias、aux-loss-free load balancing、grouped top-k 或 capacity 限制，具体实现应按模型源码核实。
