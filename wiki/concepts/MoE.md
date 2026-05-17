# MoE

## 定义

`MoE`（Mixture of Experts）把 Transformer 中常规 dense FFN 替换为多个 expert FFN，并由 router 为每个 token 选择少数 expert 执行，从而把总参数量和每 token 激活计算量部分解耦。

## 它解决什么问题

- 在不让每个 token 经过全部参数的前提下扩大模型容量
- 让不同 expert 学到不同 token / 语义 / 任务模式
- 在多卡系统中配合 `Expert Parallelism` 把 expert 权重分散到不同设备

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
- [[DP Attention]]
- [[Sparsity Allocation]]
- [[Warp Divergence]]

## 相关实体

- [[../entities/DeepSeek-AI]]
- [[../entities/Gemma 4]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构]]
- [[../sources/Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models]]

## 研究备注

- 后续可继续补独立的 `Expert Parallelism` 页面，把 MoE 中的 token dispatch / gather、EPLB、all-to-all overlap 和 serving 并行拓扑拆开。
- 不同模型的 router 细节差异很大，例如是否有 shared expert、expert bias、aux-loss-free load balancing、grouped top-k 或 capacity 限制，具体实现应按模型源码核实。
