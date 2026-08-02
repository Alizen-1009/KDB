# MoonEP 动态冗余 Expert 机制

> MoonEP 是 Kimi K3 训练系统中的 Expert Parallel方案。它不改变Router选中的expert ID，而是动态复制热点expert，把同一expert的部分token安排到其他rank上的临时副本执行，从而让每个EP rank承担完全相同数量的token–expert assignments。

相关页面：[[../../wiki/entities/MoonEP|MoonEP]]、[[../../wiki/concepts/Expert Parallelism|Expert Parallelism]]、[[../../wiki/entities/Kimi K3|Kimi K3]]。

---

## 1. 它解决什么问题

设：

```text
R：EP ranks数量
E：routed experts数量
S：每个rank的输入token数
K：每token选择的experts数量
```

全局共有：

$$
R\times S\times K
$$

个token–expert assignments。理想情况下每个rank执行：

$$
\frac{RSK}{R}=SK
$$

个assignments。

但Router负载通常不均匀。若热点experts集中在rank 0：

```text
rank 0：需要执行很多tokens
rank 1：中等
rank 2：很少
rank 3：几乎空闲
```

MoE层完成时间由最慢rank决定：

$$
T_{MoE}\approx\max_r T_r
$$

所以其他ranks即使空闲，也要等待热点rank。

---

## 2. 动态冗余的核心思想

常规EP中，每个expert有固定home rank：

```text
rank 0：E0, E1
rank 1：E2, E3
rank 2：E4, E5
rank 3：E6, E7
```

假设Router大量选择 `E0`。传统执行会把所有`E0` tokens发给rank 0。

MoonEP可以临时把`E0`复制到空闲rank：

```text
rank 0：E0 home copy，处理一部分E0 tokens
rank 3：E0 redundant copy，处理另一部分E0 tokens
```

关键是：

```text
模型语义没有改变
token仍然执行Router选中的E0
只是E0在哪张卡执行发生变化
```

这与“把token改路由到E7”完全不同。

---

## 3. Forward执行流程

每个micro-batch、每一层动态规划：

```text
1. Router产生当前micro-batch的Top-K expert IDs
2. 统计各expert/rank的token负载
3. GPU planner生成balanced placement plan
4. 把需要的hot expert weights预取到redundant slots
5. 根据plan直接把tokens发送到home或redundant copy
6. 各rank以完全相同的总assignment数执行Expert GEMM
7. 将Expert输出发回原token位置并combine
```

伪代码：

```python
expert_ids, routing_weights = router(x)       # [tokens,K]
loads = count_assignments(expert_ids)

plan = moon_ep_plan(
    expert_ids=expert_ids,
    home_placement=expert_to_home_rank,
    target_assignments_per_rank=S * K,
)

prefetch_redundant_expert_weights(plan)

dispatched = fused_permute_and_send(
    x,
    expert_ids,
    plan.destination_rank,
    plan.destination_offset,
)

expert_output = grouped_expert_gemm(dispatched, plan.local_expert_copies)
y = receive_and_unpermute(expert_output)
```

Planner使用当前micro-batch和当前层的真实Router结果，因此复制集合会随layer和step变化。

---

## 4. 一个简化例子

设：

```text
R = 4 ranks
E = 8 experts
每rank固定拥有2个home experts
全局assignment数 = 32
目标 = 8 assignments/rank
```

Router按home rank聚合后的初始负载：

```text
rank 0：14
rank 1：10
rank 2： 6
rank 3： 2
```

一种平衡计划可能是：

```text
从rank 0迁移6个assignment到rank 3
  → rank 3临时复制这些assignment对应的rank-0 experts

从rank 1迁移2个assignment到rank 2
  → rank 2临时复制对应的rank-1 expert
```

结果：

```text
rank 0：8
rank 1：8
rank 2：8
rank 3：8
```

这里“迁移assignment”表示：token仍执行原来的expert，只是转到该expert的副本。

---

## 5. 为什么最多预留 E/R 个冗余槽就够

每个rank本来拥有：

$$
E/R
$$

个home experts。

MoonEP报告给出一个构造性证明：

1. 将ranks分成overloaded与underloaded；
2. 每次选一个underloaded rank和一个overloaded rank；
3. 从overloaded rank迁移足够的assignments，恰好填满underloaded rank到目标`SK`；
4. 一个underloaded rank被填满后不再改变；
5. 最多进行`R-1`次填充。

该构造保证：

> 每个被填充rank的remote assignments只来自一个source rank。

一个source rank最多只有`E/R`个home experts，因此目标rank最多需要复制：

$$
E/R
$$

个不同experts。

所以每rank预留`E/R`个redundant-expert slots，即可保证对任意Router输出总存在perfectly balanced plan：

$$
M(I)\le E/R
$$

注意这是最坏情况容量上界，不代表每一步都真的复制`E/R`个experts。

---

## 6. 为什么这个上界基本无法继续缩小

报告构造的最坏情况是：

```text
rank 0的home experts没有收到任何token
其余R-1个ranks的experts均匀承担全部tokens
```

rank 0必须从其他ranks接收`SK`个assignments。由于每个expert能提供的tokens有限，它至少需要复制：

$$
\left\lceil\frac{E(R-1)}{R^2}\right\rceil
$$

个不同experts。

当`R`较大时：

$$
\frac{E(R-1)}{R^2}\approx E/R
$$

因此一般情况下不存在显著小于`E/R`的统一最坏情况上界。

---

## 7. Backward如何保证参数语义正确

一个expert可能同时在home与多个redundant copies上处理tokens，因此每个副本都会产生局部weight gradient。

MoonEP执行：

```text
1. redundant copy将梯度写入本地reduce buffer
2. Expert计算完成后
3. 把所有副本梯度Reduce回home rank的正式gradient buffer
4. Optimizer只更新逻辑上的一份Expert参数
```

概念伪代码：

```python
for expert_id in redundant_experts:
    local_grad = redundant_grad_buffer[expert_id]
    reduce_to_home_rank(
        local_grad,
        dst=home_rank[expert_id],
    )

home_grad[expert_id] += reduced_redundant_grads
```

因此动态冗余不产生多个独立学习的experts；它们是同一个逻辑expert的临时计算副本。

---

## 8. Perfect Balance带来的系统收益

### 8.1 消除Rank级Straggler

每个rank严格执行`SK`个assignments：

```text
总工作量相同
→ Rank级MoE tail latency显著降低
```

### 8.2 固定通信Buffer

Planner提前知道每个token的最终destination和offset，可以直接发送到远端expert-grouped位置。

MoonEP所需buffer固定为：

$$
S\times K
$$

报告对比称，传统DeepEP要在最坏不均衡下维持同样copy-free路径，buffer可能需要：

$$
S\times K\times R
$$

### 8.3 Zero-copy Permute/Unpermute

Planner预计算：

```text
destination rank
destination expert
destination offset
```

token直接落到远端grouped-GEMM输入位置，通信buffer本身可直接作为计算view，减少中间permute copy。

### 8.4 Static Shape与Sync-free Launch

传统EP每层需要知道各rank实际收到多少tokens，host可能必须等待device统计结果后才能确定GEMM shapes。

MoonEP中每rank总assignment数固定为`SK`：

```text
rank-level buffer shape静态
→ 不需要每层device-to-host同步总负载
→ 降低host launch gap与内存碎片
```

---

## 9. 完美Rank均衡不等于Expert内部均衡

即使每rank总token数完全一致，rank内部仍可能是：

```text
expert A：很多tokens
expert B：很少tokens
expert C：中等
```

因此MoonEP还需要workload-aware expert-GEMM scheduler：

- 根据当前local per-expert token counts调整SM worker schedule；
- 使用硬件成本模型选择参数；
- 参数在kernel launch后保持固定；
- shared-expert GEMM放到独立stream，与其他kernel重叠。

所以优化分两层：

```text
MoonEP planner：解决rank间不均衡
GEMM scheduler：解决rank内expert workload skew
```

---

## 10. MoonEP与Quantile Balancing的区别

| 机制 | Quantile Balancing | MoonEP |
| --- | --- | --- |
| 所在层次 | 模型Router训练 | 分布式执行系统 |
| 是否改变Top-K expert选择 | 会，通过bias影响选择 | 不会 |
| 是否改变mixture weights | 不会，bias不进入权重 | 不会 |
| 作用时间 | 训练时更新bias，推理时冻结 | 每个训练micro-batch/layer动态规划 |
| 主要目标 | 长期让experts都获得合理训练量 | 当前step让EP ranks完全等工作量 |
| 核心手段 | Router score quantile | 复制hot experts并迁移执行assignment |

两者互补：QB降低长期路由偏斜和dying experts；MoonEP处理当前micro-batch仍然存在的瞬时不均衡。

---

## 11. MoonEP 与 EPLB 的关系

两者属于同一技术家族：

```text
Logical Expert只有一个语义身份
Physical Expert可以有多个副本
根据负载复制hot experts并调整placement
```

但以DeepSeek/vLLM EPLB为对比，它们的优化时间尺度和目标不同。

### vLLM EPLB

```text
每个forward收集expert load statistics
→ 在window内聚合历史负载
→ 每隔若干engine steps重新计算physical-to-logical mapping
→ 异步迁移expert weights
```

当前vLLM默认配置示例是：

```text
window_size = 1000 engine steps
step_interval = 3000 engine steps
num_redundant_experts = 用户配置
```

EPLB根据一段历史中的expected load，选择应有更多physical replicas的logical experts，并把物理副本打包到nodes/GPUs，使预计负载尽量均匀。它是启发式长期重布局，不保证下一个micro-batch的每rank token数严格相等。

### MoonEP

```text
读取当前layer、当前micro-batch的实际Router输出
→ 当场规划每个assignment去home还是redundant copy
→ 目标是当前step每rank严格S×K assignments
```

MoonEP还需要处理训练Backward：副本梯度Reduce回home expert。

### 关键对比

| 维度 | DeepSeek/vLLM EPLB | MoonEP |
| --- | --- | --- |
| 共同机制 | hot expert replication + load-aware placement | hot expert replication + load-aware placement |
| 统计输入 | 一段历史window的expert load | 当前layer/micro-batch的真实routes |
| 调整频率 | 周期性，数千engine steps量级可配置 | 每layer、每micro-batch |
| 目标 | 让预计/平均负载尽量均衡 | 当前step严格perfect balance |
| 冗余容量 | `num_redundant_experts`配置，可能不足 | 每rank预留`E/R`可保证可行 |
| 算法性质 | greedy replication + hierarchical packing | online assignment planner +上界证明 |
| 主要场景 | vLLM在线Serving | K3大规模训练 |
| Backward | Serving无Backward | 副本梯度需Reduce回home |
| Shape | 当前请求负载仍可能波动 | rank总assignment shape固定为`S×K` |

因此可以把MoonEP理解成：

> **EPLB思想的current-batch、assignment-level、perfect-balance训练版本。**

它不是完全不同的方向，但比周期性EPLB提出了更强的当前step平衡目标和可行性保证。

## 12. MoonEP与Serving EP的边界

Kimi K3报告把MoonEP放在`3T-class Pre-Training`章节，描述了forward weight prefetch、backward gradient reduce和静态训练shape。不能据此直接断言K3在线Decode也使用相同的动态冗余策略。

Serving小batch下复制大型expert weights可能得不偿失：

- weight prefetch成本高；
- Decode更常受weight bandwidth限制；
- 请求持续到达，routing shape变化快；
- GPU需要为冗余slots预留显存。

K3 Serving章节采用的是专门的LatentMoE decode kernel、token-centric WarpDecode与EP通信优化。MoonEP最明确的适用场景是大规模训练；是否用于某个Serving backend需单独核对实现。

---

## 13. 代价与适用条件

MoonEP用以下成本换取perfect balance：

```text
预留redundant expert显存槽
动态权重prefetch/P2P传输
GPU在线planner
Backward副本梯度Reduce
更复杂的placement与一致性管理
```

它更适合：

- expert数量极多；
- Top-K较大；
- global batch/micro-batch足够大；
- EP rank straggler明显；
- 设备间有高带宽互联；
- 训练吞吐足以摊薄动态prefetch与planning开销。

---

## 14. 一句话总结

> MoonEP不让token“换Expert”，而是让Expert“临时换位置”：根据当前micro-batch的真实路由负载复制热点experts，把原expert的部分token assignment迁移到空闲ranks执行，并在Backward把副本梯度Reduce回home expert。每rank预留`E/R`个副本槽即可保证任意路由下存在完全均衡方案，从而获得固定`S×K`通信buffer、静态rank级shape和无straggler的EP执行。

## 官方来源

- [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)，§5.2.1与Appendix E
- [MoonEP](https://github.com/MoonshotAI/MoonEP)

## 待核实

- 报告未在正文公开GPU near-optimal planner的完整启发式与权重prefetch overlap细节，需结合MoonEP源码版本分析。
- `E/R`是最坏情况下需要预留的副本槽上界，不是平均每步复制数量或额外永久expert数量。
- MoonEP在报告中是训练系统；在线Serving是否启用动态冗余需按具体backend确认。
