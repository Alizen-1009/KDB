# Kimi K3 技术报告后续阅读重点

> 面向模型架构与AI Infra，而不是逐章复述。已掌握KDA计算、TP/DP/EP与Prefix Cache后，下面按收益排序。

相关实体：[[../../wiki/entities/Kimi K3|Kimi K3]]。

## 第一优先级：Stable LatentMoE 的算法—训练—部署闭环

建议连读：

```text
§2.3 Stable LatentMoE
§5.2.1 Perfectly Balanced Expert-Parallel MoE Training
§5.4.2 Stable LatentMoE serving kernel
```

这是最值得继续看的部分，因为K3约98%的总参数位于MoE相关权重，且每token激活16/896 routed experts和2个shared experts。

关键问题：

1. 为什么先把hidden `d=7168`降到latent `l=3584`再dispatch？
2. 为什么shared experts保持full-width，而routed experts工作在latent space？
3. RMSNorm为什么放在expert aggregate与up-projection之间？
4. SiTU-GLU如何同时soft-cap gate/up两支，避免四段GEMM链中的激活爆炸？
5. Quantile Balancing如何在896 experts下控制训练路由负载？
6. MoonEP为什么需要动态冗余experts，而不是只调router bias？
7. Decode如何融合latent down-projection与router，并把output AllGather放进GEMM epilogue？

这部分能直接解释：

```text
为什么K3适合EP
为什么纯EP仍有latent projection通信
为什么专家权重是MXFP4而shared/dense路径保持高精度
为什么训练负载均衡与Serving Top-k路由要分开讨论
```

## 第二优先级：Quantile Balancing 与 MoonEP 的分工

这两个机制经常被混为一谈：

```text
Quantile Balancing：模型/训练路由层
调expert-specific bias，让下一batch的选择趋向目标负载
最终bias在inference冻结

MoonEP：分布式执行层
根据当前micro-batch真实router outputs，动态复制/迁移hot experts
保证每个EP rank收到完全相同总token数
```

QB只改变“token想去哪个expert”；MoonEP解决“这些expert如何映射到ranks并均衡执行”。

MoonEP值得推导的点：

- 每rank最多预留`E/R`个redundant-expert slots即可保证存在perfectly balanced plan；
- perfect balance令通信buffer固定为`S×K`，不需要worst-case `S×K×R`；
- rank workload固定后，expert GEMM shape可静态化，去掉每层device-to-host同步；
- expert内部token数仍不均衡，所以还需要workload-aware GEMM scheduler。

## 第三优先级：Deployment-Aware Post-Training 与投机解码

建议读§4.1.4并与§5.4.2 KDA Decode对照。

### MXFP4 QAT

```text
MXFP4：routed expert weights
MXFP8：expert input activations
更高精度：attention、latent projections、shared experts、router
```

QAT贯穿SFT和RL，使rollout与训练量化配置一致。它解释了为什么部署显存不能简单按“2.78T×0.5 byte”覆盖所有权重。

### MTP到EAGLE-3 Draft

- 使用预训练MTP layer初始化draft；
- 从第1、第4和最后一个AttnRes block提取低/中/高层feature；
- draft训练unroll 7步；
- 用直接对应acceptance rate的LK loss，而不只优化KL；
- KDA验证回滚不保存每draft位置的大state，而缓存projected inputs后Replay。

这是“训练目标如何为Serving SLO服务”的完整案例。

## 第四优先级：KDA Context Parallelism 的仿射扫描

如果要继续深挖KDA Kernel/分布式，这是最值得推公式的部分。

普通线性注意力segment可只传local state并求和；KDA segment必须表示：

$$
S_{out}=M_{segment}S_{in}+\tilde S_{segment}
$$

需要理解：

- 为什么Delta correction使segment不再是简单加法；
- 两个仿射transition如何结合；
- 为什么结合满足associativity，可做prefix scan；
- 为什么KCP通信量固定，但`M[d_k,d_k]`仍可能很重；
- 单卡SM-level CP与跨卡KCP的边界。

## 第五优先级：AttnRes 的部署影响，而非模型公式

如果只关心Infra，不必完整推导depth attention；重点看：

- Full AttnRes保存所有层输出为`O(Ld)`；
- Block AttnRes将其降到`O(Nd)`，K3使用8个12-layer blocks，加embedding共9个block representations；
- Prefill通过TP ReduceScatter → sequence-sharded AttnRes → AllGather，避免每rank重复物化；
- Decode把inter-block kernel放side stream，与主stream计算重叠；
- intra-block merge + RMSNorm融合进TP AllReduce。

它是K3除KDA/MLA Cache外的第三类inference-time state/I/O问题。

## 第六优先级：NoPE 与1M Context到底意味着什么

K3的设计是：

```text
KDA：用递归decay/gating提供位置敏感与recency
周期性NoPE MLA：提供不受压缩状态限制的全局content interaction
```

因此扩展上下文不需要RoPE rescaling/YaRN，但这不等于“无需长上下文训练”。报告仍采用：

```text
8K → 64K（pre-training）
256K → 1M（cooldown）
```

并合成长距离依赖数据，避免模型只学到局部模式。值得区分：

```text
位置编码外推
≠ 长程检索能力
≠ 百万token Prefill可负担性
```

## 第七优先级：3T训练内存与流水线

若关注训练系统，再看§5.2.2：

- 统一activation manager，把recompute/quantize/offload作为tensor级storage policy；
- block-wise FP8 activation + local/remote offload；
- MoE backward重写，减少必须保存的forward tensor；
- PP rank间远程offload，利用后段rank较少的activation驻留空间；
- Pipeline ZeRO-2 gradient shard + CPU offload；
- Muon不做全参数AllGather，而用P2P只拉取本rank负责更新的parameter shards。

这部分更偏训练，不直接决定在线Serving拓扑。

## 第八优先级：Fleet级Serving

Prefix Cache之后值得补：

- cache-aware affinity：session优先路由到持有长prefix的cluster；
- primary/secondary consistent hashing：secondary不预存cache，故障后重Prefill压力分散到fleet；
- budget-based admission：避免百万token请求突发拖垮短请求SLO；
- CPU external cache采用write-back而非write-through，只有GPU驱逐的idle prefixes才占CPU带宽。

## 低优先级或需谨慎阅读

- Benchmark与cost曲线：可用于定位能力，但属于模型方自报，不能直接外推到不同Serving配置。
- Native Vision：若当前关注纯文本Infra，可后置；若关注多模态训练，则重点是dynamic CP与将ViT计算塞入PP bubbles。
- Agentic RL数据与环境：更偏训练方法，除非关注partial rollout、sandbox checkpoint和长轨迹状态管理。

## 推荐下一步

最顺畅的路线是：

```text
Stable LatentMoE计算公式
→ Quantile Balancing
→ MoonEP动态冗余Expert
→ LatentMoE Decode Kernel
→ TP/EP/DP部署账本
```

它会直接接上当前问题：“为什么K3有时TP8，有时又用Attention DP + Wide EP？”

## 来源

- [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)，§2.3、§4.1.4、§5.1–5.4
