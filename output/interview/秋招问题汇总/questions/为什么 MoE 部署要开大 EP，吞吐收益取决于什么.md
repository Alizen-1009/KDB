# 为什么 MoE 部署要开大 EP，吞吐收益取决于什么？

- 整理状态：已整理
- 题目出处：用户提供的 2026-10-10 面试回忆；答案根据本次对话整理

## 其他问法

- 为什么要大 EP、大 DP？大 EP 开多大合适？固定单卡的 batch size 为 16，以 DeepSeek 为例，计算 EP 大小？

- 为什么模型部署要开 EP？它有什么好处，为什么能提高吞吐？

## 30 秒回答

EP 把不同专家权重分布到不同 GPU，token 经 router 选择专家后分发激活，计算后返回并合并。它分摊专家权重显存，并让不同专家并行计算；释放的显存可用于 KV Cache 和更大并发。更大有效 batch 让专家收到更多 token，提高权重复用和 GEMM 效率。但吞吐收益需要足够流量、合理负载均衡和高效通信，不能只把 EP size 调大。

## 深入解释

### 英伟达五面补充

- 本次题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，AI Infra 与推理系统第 9 题；2026-11-05 收录，实际面试日期与具体轮次未提供。
- 本次原题：为什么要大 EP、大 DP？大 EP 开多大合适？固定单卡的 batch size 为 16，以 DeepSeek 为例，计算 EP 大小？

本补充显式选用 DeepSeek-V3，不把“DeepSeek”泛化为所有版本。官方配置有 256 个 routed experts，每 token 激活 8 个 routed experts；shared expert 不放入该路由均值分母。

假设 EP 组 P 个 ranks，每个 rank 每步贡献 16 个 tokens，均匀路由：

$$T=16P,\qquad M_e=\frac{Tk}{E}=\frac{16P\times8}{256}=\frac P2$$

| P | 每 expert 平均 tokens |
|---|---:|
| 8 | 4 |
| 32 | 16 |
| 64 | 32 |
| 128 | 64 |

若只是示例地要求 M_e=64，可推得 P=128；目标 64 必须来自 GEMM 性能/延迟分析，不能当作模型固有常量。这里的 batch=16 指每 rank 实际贡献 token 数；若是 TP 复制视图，不能重复计数。

固定单卡 batch 时，扩 P 同时扩大全局 batch；固定全局 T 时，M_e=Tk/E 不随 P 自动增大。目标部署还要满足权重容量、路由波动、网络拓扑、A2A、小消息和延迟约束。因此题设不足以唯一确定最优 EP。

与 [[如何用 Roofline 推导 GEMM 与 MoE 的临界 token 数|Roofline 临界 token 推导]] 联合使用时，先估目标 M_e，再推并行规模，最后压测。

### 从容量到吞吐的因果链

1. MoE 每个 token 只激活少量专家，但部署仍需保存所有专家权重；EP 让各卡仅保存部分专家。
2. 不同专家在不同 GPU 上处理 token，利用多卡算力和显存带宽。
3. 在没有其他容量限制时，权重占用下降可扩大 KV Cache 或运行时空间，承载更多请求。
4. 汇聚更多请求的 token 后，单专家 GEMM 的 M 维更大，改善权重复用、tile 利用率并摊薄调度成本。

设全局本步 token 数为 B、专家数为 E、每个 token 激活 k 个专家。均匀路由的简化推导为：

$$M_{expert}\approx\frac{Bk}{E}$$

只扩大 EP group 而不改变全局 B，不会自动增大每个专家的 token 数；低流量下小 expert batch 的问题仍可能存在。

### Wide-EP 与 DP Attention

对 MLA + MoE，Wide-EP 常用独立 Attention DP 副本维护不同请求的 KV Cache，同时共享跨卡专家池。它可减少某些普通 TP 实现中的 latent KV 重复，提高有效缓存容量和并发；这不是任意 MoE 或任意 TP 布局的统一结论。

### 收益的代价

- Dispatch/combine 的逻辑通信接近 all-to-all；具体后端可采用不同实现。
- 热门专家和 rank 不均衡决定慢尾，同步让快 rank 等慢 rank。
- EP 越宽，跨节点通信、小消息、碎片化和同步成本可能越突出。
- 配套手段包括 EPLB、grouped GEMM、通信计算重叠；不能把整套优化的收益全部归因于 EP。
- 评估应同时比较相同延迟约束下的总吞吐、每卡吞吐与成本，区分“增加卡数”和“提高效率”。

## 面试官可能追问

- 与 [[MoE 的专家并行 EP 如何工作，核心通信与系统难点是什么|EP 基础通信机制]] 有什么关系？
- 为什么 EP size 增大不保证每个专家 batch 增大？
- [[Grouped GEMM 的 contiguous 与 masked 布局及 tile 调度有什么区别|Grouped GEMM]] 如何处理动态 token 数与尾块？

## 关联知识

- [[../../../../wiki/concepts/Expert Parallelism|Expert Parallelism]]
- [[../../../../wiki/concepts/Wide Expert Parallelism|Wide Expert Parallelism]]
- [[../../../../wiki/concepts/DP Attention|DP Attention]]
- [[../../../../wiki/concepts/Expert Parallel Load Balancing|EPLB]]

## 参考来源与待核实

- [vLLM：Large Scale Serving with Wide-EP](https://vllm-project.github.io/2025/12/17/large-scale-serving.html)：Wide-EP、MLA KV 容量与配套优化。
- 专家平均 token 数为均匀路由条件下的机制推导；实际路由有偏斜。最优 EP size 和实际吞吐需绑定模型、GPU、互联、流量及 backend 实测；本次未运行 benchmark。

### 本次补充依据与边界

- [DeepSeek-V3 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)。
- P/2 为均匀路由条件下的平均值，不是每个 expert 的实际 token 数；未运行 benchmark。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]

- [[../sets/10月10号面试遇到的问题|10月10号面试遇到的问题]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
