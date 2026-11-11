---
type: source
source_kind: 论文
topic: 并行与分布式
updated: 2026-11-11
---

# DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72

## 来源信息

- 标题：DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72
- 作者：Wanqian Li、Jintao Peng、Zongfei Jing 等，NVIDIA
- 日期：2026-05-13；arXiv:2604.01621v2（2026-05-12）
- 类型：论文
- 原始文件：`raw/papers/DWDP.pdf`

## 2-3 条核心摘要

- [[../concepts/DWDP]] 把 attention 权重复制到各 rank，将 MoE 专家权重分布存储于同组 GPU；每个 rank 在本地处理自己的请求，按层异步拉取缺失的远程专家，避免推理关键路径上的跨 rank collective 同步。
- 为降低权重搬运成本，作者通过 copy engine 的 P2P 拉取与计算重叠、double buffering、支持多权重缓冲区的 grouped GEMM，以及切片轮转减轻源端争用；计算与通信仍会争用共享资源和功耗预算。
- 在 GB200 NVL72、TensorRT-LLM、DeepSeek-R1 NVFP4 的 PD 分离实验中，DWDP 用于 context 侧。作者报告在 8K/1K、20–100 TPS/user 范围，相近用户吞吐下端到端输出 TPS/GPU 平均提高 8.8%；该收益不能忽略 TTFT 或外推为普适结果。

## 值得关注的论断

- 论文的理想 roofline 模型对 batch size 1 的 DWDP4 与 DEP4 预测约 16K ISL 开始有利；该阈值不是普适门槛，较大 batch 可延长计算窗口。模型假设负载均衡，也未计入通信计算干扰（第 4–5 页）。
- context-only 消融中，ISL 为 16K、MNT 为 32768 时，随着输入长度标准差从 0 增至 4096，TPS/GPU speedup 从 1.09× 增至 1.15×。同一论文中的 TTFT 与 TPS/GPU 取决于配置和排队（表 3）。
- 端到端对照的中位 TTFT 均上升：20–30 TPS/user 点从 2538 ms 增至 8314 ms；170–180 TPS/user 的 TPS/GPU speedup 为 0.97×。作者将低 TPS/user 的较大 TTFT 增幅部分归因于减少 context GPU 后排队增加；端到端结果默认未包含争用缓解优化（表 5–6）。
- 附录的受控实验显示短时重叠下 attention kernel 时间为基线的 1.226×、频率为 0.798×；作者据此认为该实验中功耗引发的降频是主要干扰，而非宣称所有 GPU 或负载均如此（表 7）。

## 关键概念

- [[../concepts/DWDP]]
- [[../concepts/DP Attention]]
- [[../concepts/Expert Parallelism]]
- [[../concepts/PD分离]]

## 相关实体

- [[../entities/TensorRT-LLM]]
- [[../entities/DeepSeek-AI]]

## 与现有 wiki 的关系

- 为 DP Attention + EP 的 token dispatch 路径补充按层预取专家权重的对照；DWDP 不取消远程数据依赖。
- 为 PD 分离增加 context 效率与整体 TTFT/资源配比之间的具体案例。
- 已检查相关页面，未发现直接矛盾；已有结论与本文实验结果的适用条件不同。

## 待确认

- TensorRT-LLM 上游集成状态和当前版本性能：论文基于 commit `3a89495`，称 PR #12136 正在集成，不应视为当前状态。
- 其他模型、拓扑、精度、流量和 TTFT SLO 下的收益需要独立测量。
