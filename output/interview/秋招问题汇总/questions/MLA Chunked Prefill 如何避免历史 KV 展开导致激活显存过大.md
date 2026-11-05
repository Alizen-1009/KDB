# MLA Chunked Prefill 如何避免历史 KV 展开导致激活显存过大？

- 整理状态：已整理
- 题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，AI Infra 与推理系统第 4 题；2026-11-05 收录，实际面试日期与具体轮次未提供。

## 原题与条件

Chunk Prefill 在 MLA 中，KV load 之后做 proj，如果 KV 很大导致激活值很大怎么办？

## 30 秒回答

先澄清“激活大”是数值幅度还是张量占用。若指显存，当前 query chunk 小不代表历史 KV 短；一次将全部 latent KV 展开成 per-head K/V 会产生大中间张量。可以分块展开并在线合并 Attention，或走矩阵吸收路径，再结合 projection/Attention 融合，限制峰值 workspace。

## 深入解释

本答案按“激活张量占用过大”解释；若原题指溢出或数值异常，需要另查量化、缩放、归约精度和数值范围，不能用切块显存答案代替。

MLA 原缓存约为 [S,d_c] 的 latent 加位置部分。若恢复所有 heads 的 K/V，中间形状可能达到 [S,H,d_k+d_v]。Chunked Prefill 主要缩小本轮 Q，历史长度 S 仍可能很长。

- **历史 KV 分块解压**：一次恢复一个历史 tile，立即参与 Attention，随后释放；用 Online Softmax 合并各 tile。
- **矩阵吸收**：对 content key 使用 qᵀW_UKc=(W_UKᵀq)ᵀc；对 value 使用 ΣpⱼW_UVcⱼ=W_UVΣpⱼcⱼ。先在 latent 上算分数/聚合，避免物化完整历史 K/V；RoPE 部分按 decoupled 路径单独处理。
- **融合与缓冲复用**：将 projection 和 Attention 数据流衔接，减少展开结果写回 HBM，按生命周期复用 workspace。

吸收与展开路径的 GEMM 维度、FLOPs 和 kernel 效率不同。Prefill 不应无条件选吸收路径；要结合 query chunk、历史长度、dtype 和 backend 实测。DeepSeek 官方示例有 naive/absorb 两条代数路径，但不是生产融合 kernel 的性能证明。

## 关联知识

- [[../../../../wiki/concepts/MLA|MLA]]
- [[../../../../wiki/concepts/Chunked Prefill|Chunked Prefill]]
- [[../../../../wiki/concepts/Online Softmax|Online Softmax]]

## 参考来源与待核实

- [DeepSeek-V3 官方 MLA 示例](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)。
- 待确认：原题“激活值很大”所指语义；具体 backend 的峰值显存与性能未实测。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
