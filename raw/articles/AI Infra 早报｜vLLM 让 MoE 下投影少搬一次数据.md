---
title: "AI Infra 早报｜vLLM 让 MoE 下投影少搬一次数据"
source: "https://mp.weixin.qq.com/s/MxjkwKiI0M3n2FCcoWoJWQ"
author:
  - "[[Ethan]]"
published:
created: 2026-11-09
description: "vLLM 针对 GLM 5.3 与 H200 的 FP8 MoE 下投影调整数据布局，试图减少显存搬运；FlashInfer 与 LangGraph 也带来新的硬件和 Agent 能力边界。"
tags:
  - "clippings"
---
Ethan Miracle Farms *2026年11月5日 08:37*

Miracle *Farms* BRIEF · DAILY

![题图](https://mmbiz.qpic.cn/sz_mmbiz_png/bPXzKRYb7HstDicLK5m8ky4IfRoRFZM97aiav6RtQGbZPt4FA4UjvonlBFQ5Mmp29tXCqicXplYgvKOICpUtsQIpBYQDicEyLgnCD4xngzabKT8/640?from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

> 当 MoE 模型越来越依赖低精度和异步搬运，真正影响吞吐的可能不是专家计算本身，而是结果能不能顺畅交给下一个计算步骤。今天这篇早报解释 vLLM 在 GLM 5.3、H200 上改了什么，以及这组收益还缺哪些验证。

## 先看三条信号

**推理｜已合并，尚未证明进入稳定版。** vLLM 调整 GLM 5.3 在 H200 上进行 FP8 MoE 下投影时的数据顺序，并把激活、乘法与量化合并到更紧凑的执行路径中。PR 声称端到端吞吐提升 6.0%–7.6%，但完整 batch、序列长度和基线条件仍未说明。\[1\]

**推理｜v0.7.3 已发布。** FlashInfer 为 Rubin 增加 MTP 与 NVFP4 MLA decode 支持，为 Blackwell 加速专家并行 MoE decode，并在 Hopper 增加 MXFP4 与 NVFP4 experts。 **这些能力仍需按具体 GPU、模型和性能矩阵核对。** \[2\]

**Agent｜PR 已合并，SDK 版本待确认。** LangGraph Python SDK 增加 thread state 与 history 的字段选择能力，调用方可以只读取 `values.messages` 等指定路径。 **服务端兼容关系和真实负载收益仍没有给出。** \[3\]

## 只展开一件事：MoE 下投影为什么还会浪费显存带宽

MoE 会把不同 token 分发给不同专家。专家计算结束后，结果还要进入下投影；如果中间张量的排列方式不适合下一步读取，GPU 就得先整理数据，激活和权重也难以按连续块搬运。 **专家计算完成，并不代表下一个计算步骤已经能顺畅接住输出。**

vLLM 这次改动让同一专家产生的数据继续排在一起，也就是保持 expert-sorted 顺序。这样，下投影可以使用 TMA——由 GPU 硬件协助异步搬运规则张量块的路径——同时读取激活和权重。原本分散在不同阶段的 SiLU、乘法与 FP8 量化，也被收进更紧凑的融合路径，目标是减少数据重排、显存读取和 kernel launch 的开销。

这条路线的适用条件很具体：PR 针对 GLM 5.3、H200 和 FP8 MoE 下投影，作者报告相对原有实现有 6.0%–7.6% 的端到端吞吐提升。由于材料没有列出完整负载、并行配置和基线版本， **这组数字只能视为该 PR 的测试声明，不能直接外推到其他模型或 GPU。**

布局优化也换来了更专用的 kernel。模型形状、量化格式或硬件代际变化后，收益可能需要重新验证，维护和迁移成本也会增加。对服务端而言，真正值得关注的是这项改动进入哪个 release，以及在真实 batch 和序列长度下是否仍能减少下投影的搬运时间。

## 接下来关注

优先核对 vLLM #64760 的完整 benchmark 条件和后续 release 收录状态，再在目标 GPU、模型和 batch 上比较端到端吞吐、显存占用与下投影 kernel 时间。FlashInfer v0.7.3 也需要确认实际是否选中了对应硬件 kernel。 **在这些条件补齐前，今天的信号只能表述为按硬件细化低精度路径，不能升级为普遍性能结论。**

> 一句话结论： **MoE 推理的瓶颈不只在专家计算，也在结果能否少搬一次数据；vLLM 的收益仍需按完整条件复测。**

更多完整证据和技术细节，请点击文末“阅读原文”。

## 参考

\[1\] vLLM PR #64760：https://github.com/vllm-project/vllm/pull/64760

\[2\] FlashInfer v0.7.3 Release：https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.3

\[3\] LangGraph PR #9568：https://github.com/langchain-ai/langgraph/pull/9568

Less hype · more systems.

MiracleFarms · BRIEF · DAILY

Briefs · 目录

阅读原文