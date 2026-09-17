# MTP 与 DSpark 的基本思路和区别是什么？

- 整理状态：已整理
- 题目出处：用户提供的一面面经（SGLang / CUDA 系统专题）

## 30 秒回答

MTP 是让额外预测模块一次提出多个未来 token，再由 target model 验证，以减少串行 decode 步数；具体模型可能采用多个预测头或顺序 MTP module，不能一概而论。DSpark 是并行 drafter 路线上的半自回归方案：加入轻量顺序修正和 conditional confidence，再由 hardware-aware scheduler 根据候选存活概率与硬件吞吐曲线决定送多少 token 给 target 验证。前者描述一类多 token 预测机制，后者是具体的 drafter 与调度设计。

## 面试官可能追问

- target model 如何验证，greedy 与 sampling 的接受规则有何不同？
- draft 越长为什么不一定越快？
- speculative KV 如何预留、提交和回滚？

## 关联知识

- [[../../../../wiki/concepts/Multi-Token Prediction|Multi-Token Prediction]]
- [[../../../../wiki/concepts/MTP Drafter|MTP Drafter]]
- [[../../../../wiki/concepts/DSpark|DSpark]]
- [[../../../../wiki/concepts/Speculative Decoding|Speculative Decoding]]

## 参考来源与待核实

- [[../../../../wiki/concepts/DSpark|DSpark 概念页]]
- [[../../../../wiki/concepts/MTP Drafter|MTP Drafter 概念页]]
- DSpark 的论文机制已经在 wiki 中核对；线上部署数字仍缺少完整生产配置，不作为本题答案依据。

## 所属题单

- [[../sets/SGLang与CUDA系统一面|SGLang 与 CUDA 系统一面]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
