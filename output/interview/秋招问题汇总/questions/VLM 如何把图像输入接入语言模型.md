# VLM 如何把图像输入接入语言模型？

- 整理状态：已整理
- 题目出处：用户提供的 SDE / 强化学习岗位面试回忆

## 30 秒回答

典型生成式 VLM 有 vision encoder、connector 和语言模型。以 LLaVA 为例，图像 patch 经过视觉编码器，视觉特征投影到 LLM 隐维，再与文本 embedding 一起输入自回归 decoder。

## 深入解释

视觉编码器负责空间特征，projector/merger/resampler 对齐维度并可压缩视觉 token。另一类融合方式是由文本状态 cross-attend 视觉特征。训练通常先做图文对齐，再做多模态指令微调；生成任务只在有效答案 token 计算语言模型 loss。分辨率或视频帧数增大会推高视觉编码、prefill 和 KV 开销；可用动态分辨率、patch merge 或帧采样。具体位置编码和冻结策略按模型区分。

## 关联知识

- [[../../../../wiki/entities/Qwen VL|Qwen VL]]

## 参考来源与待核实

- [参考 1](https://arxiv.org/abs/2304.08485)

## 所属题单

- [[../sets/SDE与强化学习岗位面经|SDE与强化学习岗位面经]]
- [[../README|秋招问题汇总]]
