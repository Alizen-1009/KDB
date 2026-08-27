---
type: map
topic: 训练与 Scaling
---

# 训练与 Scaling

## 导读

先算清预算怎么花，再谈工程手段。

- **预算分配**：[[../concepts/Scaling Laws|Scaling Laws]] → [[../concepts/Chinchilla Scaling|Chinchilla Scaling]] → [[../concepts/数据缩放定律|数据缩放定律]] → [[../concepts/Critical Batch Size|Critical Batch Size]]
- **省显存换算力**：[[../concepts/重计算|重计算]]、[[../concepts/混合精度训练与推理|混合精度训练与推理]]

并行策略在 [[并行与分布式]]，这里只放规模与预算侧。

Qwen3.8-Flash-Next 的训练配方可按 [[../concepts/Muon Optimizer|Muon Optimizer]] → [[../concepts/Scaling Laws|Scaling Laws]] → [[../concepts/Critical Batch Size|Critical Batch Size]] 阅读，注意 batch/LR 与稳定性结论都绑定特定模型规模和 stress-test 设置。

<!-- BEGIN AUTO：以下由 scripts/update_index.py 生成，改动会被覆盖 -->

## 概念（7）

- [[../concepts/Chinchilla Scaling|Chinchilla Scaling]]
- [[../concepts/Critical Batch Size|Critical Batch Size]]
- [[../concepts/Muon Optimizer|Muon Optimizer]]
- [[../concepts/Scaling Laws|Scaling Laws]]
- [[../concepts/数据缩放定律|数据缩放定律]]
- [[../concepts/混合精度训练与推理|混合精度训练与推理]]
- [[../concepts/重计算|重计算]]

## 实体（1）

- [[../entities/Stanford CS336|Stanford CS336]]

## 来源（1）

- [[../sources/斯坦福CS336 Lecture 9 - Scaling laws basics|斯坦福CS336 Lecture 9 - Scaling laws basics]]
