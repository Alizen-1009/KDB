# 斯坦福CS336 Lecture 9 - Scaling laws basics

## 来源信息

- 官方课程：Stanford CS336: Language Modeling from Scratch
- 官方课程仓库：https://github.com/stanford-cs336/spring2025-lectures
- 官方讲义 PDF：https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf
- 讲师：Tatsu Hashimoto
- 课程时间：Spring 2025
- 原始类型：课程讲义

## 原始说明

- 这一讲从系统实现问题切换到模型设计与资源分配问题：给你固定算力、固定时间和很多 GPU，模型应该做多大、吃多少数据、训练多少步。
- Stanford 用 scaling laws 把“大模型设计”变成一类可通过小规模实验外推的大致可预测问题。
- 讲义主线覆盖三块：
  - 数据规模与性能的关系
  - 模型规模、数据规模与性能的联合关系
  - 固定 compute 预算下的大模型设计权衡

## 讲义结构

### Part 1: Scaling laws 的历史与直觉

- 从更早的 sample complexity 和经验型 data scaling 讨论进入神经网络 scaling。
- 讲义强调 scaling law 的核心形式是：在 log-log 图上接近线性，也就是 power law。
- 数据 scaling 的一个关键直觉是：误差常以多项式速度衰减，例如 `1 / n^alpha`。

### Part 2: Neural / LLM scaling behaviors

- 讲义先讨论 `data vs performance`：
  - 数据量增长通常带来单调的性能改进
  - 对语言模型来说，loss 与 dataset size 在 log-log 图上近似直线
- 然后讨论 `model/data/hyperparameter`：
  - 架构选择、优化器、深宽比、batch size 等超参数在大模型上的效果，可以先通过小模型实验拟合外推
  - `critical batch size` 表示 batch 增大到某个点之后收益快速递减

### Part 3: Joint scaling and compute tradeoffs

- 讲义引入模型大小和数据大小的联合 scaling：
  - 给定资源，不能只问“更大模型还是更多数据”，而要同时优化两者
- 接着转向 compute-optimal 设计：
  - 固定 FLOPs 预算下，是“大模型少训练”还是“小模型多训练”
- 讲义最后引出 Chinchilla：
  - Kaplan 风格 joint fit 与 Chinchilla 风格 compute-optimal fit 在常数和训练策略上有差异
  - 但它们共同说明：token / parameter 比例和学习率调度会显著影响“最优缩放”结论

## 从讲义中抽出的高信号结论

- Scaling laws 的工程价值不在于“绝对精确”，而在于它让大模型设计从拍脑袋调参，转向基于小规模实验的可预测外推。
- 数据、模型和 compute 三者不是独立优化问题，而是一个联合资源分配问题。
- Chinchilla 的重要性不只是“20 tokens per parameter”这类口号，而是它把“固定训练算力下的最优配比”明确提出并实证化了。
