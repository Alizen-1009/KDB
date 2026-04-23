# 斯坦福CS336 Lecture 10 - Inference systems and optimization

## 来源信息

- 官方课程：Stanford CS336: Language Modeling from Scratch
- 官方课程仓库：https://github.com/stanford-cs336/spring2025-lectures
- 官方脚本路径：https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_10.py
- 讲师：Tatsu Hashimoto
- 课程时间：Spring 2025
- 原始类型：可执行课程讲稿

## 原始说明

- 这一讲聚焦于固定模型下的推理 workload：用户已经不再训练模型，而是在真实请求流里追求更好的延迟、吞吐和单位成本。
- 讲稿主线是：
  - 为什么推理天然 memory-bound
  - 如何通过架构、量化、剪枝、推测解码等办法减少推理成本
  - 面对动态请求流时，系统如何通过 continuous batching 和 PagedAttention 提高利用率
- 这一讲和 Stanford Lecture 5/6/7/8 的关系非常强：
  - Lecture 5/6 给硬件和 kernel 直觉
  - Lecture 7/8 给通信和并行直觉
  - Lecture 10 则把这些系统直觉落到服务型推理上

## 讲义结构

### 1. 推理 workload 本身

- 推理的两个阶段：
  - `prefill`：对 prompt 做并行编码
  - `generation / decode`：逐 token 自回归生成
- 核心结论：
  - prefill 更接近 compute-limited
  - generation 更接近 memory-limited
- 讲稿用 arithmetic intensity 分析了为什么：
  - MLP 层在 generation 阶段仍然依赖 batch 才可能吃满算力
  - attention 在 generation 阶段几乎注定 memory-bound

### 2. KV cache 与推理延迟/吞吐

- KV cache 是推理的基础设施，因为它把复杂度从“每步重算全部历史”变成“复用历史状态”。
- 但 KV cache 同时也是内存压力的主要来源，因此很多优化本质上是在缩小、共享或更高效地管理 KV cache。
- 讲稿把 latency / throughput 的核心约束直接写成参数大小 + KV cache 大小 + memory bandwidth 的函数。

### 3. 有损 shortcut

- 讲稿讨论了几类会改变模型表示或近似精度的办法：
  - GQA / MLA / CLA / local attention：减少 KV cache 或注意力代价
  - quantization：减少权重与激活精度
  - model pruning：裁剪模型，再修复
  - state-space / diffusion 这类替代架构：尝试从根上改变推理复杂度

### 4. 无损 shortcut

- speculative sampling / speculative decoding 利用“检查比生成便宜”的非对称性：
  - 小 draft model 先猜
  - 大 target model 并行检查并选择接受/拒绝
- 讲稿强调它的关键魅力是：可以在数学上保持对 target model 的精确采样分布。

### 5. 动态 workload 的系统优化

- 线上推理请求的困难不是只有单个序列多长，而是请求到达时间不同、长度不同、前缀不同。
- continuous batching 通过 iteration-level scheduling 解决“不能等整批请求都完成再加新请求”的问题。
- PagedAttention 通过分页管理 KV cache 解决内外部碎片、前缀共享和 copy-on-write 等问题。

## 从讲义中抽出的高信号结论

- 推理系统优化的核心前提是承认 generation 阶段天然 memory-bound，很多优化都在围绕这个现实做文章。
- KV cache 是推理加速的前提，同时也是很多推理系统复杂性的来源。
- 连续批处理、分页和推测解码这些系统技术，本质上是在把操作系统、编译器和体系结构的思路移植到 LLM 推理里。
