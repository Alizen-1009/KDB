# Tensor Core 与 CUDA Core 活跃度如何区分？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 本页从历史原稿提炼；本轮未逐题重新核验技术事实或运行代码，不表示答案已通过事实复核。

## 其他问法

- Tensor Core Active 与 CUDA Core Active 应如何区分？
- Tensor Core Active 和 CUDA Core Active 怎么解释？

## 30 秒回答

Tensor Core Active 关注矩阵乘加管线，常对应投影、MLP 和大注意力计算。CUDA Core Active 更像口语统称，普通算术、地址计算、逐元素和转换要分别看 FP32、ALU、SFU、LSU 等管线，不能将它当成跨架构统一的单一指标。

## 深入解释

沿文末原稿章节阅读完整机制、推导或代码；本页保留短答，不复制长篇正文。

## 关联知识

- [[../../../../wiki/concepts/Profiling|Profiling]]
- [[../../../../wiki/concepts/GPU执行模型|GPU执行模型]]

## 参考来源与待核实

- [[../../多卡GPU监控与SM执行模型面试整理#4. Tensor Core Active 和 CUDA Core Active 怎么解释？|多卡GPU监控与SM执行模型面试整理]]

- 待核实 / 原稿边界：比较题保持整体。原稿明确没有统一、跨架构固定的 CUDA Core Active 指标；列举的 FP16/BF16/TF32/FP8/INT8 能力与活跃度分母需结合具体 GPU 和工具解释。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

## 所属题单

- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
