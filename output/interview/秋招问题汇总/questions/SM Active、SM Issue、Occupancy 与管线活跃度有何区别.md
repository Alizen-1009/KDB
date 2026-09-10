# SM Active、SM Issue、Occupancy 与管线活跃度有何区别？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 本页从历史原稿提炼；本轮未逐题重新核验技术事实或运行代码，不表示答案已通过事实复核。

## 其他问法

- SM Active、SM Issue、Occupancy 和管线活跃度分别说明什么？
- Warp 和 SM Activity / SM Active / SM Occupancy 的关系是什么？
- Occupancy 高是否代表 GPU 性能好？

## 30 秒回答

Occupancy 是驻留 warp 占硬件上限的比例；SM Active 看是否有 warp 在运行，SM Issue 看是否发射指令，管线活跃度看指令落到哪里。高 Active、低 Issue 可能在等待；低占用率也可能因大分块和良好复用而高效，不能直接当性能分数。

## 深入解释

沿文末原稿章节阅读完整机制、推导或代码；本页保留短答，不复制长篇正文。

## 关联知识

- [[../../../../wiki/concepts/Profiling|Profiling]]
- [[../../../../wiki/concepts/GPU执行模型|GPU执行模型]]
- [[../../../../wiki/concepts/Occupancy|Occupancy]]

## 参考来源与待核实

- [[../../多卡GPU监控与SM执行模型面试整理#5. Warp 和 SM Activity / SM Active / SM Occupancy 的关系|多卡GPU监控与SM执行模型面试整理]]

- 待核实 / 原稿边界：四类指标作为整体比较保留。原稿 64 个 warp 上限、驻留 32 个而得到 50% 的例子仅为假设示例，不是所有 SM 的规格；采样、聚合口径与 metric 定义需按工具和硬件核对。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

## 所属题单

- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
