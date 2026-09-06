# Nsight Systems 与 Nsight Compute 如何配合定位瓶颈？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 本页从历史原稿提炼；本轮未逐题重新核验技术事实或运行代码，不表示答案已通过事实复核。

## 其他问法

- 如何配合使用 Nsight Systems 与 Nsight Compute 定位 GPU 性能瓶颈？
- GPU 性能优化中，Nsight 工具链如何使用？应关注哪些计算效率与带宽效率指标？
- DCGM/NVML 不够时怎么分析多卡性能？
- Nsight Systems 与 Nsight Compute 有何区别，应如何联合定位瓶颈？
- 什么时候先用 nsys，什么时候直接用 ncu？
- 如何判断瓶颈是 launch、memory 还是 kernel 本身？
- 慢 kernel 如何排查 occupancy、访存与同步问题？

## 30 秒回答

先用稳定基准复现，再用 Nsight Systems 看全局时间线，定位 CPU 发射、拷贝、同步、通信和各 rank 差异。锁定热点后用 Nsight Compute 看占用率、管线利用率、停顿、DRAM/L2、合并访问及共享内存冲突；常驻监控不能替代这两层诊断。

## 深入解释

沿文末原稿章节阅读完整机制、推导或代码；本页保留短答，不复制长篇正文。

## 关联知识

- [[../../../../wiki/concepts/Profiling|Profiling]]

## 参考来源与待核实

- [[../../算子与GPU优化、推理优化补充#8. GPU 性能优化中，Nsight 工具链如何使用？应关注哪些计算效率与带宽效率指标？|算子与GPU优化、推理优化补充]]
- [[../../量化剪枝推理瓶颈Nsight与异构集群面试整理#Nsight：DCGM/NVML 不够时怎么分析多卡性能|量化剪枝推理瓶颈Nsight与异构集群面试整理]]
- [[../../面试经验#3. Nsight Systems 和 Nsight Compute 怎么用|面试经验]]

- 待核实 / 原稿边界：两份 Nsight 内容已聚合，不拆 Systems/Compute 的比较与协同工作流。具体指标名称随 GPU 架构和 Nsight 版本变化；原稿命令是示例，多 rank 输出命名、采样权限与采集范围仍需按实际环境验证，未执行。 原稿命令为示例，未记录工具版本、采样权限与 profiler 对运行时间的影响；本次未执行 profiling。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

## 所属题单

- [[../sets/量化与性能|量化与性能]]
- [[../README|秋招问题汇总]]
