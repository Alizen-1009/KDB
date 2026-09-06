# CUDA Graph 的加速原理与 capture-replay 约束？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 本页从历史原稿提炼；本轮未逐题重新核验技术事实或运行代码，不表示答案已通过事实复核。

## 其他问法

- CUDA Graphs 为什么能加速，capture 和 replay 有哪些关键约束与常见陷阱？
- CUDA Graphs 为什么快，以及为什么难用？

## 30 秒回答

CUDA Graphs 将稳定 GPU 工作流提前捕获，重放时一次提交整段执行，减少 CPU launch 和 GPU 空洞。难点在于路径稳定、API 可捕获，以及参数地址和生命周期不被破坏；动态控制流、旁路线程分配、残留 autograd 引用和非原地更新都可能出问题。

## 深入解释

沿文末原稿章节阅读完整机制、推导或代码；本页保留短答，不复制长篇正文。

## 关联知识

- [[../../../../wiki/concepts/Expert Parallelism|Expert Parallelism]]
- [[../../../../wiki/concepts/CUDA Graph 执行模式|CUDA Graph 执行模式]]

## 参考来源与待核实

- [[../../面试经验#7. CUDA Graphs：为什么快，以及为什么难用|面试经验]]

- 待核实 / 原稿边界：五个“经典坑”和“最低成本的入场方式”属于本题答案细节，不拆新题。GTC 2026 笔记中的 LoRA 70B/512 卡约 2.75x、PyTorch 2.12 Conditional Node、check_input_liveness=True 均缺独立核实。thread_local、关闭 autograd 多线程、del loss、设备张量学习率及 torch.cond 修法依赖版本和调用场景；torch.compile(reduce-overhead) 的自动捕获/回退说法也需核实，不能当成保证。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

## 所属题单

- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
